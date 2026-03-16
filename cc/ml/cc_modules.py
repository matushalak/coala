# author: Matúš Halák (@matushalak)
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Literal

# TODO: remove lateral inhibition
class LateralInhibition(nn.Module):
    """
    Lateral inhibition convolution with channel+spatial averaging kernel.

    Each output location/channel receives the average of all channels across the
    local spatial neighborhood.
    """

    def __init__(self, n_channels: int, kernel_size: tuple[int, int] = (3, 3)):
        super().__init__()
        kh, kw = kernel_size
        assert n_channels > 0, f"n_channels must be > 0, got {n_channels}."
        assert kh > 0 and kw > 0, f"kernel_size must be positive, got {kernel_size}."

        norm = float(n_channels * kh * kw)
        weight = torch.full((n_channels, n_channels, kh, kw), fill_value=1.0 / norm
                            )
        self.register_buffer("weight", weight, persistent=False)
        self.padding = (kh // 2, kw // 2)
        self.n_channels = n_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(x, weight=self.weight, padding=self.padding, 
                        # groups = self.n_channels
                        )


class LambdaModule(nn.Module):
    '''
    Module to learn dynamic lambdas for each synapse (feedforward, feedback, lateral)
    '''
    def __init__(self, channels:int, spatial_dims:tuple[int, int], learnable:bool = True,
                 init_Lambda:float|str = 0.0, 
                 init_lr:float = 1e-3, 
                 plus_one:bool = False,
                 learning_rule:Literal['hebbian', 'anti_hebbian', 'dampened_hebbian', 'pc'] = 'hebbian'):
        super().__init__()
        self.dims_ = (1, channels, *spatial_dims) # to enable batch broadcasting
        self.plus_one = plus_one
        self.init_Lambda = init_Lambda
        self.register_buffer("Lambda", self.initialize(self.init_Lambda), persistent=False)
        self.raw_lr = nn.Parameter(torch.logit(torch.tensor(init_lr)), requires_grad=False)
        # self.decay_alpha = nn.Parameter(torch.tensor(init_decay_alpha), requires_grad=False)
        self.learning_rule = learning_rule
        assert learning_rule in ('hebbian', 'anti_hebbian', 'dampened_hebbian', 'pc'
                                 ), 'Provided learning rule is not supported'
    
    def forward(self, y:torch.Tensor) -> torch.Tensor:
        if self.plus_one:
            return (1 + self.Lambda) * y
        else:
            return self.Lambda * y
    
    @torch.no_grad()
    def update(self, Y:torch.Tensor, ydrive:torch.Tensor)->None:
        # Local update, leaks back to 0; average over batch
        match self.learning_rule:
            case 'hebbian':
                self.Lambda = self.Lambda + self.lr * ((Y * ydrive).mean(dim=0, keepdim=True) - self.Lambda)
            case 'anti_hebbian':
                self.Lambda = self.Lambda - self.lr * ((Y * ydrive).mean(dim=0, keepdim=True) - self.Lambda)
            case 'dampened_hebbian':
                self.Lambda = self.Lambda + self.lr * (((1/(1+Y))*(ydrive)).mean(dim=0, keepdim=True) - self.Lambda)
            case 'pc':  
                self.Lambda = self.Lambda + self.lr * ((NotImplementedError).mean(dim=0, keepdim=True) - self.Lambda)
    
    @property
    def lr(self):
        ''''
        Learning rate for local update, constrained to (0, 1) via sigmoid.
        Also controls rate of decay back to 0 (leak) when no drive is present.
        '''
        return torch.sigmoid(self.raw_lr) * 0.99

    def initialize(
        self,
        init_value:float|str = 0.0,
        device:torch.device|None = None,
        dtype:torch.dtype|None = None,
    ) -> torch.Tensor:
        if isinstance(init_value, str):
            assert init_value == "random", "init_value must be 'random' or a float."
            return torch.rand(*self.dims_, device=device, dtype=dtype)
        else: # fill with float value
            return torch.full(self.dims_, init_value, device=device, dtype=dtype)
    
    def reset(self, ref_tensor:torch.Tensor|None = None):
        device = ref_tensor.device if ref_tensor is not None else self.raw_lr.device
        dtype = ref_tensor.dtype if ref_tensor is not None else self.Lambda.dtype
        self.Lambda = self.initialize(self.init_Lambda, device=device, dtype=dtype)

# TODO: updates should be precision-weighed (mostly use internal representations where sensory input noisy)
# TODO: probably better ways of doing lateral inhibition
class CCModule(nn.Module):
    '''
    Contextual Contrasting recurrent cell module
    '''
    def __init__(self, spatial_dims:tuple[int, int], FF_conv:nn.Conv2d, FB_conv:nn.Conv2d|None, 
                 LAT_ksize:tuple[int, int] = (3,3), activation_fn:nn.Module = nn.Identity(),
                 time_alpha:float|torch.Tensor | None = 0.04
                 ):
        super().__init__()
        assert FF_conv is not None, "Feedforward convolution layer (FF_conv) must be provided."
        self.dims_ = (FF_conv.out_channels, *spatial_dims)
        # Initialize convolution layers for feedforward, feedback, and lateral inhibition
        self.FF_conv = FF_conv # pre-trained feedforward convolution layer (fixed weights)
        self.FB_conv = FB_conv # pre-trained feedback convolution layer (fixed weights)
        self.LAT_conv = LateralInhibition(FF_conv.out_channels, LAT_ksize) # average kernel across channels+space.
        
        # Define dynamic lambdas for individual synapses (3-compartment neuron)
        self.Lambda_FF = LambdaModule(FF_conv.out_channels, spatial_dims, learnable=True, 
                                      init_Lambda=0.0, init_lr=5e-3, plus_one=True,
                                      learning_rule='anti_hebbian'
                                    #   learning_rule='hebbian'
                                      )
        
        if FB_conv is not None:
            self.Lambda_FB = LambdaModule(FF_conv.out_channels, spatial_dims, learnable=True, 
                                          init_Lambda=0.0, init_lr=5e-3, plus_one=True,
                                          learning_rule='dampened_hebbian'
                                          )
        else:
            self.Lambda_FB = None
        
        self.Lambda_LAT = LambdaModule(FF_conv.out_channels, spatial_dims, learnable=True, 
                                       init_Lambda=0.0, init_lr=2e-3,
                                       learning_rule='hebbian')

        # activation function (e.g., ReLU)
        self.activation_fn = activation_fn
        self.sigmoid = nn.Sigmoid()

        # (dt/tau) time alpha for neurons (TODO: this should probably be fine-tuned or learned with backprop per task)
        if time_alpha is None:
            time_alpha = 0.2
        self.time_alpha = nn.Parameter(torch.as_tensor(time_alpha), requires_grad=False)
        
        # TODO: parameters to learn with backprop learning
        # time alphas for each of the lambdas

        # LRs for each of the lambda local learning rules

        # keep mask
        self.register_buffer("keep_mask", torch.ones(1, 1, *spatial_dims, dtype=torch.bool), persistent=False)

    
    def forward(self, x:torch.Tensor|None, context:torch.Tensor|None, Y_old:torch.Tensor|None)->torch.Tensor:
        # Compute feedforward, feedback, and lateral inhibition contributions.
        ref = x if x is not None else (Y_old if Y_old is not None else context)
        assert ref is not None, "CCModule.forward needs at least one of x, context, or Y_old to infer batch/device."

        if x is not None:
            y_FF = self.FF_conv(x, self.keep_mask)
        else:
            y_FF = torch.zeros((ref.shape[0], *self.dims_), device=ref.device, dtype=ref.dtype)
        y_FB = torch.zeros_like(y_FF)
        y_LAT = torch.zeros_like(y_FF)
        
        # Combine contributions with dynamic lambdas
        drive = self.Lambda_FF(y_FF)

        if context is not None and self.FB_conv is not None and self.Lambda_FB is not None:
            y_FB = self.FB_conv(context,
                                # None 
                                y_FF
                                ) # y_FF is skip connection from SparK pretraining
            drive += self.Lambda_FB(y_FB)
            # drive /= 2

        # if Y_old is not None:
        y_LAT = self.LAT_conv(drive)
        drive -=  self.Lambda_LAT(y_LAT) # "PV cells"
        
        # Apply activation function (drive term)
        drive = self.activation_fn(drive)

        # Evolution of Y
        if Y_old is None:
            Y = drive
        else:
            Y = Y_old + self.time_alpha * (drive - Y_old)
        
        return Y, y_FF, y_FB, y_LAT # return all drives for local update
    
    @torch.no_grad()
    def update(self, Y:torch.Tensor, 
               y_FF:torch.Tensor, y_FB:torch.Tensor, y_LAT:torch.Tensor)->None:
        ''''
        Local update, leaks back to 0; average over batch
        '''
        # self.Lambda_FF.update(Y, y_FF)
        # self.Lambda_LAT.update(Y, y_LAT)
        # if self.Lambda_FB is not None:
        #     self.Lambda_FB.update(Y, y_FB)
        pass

    def reset_dynamic_state(self, ref_tensor:torch.Tensor|None = None)->None:
        self.Lambda_FF.reset(ref_tensor=ref_tensor)
        self.Lambda_LAT.reset(ref_tensor=ref_tensor)
        if self.Lambda_FB is not None:
            self.Lambda_FB.reset(ref_tensor=ref_tensor)

if __name__ == "__main__":
    # Example usage
    spatial_dims = (32, 32) # example spatial dimensions
    FF_conv = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
    FB_conv = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, padding=1)
    
    cc_module = CCModule(spatial_dims, FF_conv, FB_conv)
    input_tensor = torch.randn(1, 3, *spatial_dims) # example input
    hidden_tensor = torch.randn(1, 32, *spatial_dims) # example hidden state
    output = cc_module(input_tensor, hidden_tensor)
    print(output.shape)
