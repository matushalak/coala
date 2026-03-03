# author: Matúš Halák (@matushalak)
import torch
import torch.nn as nn
import torch.nn.functional as F

class CCModule(nn.Module):
    '''
    Contextual Contrasting RNN module
    '''
    def __init__(self, spatial_dims:tuple[int, int], FF_conv:nn.Conv2d, FB_conv:nn.Conv2d|None, 
                 LAT_ksize:tuple[int, int] = (3,3), activation_fn:nn.Module = nn.GELU(),
                 time_alpha:float|torch.Tensor = 1.0
                 ):
        super().__init__()
        assert FF_conv is not None, "Feedforward convolution layer (FF_conv) must be provided."
        self.dims_ = (FF_conv.out_channels, *spatial_dims)
        # Initialize convolution layers for feedforward, feedback, and lateral inhibition
        self.FF_conv = FF_conv # pre-trained feedforward convolution layer (fixed weights)
        self.FB_conv = FB_conv # pre-trained feedback convolution layer (fixed weights)
        # Lateral inhibition implemented as a convolution with fixed weights (sum over local neighborhood)
        self.LAT_conv = lambda y: F.conv2d(
            y, weight=torch.ones(size=(FF_conv.out_channels, FF_conv.out_channels, *LAT_ksize),requires_grad=False),
            padding = (LAT_ksize[0]//2, LAT_ksize[1]//2))
        
        # Define dynamic lambdas for individual synapses (3-compartment neuron)
        self.Lambda_FF = nn.Parameter(torch.randn(1, FF_conv.out_channels, *spatial_dims), requires_grad=False)
        
        if FB_conv is not None:
            self.Lambda_FB = nn.Parameter(torch.randn(1, FF_conv.out_channels, *spatial_dims), requires_grad=False)
        else:
            self.Lambda_FB = None
        
        self.Lambda_LAT = nn.Parameter(torch.randn(1, FF_conv.out_channels, *spatial_dims), requires_grad=False)

        # activation function (e.g., ReLU)
        self.activation_fn = activation_fn

        # (dt/tau) time alpha for neurons (TODO: this should probably be fine-tuned or learned with backprop per task)
        self.time_alpha = nn.Parameter(torch.as_tensor(time_alpha), requires_grad=False)

        # keep mask
        self.register_buffer("keep_mask", torch.ones(1, 1, *spatial_dims, dtype=torch.bool), persistent=False)

    
    def forward(self, x:torch.Tensor|None, context:torch.Tensor|None, Y_old:torch.Tensor|None,
                train:bool = False)->torch.Tensor:
        # Compute feedforward, feedback, and lateral inhibition contributions.
        ref = x if x is not None else (Y_old if Y_old is not None else context)
        if ref is None:
            raise ValueError("CCModule.forward needs at least one of x, context, or Y_old to infer batch/device.")

        if x is not None:
            y_FF = self.FF_conv(x, self.keep_mask)
        else:
            y_FF = torch.zeros((ref.shape[0], *self.dims_), device=ref.device, dtype=ref.dtype)
        y_FB = torch.zeros_like(y_FF)
        y_LAT = torch.zeros_like(y_FF)
        
        # Combine contributions with dynamic lambdas
        drive = (1+self.Lambda_FF) * y_FF 

        if Y_old is not None:
            y_LAT = self.LAT_conv(Y_old)
            drive -= (1+self.Lambda_LAT) * y_LAT

        if context is not None and self.FB_conv is not None and self.Lambda_FB is not None:
            y_FB = self.FB_conv(context)
            drive += (1+self.Lambda_FB) * y_FB

        # Apply activation function (drive term)
        drive = self.activation_fn(drive)

        # Evolution of Y
        if Y_old is None:
            Y = drive
        else:
            Y = Y_old + (drive - Y_old) / self.time_alpha
        
        # Local update in forward pass 
        # (TODO: this is a bit hacky, should probably be done in a separate update step or with a custom autograd function)
        if train:
            self.update(Y, y_FF, y_FB, y_LAT)
        return Y
    
    @torch.no_grad()
    def update(self, Y:torch.Tensor, 
               y_FF:torch.Tensor, y_FB:torch.Tensor, y_LAT:torch.Tensor,
               lr_FF:float = 3e-3, lr_FB:float = 2e-3, lr_LAT:float = 1e-3)->None:
        ''''
        Local update, leaks back to 0; average over batch
        '''
        self.Lambda_FF += lr_FF * (-self.Lambda_FF - (Y * y_FF).mean(dim=0, keepdim=True))
        self.Lambda_LAT += lr_LAT * (-self.Lambda_LAT + (Y * y_LAT).mean(dim=0, keepdim=True))
        if self.Lambda_FB is not None:
            self.Lambda_FB += lr_FB * (-self.Lambda_FB + (((1/(1+Y))) * (Y * y_FB)).mean(dim=0, keepdim=True))


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
