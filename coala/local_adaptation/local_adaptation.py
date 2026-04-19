import torch
import torch.nn as nn
from typing import Literal

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