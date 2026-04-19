# author: Matúš Halák (@matushalak)
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Literal


class CollapsedModule(nn.Module):
    '''
    Collapsed recurrent cell module
    '''
    def __init__(self, 
                 spatial_dims:tuple[int, int], 
                 FF_conv:nn.Conv2d, 
                 FB_conv:nn.Conv2d|None, 
                 local_block:nn.Module, 
                 activation_fn:nn.Module = nn.GELU(),
                 time_alpha:float|torch.Tensor | None = 0.08
                 ):
        super().__init__()
        assert FF_conv is not None, "Feedforward convolution layer (FF_conv) must be provided."
        self.dims_ = (FF_conv.out_channels, *spatial_dims)
        # Initialize convolution layers for feedforward, feedback, and lateral inhibition
        self.FF_conv = FF_conv # pre-trained feedforward convolution layer (fixed weights)
        self.FB_conv = FB_conv # pre-trained feedback convolution layer (fixed weights)
        self.local_block = local_block # local processing block (within-cortical column refinement of features)

        # activation function (e.g., ReLU)
        self.activation_fn = activation_fn

        # (dt/tau) time alpha for neurons in current layer
        self.time_alpha = nn.Parameter(torch.as_tensor(time_alpha), requires_grad=False)

        # keep mask
        self.register_buffer("keep_mask", torch.ones(1, 1, *spatial_dims, dtype=torch.bool), persistent=False)

    
    def forward(self, x:torch.Tensor, context:torch.Tensor, Y_old:torch.Tensor)->torch.Tensor:
        # Compute feedforward, feedback, and lateral inhibition contributions.
        assert x is not None, "CollapsedModule.forward needs x to infer batch/device."
        keep_mask = self.keep_mask.expand(x.shape[0], -1, -1, -1)

        # Feedforward drive / initial drives
        if x is not None:
            # NOTE: this should be only a spatial convolution, without activation or normalization
            y_FF = self.FF_conv(x, keep_mask)
        else:
            y_FF = torch.zeros((ref.shape[0], *self.dims_), device=ref.device, dtype=ref.dtype)
        y_FB = torch.zeros_like(y_FF)
        
        # Feedback drive
        if context is not None and self.FB_conv is not None:
            # NOTE: this should be only a spatial transposed convolution, without activation or normalization
            y_FB = self.FB_conv(context, y_FF)
            
        # Dynamically combine input streams
        pre_local_drive = self.activation_fn(y_FF + y_FB)

        # Apply local cortical column processing on neuronal activations (same spatial resolution)
        post_local_drive = self.local_block(pre_local_drive)

        # Leaky integration of local drive with previous state
        Y = ((1 - self.time_alpha) * Y_old) + (self.time_alpha * post_local_drive)
        
        return Y, y_FF, y_FB
    
    # NOTE: current update rules do NOT work
    @torch.no_grad()
    def update(self, 
               Y:torch.Tensor, y_FF:torch.Tensor, y_FB:torch.Tensor)->None:
        ''''
        Local update, leaks back to 0; average over batch
        '''
        # self.Lambda_FF.update(Y, y_FF)
        # if self.Lambda_FB is not None:
        #     self.Lambda_FB.update(Y, y_FB)
        # self.Lambda_LAT.update(Y, y_LAT)

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
    
    cc_module = CollapsedModule(spatial_dims, FF_conv, FB_conv)
    input_tensor = torch.randn(1, 3, *spatial_dims) # example input
    hidden_tensor = torch.randn(1, 32, *spatial_dims) # example hidden state
    output = cc_module(input_tensor, hidden_tensor)
    print(output.shape)
