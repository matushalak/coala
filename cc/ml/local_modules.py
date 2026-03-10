import torch
import torch.nn as nn

class LocalIntegrationModule(nn.Module):
    def __init__(self, 
                 n_channels:int, spatial_dims:tuple[int,int], 
                 FF_conv, FB_conv, Fusion_module, time_alpha:float = 0.2):
        super().__init__()
        assert FF_conv is not None, "Feedforward convolution layer (FF_conv) must be provided."
        self.dims_ = (FF_conv.out_channels, *spatial_dims)
        self.time_alpha = nn.Parameter(torch.as_tensor(time_alpha), requires_grad=False)
        self.FF_conv = FF_conv
        self.FB_conv = FB_conv
        self.Fusion_module = Fusion_module
        
    def forward(self, x:torch.Tensor|None, context:torch.Tensor|None, Y_old:torch.Tensor|None)->torch.Tensor:
        # Compute feedforward contribution.
        ref = x if x is not None else (Y_old if Y_old is not None else context)
        assert ref is not None, "LocalIntegrationModule.forward needs at least one of x, context, or Y_old to infer batch/device."
        y_FF = self.FF_conv(x) if x is not None else torch.zeros((ref.shape[0], *self.dims_), device=ref.device, dtype=ref.dtype)
        y_FB = self.FB_conv(context) if context is not None else torch.zeros_like(y_FF)