import torch
import torch.nn as nn
from cc.ml.sparse_cnn_unet import FF_Conv2d, FB_Conv2d, SparseNormalizedFusion

class LocalIntegrationModule(nn.Module):
    def __init__(self, 
                 spatial_dims:tuple[int,int], 
                 FF_stream:FF_Conv2d, FB_stream:FB_Conv2d, 
                 Fusion_module:SparseNormalizedFusion, 
                 Mask_conv:nn.Conv2d,
                 time_alpha:float = 0.2):
        super().__init__()
        assert FF_stream is not None, "Feedforward stream must be provided."
        self.dims_ = (FF_stream.out_channels, *spatial_dims)
        self.time_alpha = nn.Parameter(torch.as_tensor(time_alpha), requires_grad=False)
        self.FF_stream = FF_stream
        self.FB_stream = FB_stream
        self.Fusion_module = Fusion_module
        self.Mask_module = nn.Sequential(Mask_conv, nn.Sigmoid())

    def forward(self, x:torch.Tensor|None, context:torch.Tensor|None, Y_old:torch.Tensor|None,
                keep_mask:torch.BoolTensor)->torch.Tensor:
        ref = x if x is not None else (Y_old if Y_old is not None else context)
        assert ref is not None, "LocalIntegrationModule.forward needs at least one of x, context, or Y_old to infer batch/device."
        if keep_mask is None:
            keep_mask = torch.ones((ref.shape[0], 1, *self.dims_[1:]), device=ref.device, dtype=torch.bool)
        # Feedforward contribution
        y_FF = self.FF_stream(x, keep_mask) if x is not None else torch.zeros((ref.shape[0], *self.dims_), device=ref.device, dtype=ref.dtype)
        # Feedback contribution & predicted precision
        y_FB = self.FB_stream(context, y_FF) if context is not None else torch.zeros_like(y_FF)
        precision = self.Mask_module(y_FB) # shape: (B, 1, H, W), values in [0, 1]
        hard_mask = precision.round().bool() # Keep if precision > 0.5, else discard
        # Precision-weighted combination
        y = precision * y_FF + (1 - precision) * y_FB
        # Pass through fusion (normalization) module 
        y = self.Fusion_module(y, hard_mask)

        return Y_old + self.time_alpha * (y - Y_old) if Y_old is not None else y, hard_mask
