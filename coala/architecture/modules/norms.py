import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Literal
from coala.architecture.modules.utils import _masked_mean_and_var, _resolve_eps, _expand_keep_mask


# Sparse versions
# Layernorm
class SparseLayerNorm2d(nn.LayerNorm):
    def forward(self, x: torch.Tensor, keep_mask: torch.BoolTensor) -> torch.Tensor:
        reduce_dims = tuple(range(x.ndim - len(self.normalized_shape), x.ndim))
        mask, mean, var = _masked_mean_and_var(x, keep_mask, reduce_dims)
        eps = _resolve_eps(self.eps, x)
        y = (x - mean) / torch.sqrt(var + eps)
        if self.elementwise_affine:
            y = y * self.weight + self.bias
        return y * mask


# RMSNorm
class SparseRMSNorm2d(nn.RMSNorm):
    def forward(self, x: torch.Tensor, keep_mask: torch.BoolTensor) -> torch.Tensor:
        reduce_dims = tuple(range(x.ndim - len(self.normalized_shape), x.ndim))
        mask = _expand_keep_mask(x, keep_mask)
        count = mask.sum(dim=reduce_dims, keepdim=True).clamp_min(1.0)
        eps = _resolve_eps(self.eps, x)
        rms = torch.rsqrt((x.pow(2) * mask).sum(dim=reduce_dims, keepdim=True) / count + eps)
        y = x * rms
        if self.elementwise_affine:
            y = y * self.weight
        return y * mask



# Wrapper around 'layer-norm' type normalization (layer & rms)
class DenseNorm2d(nn.Module):
    def __init__(self, norm_type: str = "rmsnorm", *args, **kwargs):
        super().__init__()
        norm_type = norm_type.lower()
        if norm_type == "layernorm":
            self.norm = nn.LayerNorm(*args, **kwargs)
        elif norm_type == "rmsnorm":
            self.norm = nn.RMSNorm(*args, **kwargs)
        else:
            raise ValueError(f"Unsupported norm_type {norm_type!r}. Expected 'layernorm' or 'rmsnorm'.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x)


class SparseNorm2d(nn.Module):
    def __init__(self, norm_type: str = "rmsnorm", *args, **kwargs):
        super().__init__()
        norm_type = norm_type.lower()
        if norm_type == "layernorm":
            self.norm = SparseLayerNorm2d(*args, **kwargs)
        elif norm_type == "rmsnorm":
            self.norm = SparseRMSNorm2d(*args, **kwargs)
        else:
            raise ValueError(f"Unsupported norm_type {norm_type!r}. Expected 'layernorm' or 'rmsnorm'.")

    def forward(self, x: torch.Tensor, keep_mask: torch.BoolTensor) -> torch.Tensor:
        return self.norm(x, keep_mask)


    
# GRN
class GlobalResponseNorm(nn.Module):
    """
    Global response normalization (GRN) as used in ConvNeXtV2.
        GRN normalizes each channel by the global L2 norm across spatial dimensions, 
        with learnable scaling and bias.
    
    Assumes (B, C, H, W) input shape.
    """

    def __init__(self, n_channels: int, eps: float = 1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(1, n_channels, 1, 1)
                                #   *1e-3
                                  )
        self.beta = nn.Parameter(torch.zeros(1, n_channels, 1, 1))
        self.eps = eps

    def forward(self, x: torch.Tensor, keep_mask: torch.BoolTensor | None = None) -> torch.Tensor:
        if keep_mask is None:
            mask = 1.0
        else:
            mask = keep_mask.to(dtype=x.dtype)

        # L2 norm "pooling" across spatial dimensions.
        gx = torch.sqrt((x.pow(2) * mask).sum(dim=[2, 3], keepdim=True)+self.eps)
        # Competition across channels.
        nx = gx / (gx.mean(dim=1, keepdim=True) + self.eps)
        # Apply scaling and bias
        return self.gamma * (x * nx) + self.beta + x
    

class SparseGlobalResponseNorm(GlobalResponseNorm):
    def forward(self, x: torch.Tensor, keep_mask: torch.BoolTensor) -> torch.Tensor:
        return super().forward(x, keep_mask=keep_mask) * keep_mask.to(dtype=x.dtype)