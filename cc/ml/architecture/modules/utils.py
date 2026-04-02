import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Literal


def downsample_center_mask(
    keep_mask: torch.BoolTensor,
    out_size: tuple[int, int],
    stride: int = 2,
) -> torch.BoolTensor:
    """
    Downsample activity by selecting center-aligned positions (slice-based).
    For k=3, p=1, s=2 convs, output (i, j) aligns to input (2i, 2j).
    """
    out_h, out_w = out_size
    return keep_mask[:, :, ::stride, ::stride][:, :, :out_h, :out_w]


def sp_conv_forward(self, x: torch.Tensor, keep_mask: torch.BoolTensor):
    ''''
    Simplest sparse convolution workaround (like SparK-style pretraining)
    '''
    x = super(type(self), self).forward(x)
    return x * keep_mask.to(dtype=x.dtype)


def _expand_keep_mask(x: torch.Tensor, keep_mask: torch.BoolTensor) -> torch.Tensor:
    if keep_mask.ndim != x.ndim:
        raise ValueError(
            f"keep_mask must have the same rank as x, got {keep_mask.ndim} and {x.ndim}."
        )
    if keep_mask.shape[0] != x.shape[0] or keep_mask.shape[-2:] != x.shape[-2:]:
        raise ValueError(
            f"keep_mask shape {tuple(keep_mask.shape)} is incompatible with x shape {tuple(x.shape)}."
        )
    if keep_mask.shape[1] not in (1, x.shape[1]):
        raise ValueError(
            f"keep_mask channel dimension must be 1 or match x, got {keep_mask.shape[1]} and {x.shape[1]}."
        )
    return keep_mask.to(dtype=x.dtype).expand_as(x)


def _masked_mean_and_var(
    x: torch.Tensor,
    keep_mask: torch.BoolTensor,
    reduce_dims: tuple[int, ...],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mask = _expand_keep_mask(x, keep_mask)
    masked_x = x * mask
    count = mask.sum(dim=reduce_dims, keepdim=True).clamp_min(1.0)
    mean = masked_x.sum(dim=reduce_dims, keepdim=True) / count
    var = ((x - mean).pow(2) * mask).sum(dim=reduce_dims, keepdim=True) / count
    return mask, mean, var


def _resolve_eps(eps: float | None, x: torch.Tensor) -> float:
    return torch.finfo(x.dtype).eps if eps is None else eps