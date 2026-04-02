import re

import torch


def pair(value: int | tuple[int, int]) -> tuple[int, int]:
    if isinstance(value, tuple):
        assert len(value) == 2
        return value
    return value, value


def conv_output_shape(
    input_shape: tuple[int, int],
    kernel_size: int | tuple[int, int],
    stride: int | tuple[int, int],
    padding: int | tuple[int, int],
) -> tuple[int, int]:
    kernel_size = pair(kernel_size)
    stride = pair(stride)
    padding = pair(padding)
    out_h = ((input_shape[0] + 2 * padding[0] - kernel_size[0]) // stride[0]) + 1
    out_w = ((input_shape[1] + 2 * padding[1] - kernel_size[1]) // stride[1]) + 1
    return out_h, out_w


def convtranspose_output_padding(
    input_shape: tuple[int, int],
    output_shape: tuple[int, int],
    kernel_size: int | tuple[int, int],
    stride: int | tuple[int, int],
    padding: int | tuple[int, int],
) -> tuple[int, int]:
    kernel_size = pair(kernel_size)
    stride = pair(stride)
    padding = pair(padding)
    base_h = (input_shape[0] - 1) * stride[0] - (2 * padding[0]) + kernel_size[0]
    base_w = (input_shape[1] - 1) * stride[1] - (2 * padding[1]) + kernel_size[1]
    output_padding = output_shape[0] - base_h, output_shape[1] - base_w
    assert 0 <= output_padding[0] < stride[0]
    assert 0 <= output_padding[1] < stride[1]
    return output_padding


def downsample_center_mask(
    keep_mask: torch.BoolTensor,
    out_size: tuple[int, int],
    stride: int = 2,
) -> torch.BoolTensor:
    out_h, out_w = out_size
    return keep_mask[:, :, ::stride, ::stride][:, :, :out_h, :out_w]


def sp_conv_forward(self, x: torch.Tensor, keep_mask: torch.BoolTensor):
    x = super(type(self), self).forward(x)
    return x * keep_mask.to(dtype=x.dtype)


def _expand_keep_mask(x: torch.Tensor, keep_mask: torch.BoolTensor) -> torch.Tensor:
    assert keep_mask.ndim == x.ndim
    assert keep_mask.shape[0] == x.shape[0]
    assert keep_mask.shape[-2:] == x.shape[-2:]
    assert keep_mask.shape[1] in (1, x.shape[1])
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


def sorted_stage_keys(latents: dict[str, torch.Tensor], prefix: str) -> list[str]:
    keys = [key for key in latents if key.startswith(prefix)]

    def _suffix(name: str) -> int:
        match = re.search(r"(\d+)$", name)
        assert match is not None
        return int(match.group(1))

    return sorted(keys, key=_suffix)
