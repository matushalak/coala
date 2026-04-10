from __future__ import annotations

from pathlib import Path
import sys

import torch

# Allow direct script execution from inside this folder.
if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[3]))

from coala.architecture.sparse_cnn_unet import (
    SparseGlobalResponseNorm,
    SparseLayerNorm2d,
    SparseRMSNorm2d,
)


def _resolve_eps(eps: float | None, x: torch.Tensor) -> float:
    return torch.finfo(x.dtype).eps if eps is None else eps


def _masked_layer_norm_expected(
    x: torch.Tensor,
    keep_mask: torch.BoolTensor,
    eps: float | None,
) -> torch.Tensor:
    eps = _resolve_eps(eps, x)
    mask = keep_mask.to(dtype=x.dtype).expand_as(x)
    count = mask.sum()
    mean = (x * mask).sum() / count
    var = (((x - mean).pow(2)) * mask).sum() / count
    return ((x - mean) / torch.sqrt(var + eps)) * mask


def _masked_rms_norm_expected(
    x: torch.Tensor,
    keep_mask: torch.BoolTensor,
    eps: float | None,
) -> torch.Tensor:
    eps = _resolve_eps(eps, x)
    mask = keep_mask.to(dtype=x.dtype).expand_as(x)
    count = mask.sum()
    rms = torch.rsqrt((x.pow(2) * mask).sum() / count + eps)
    return x * rms * mask


def test_sparse_layer_norm_ignores_masked_pixels():
    layer = SparseLayerNorm2d((1, 2, 2), elementwise_affine=False)
    x = torch.tensor([[[[1.0, 3.0], [100.0, 200.0]]]])
    keep_mask = torch.tensor([[[[True, True], [False, False]]]])

    y = layer(x, keep_mask)

    expected = _masked_layer_norm_expected(x, keep_mask, layer.eps)
    torch.testing.assert_close(y, expected, atol=1e-6, rtol=0.0)


def test_sparse_rms_norm_ignores_masked_pixels():
    layer = SparseRMSNorm2d((1, 2, 2), elementwise_affine=False)
    x = torch.tensor([[[[3.0, 4.0], [100.0, 200.0]]]])
    keep_mask = torch.tensor([[[[True, True], [False, False]]]])

    y = layer(x, keep_mask)

    expected = _masked_rms_norm_expected(x, keep_mask, layer.eps)
    torch.testing.assert_close(y, expected, atol=1e-6, rtol=0.0)


def test_sparse_grn_ignores_masked_pixels():
    layer = SparseGlobalResponseNorm(2)
    x = torch.tensor(
        [[
            [[3.0, 100.0], [100.0, 100.0]],
            [[6.0, 200.0], [200.0, 200.0]],
        ]]
    )
    keep_mask = torch.tensor([[[[True, False], [False, False]]]])

    y = layer(x, keep_mask)

    mask = keep_mask.to(dtype=x.dtype)
    gx = torch.sqrt((x.pow(2) * mask).sum(dim=[2, 3], keepdim=True))
    nx = gx / (gx.mean(dim=1, keepdim=True) + layer.eps)
    expected = (x * nx + x) * mask
    torch.testing.assert_close(y, expected, atol=1e-6, rtol=0.0)

if __name__ == "__main__":
    test_sparse_layer_norm_ignores_masked_pixels()
    test_sparse_rms_norm_ignores_masked_pixels()
    test_sparse_grn_ignores_masked_pixels()
    print("All tests passed!")
