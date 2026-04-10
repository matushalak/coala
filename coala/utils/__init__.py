from __future__ import annotations

from typing import Iterable

import torch
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn

from .utils import load_checkpoint


class EMA(torch.nn.Module):
    """
    EMA (Exponential Moving Average) = discretized leaky integrator.
    """

    def __init__(self, shape: tuple, alpha: float = 0.1, baseline: torch.Tensor | None = None):
        super().__init__()
        self.alpha = alpha
        self.baseline = baseline if baseline is not None else torch.zeros(shape, requires_grad=False)
        self.ema = self.baseline.clone()

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.ema = (1 - self.alpha) * self.ema + self.alpha * (x + self.baseline)
        return self.ema

    def reset_state(self):
        self.ema = self.baseline.clone()


class ExponentialMovingAverage(torch.nn.Module):
    """
    Wrapper around torch.optim.swa_utils.AveragedModel for model-weight EMA.
    """

    def __init__(self, model: torch.nn.Module, decay: float = 0.99, use_buffers: bool = True):
        super().__init__()
        if not (0.0 < decay < 1.0):
            raise ValueError(f"decay must be in (0, 1), got {decay}")

        self.decay = decay
        self.ema_model = AveragedModel(
            model,
            multi_avg_fn=get_ema_multi_avg_fn(decay),
            use_buffers=use_buffers,
        )
        self._freeze()

    def _freeze(self):
        self.training = False
        self.ema_model.eval()
        for param in self.ema_model.parameters():
            param.requires_grad_(False)

    def train(self, mode: bool = True):
        super().train(mode)
        self.training = False
        self.ema_model.eval()
        return self

    @torch.no_grad()
    def update(self, model: torch.nn.Module):
        self.ema_model.update_parameters(model)

    @torch.no_grad()
    def forward(self, *args, **kwargs):
        return self.ema_model(*args, **kwargs)

    def state_dict(self, *args, **kwargs):
        return self.ema_model.state_dict(*args, **kwargs)

    def load_state_dict(self, state_dict, strict: bool = True):
        return self.ema_model.load_state_dict(state_dict, strict=strict)


class ThresholdReLU(torch.nn.Module):
    """
    Thresholded ReLU activation: max(0, x - threshold).
    """

    def __init__(self, threshold: float = 0.0):
        super().__init__()
        self.threshold = threshold

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.clamp(x - self.threshold, min=0.0)


def nonnegative(x: torch.Tensor) -> torch.Tensor:
    return torch.clamp(x, min=0.0)


def randn_reparam(size: tuple[int, ...], mu: float | Iterable, sigma: float | Iterable) -> torch.Tensor:
    mu = torch.as_tensor(mu)
    sigma = torch.as_tensor(sigma)

    if len(size) == 0:
        size = (1,)

    mu_shape = mu.shape if len(mu.shape) > 0 else (1,)
    sigma_shape = sigma.shape if len(sigma.shape) > 0 else (1,)

    assert isinstance(size, tuple), "size must be a tuple of ints. For generating a single sample, use size=()."
    if len(size) == 1:
        z = torch.randn(*size, *mu_shape)
    else:
        assert (
            (mu_shape == (1,) or mu_shape == (size[1],))
            and (sigma_shape == (1,) or sigma_shape == (size[1],))
        ), "mu and sigma must be scalar if generating a random matrix."
        z = torch.randn(*size)
    if len(sigma_shape) == 1 and sigma_shape[0] == 1:
        sigma = sigma * torch.eye(z.shape[-1])
    elif len(sigma_shape) == 1:
        sigma = torch.diag(sigma)
    else:
        assert sigma_shape == (mu_shape[0], mu_shape[0]), (
            "If sigma is a matrix, it must have shape (n_features, n_features)."
        )

    return (mu + z @ sigma).squeeze()


__all__ = [
    "EMA",
    "ExponentialMovingAverage",
    "ThresholdReLU",
    "load_checkpoint",
    "nonnegative",
    "randn_reparam",
]
