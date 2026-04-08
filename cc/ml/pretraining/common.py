from __future__ import annotations

import copy
import math

import torch
import torch.nn as nn

from cc.ml.architecture import (
    HierarchicalAutoencoder,
    PredictiveHierarchicalAutoencoder,
)
from cc.ml.architecture.HAE import build_predictive_hae_modules
from cc.ml.architecture.modules import resolve_module_family
from cc.ml.architecture.modules.utils import sorted_stage_keys

PREDICTOR_MODES = ("decoder", "predictor", "predictor+decoder")


def _default_num_heads(dim: int) -> int:
    for candidate in (8, 4, 2, 1):
        if dim % candidate == 0:
            return candidate
    return 1


def default_model_config(
    *,
    image_size: int,
    num_input_channels: int,
    num_filters: int,
    norm_type: str,
    decoder_densify_mode: str,
    use_skip: bool,
    upconv_method: str,
) -> dict:
    return {
        "input_shape": (num_input_channels, image_size, image_size),
        "d_layers": [num_filters // 2, num_filters, 2 * num_filters, 4 * num_filters],
        "layers_E": "ConvNet",
        "layers_D": "ConvNet",
        "layers_E_kwargs": [
            {"depth": 1, "transition_kernel_size": 3, "transition_stride": 1, "transition_padding": 1, "block_kernel_size": 1},
            {"depth": 1, "transition_kernel_size": 3, "transition_stride": 2, "transition_padding": 1, "block_kernel_size": 1},
            {"depth": 1, "transition_kernel_size": 3, "transition_stride": 2, "transition_padding": 1, "block_kernel_size": 1},
            {"depth": 1, "transition_kernel_size": 3, "transition_stride": 2, "transition_padding": 1, "block_kernel_size": 1},
        ],
        "layers_D_kwargs": [
            {"depth": 1, "transition_kernel_size": 3, "transition_stride": 1, "transition_padding": 1, "block_kernel_size": 1},
            {"depth": 1, "transition_kernel_size": 3, "transition_stride": 2, "transition_padding": 1, "block_kernel_size": 1},
            {"depth": 1, "transition_kernel_size": 3, "transition_stride": 2, "transition_padding": 1, "block_kernel_size": 1},
            {"depth": 1, "transition_kernel_size": 3, "transition_stride": 2, "transition_padding": 1, "block_kernel_size": 1},
        ],
        "E_kwargs": {"norm_type": norm_type},
        "D_kwargs": {"norm_type": norm_type, "densify_mode": decoder_densify_mode, "use_skip": use_skip, "upsampling": upconv_method},
    }


def normalize_model_config(config: dict) -> dict:
    cfg = copy.deepcopy(config)
    assert "input_shape" in cfg
    assert "d_layers" in cfg
    cfg["input_shape"] = tuple(cfg["input_shape"])
    cfg["d_layers"] = list(cfg["d_layers"])
    n_layers = len(cfg["d_layers"])
    cfg["n_layers"] = int(cfg.get("n_layers", n_layers))
    assert cfg["n_layers"] == n_layers
    cfg["layers_E"] = cfg.get("layers_E", "ConvNet")
    cfg["layers_D"] = cfg.get("layers_D", cfg["layers_E"])
    cfg["layers_E_kwargs"] = copy.deepcopy(cfg.get("layers_E_kwargs", [{} for _ in range(n_layers)]))
    cfg["layers_D_kwargs"] = copy.deepcopy(cfg.get("layers_D_kwargs", cfg["layers_E_kwargs"]))
    assert len(cfg["layers_E_kwargs"]) == n_layers
    assert len(cfg["layers_D_kwargs"]) == n_layers
    cfg["E_kwargs"] = copy.deepcopy(cfg.get("E_kwargs", {}))
    cfg["D_kwargs"] = copy.deepcopy(cfg.get("D_kwargs", {}))
    return cfg


def normalize_predictor_config(config: dict | None) -> dict:
    return {} if config is None else copy.deepcopy(config)


def default_reconstruction_head_config(
    *,
    family: str,
    input_shape: tuple[int, int],
    output_shape: tuple[int, int],
    feature_dim: int,
    num_output_channels: int,
) -> dict:
    assert output_shape[0] % input_shape[0] == 0
    assert output_shape[1] % input_shape[1] == 0
    scale_h = output_shape[0] // input_shape[0]
    scale_w = output_shape[1] // input_shape[1]
    assert scale_h == scale_w
    scale = scale_h
    kwargs = {"depth": 1}
    if scale > 1:
        kwargs.update(
            {
                "transition_kernel_size": scale,
                "transition_stride": scale,
                "transition_padding": 0,
            }
        )
    if family.lower() == "vit":
        kwargs["num_heads"] = _default_num_heads(feature_dim)
    return {
        "family": family,
        "num_output_channels": num_output_channels,
        "kwargs": kwargs,
    }


def normalize_reconstruction_head_config(config: dict) -> dict:
    cfg = copy.deepcopy(config)
    assert "family" in cfg
    assert "num_output_channels" in cfg
    cfg["kwargs"] = copy.deepcopy(cfg.get("kwargs", {}))
    return cfg


class GenerativeHead(nn.Module):
    def __init__(
        self,
        *,
        family: str,
        in_channels: int,
        input_spatial_shape: tuple[int, int],
        output_spatial_shape: tuple[int, int],
        num_output_channels: int,
        kwargs: dict | None = None,
    ):
        super().__init__()
        family_impl = resolve_module_family(family)
        kwargs = {} if kwargs is None else dict(kwargs)
        self.stage = family_impl.make_decoder_stage(
            in_channels=in_channels,
            out_channels=in_channels,
            input_spatial_shape=input_spatial_shape,
            output_spatial_shape=output_spatial_shape,
            **kwargs,
        )
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, num_output_channels, kernel_size=1, stride=1, padding=0),
            nn.Hardtanh(-1.0, 1.0),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.stage(x))


def instantiate_autoencoder(model_config: dict, *, predictive: bool, predictor_config: dict | None = None):
    if predictive:
        return PredictiveHierarchicalAutoencoder(**model_config, P_kwargs=normalize_predictor_config(predictor_config))
    return HierarchicalAutoencoder(**model_config)


def instantiate_jepa_modules(
    model_config: dict,
    *,
    predictor_mode: str,
    predictor_config: dict | None = None,
):
    assert predictor_mode in PREDICTOR_MODES
    return build_predictive_hae_modules(
        d_layers=list(model_config["d_layers"]),
        predictor_mode=predictor_mode,
        E_kwargs=model_config.get("E_kwargs"),
        layers_E=model_config.get("layers_E", "ConvNet"),
        layers_E_kwargs=model_config.get("layers_E_kwargs"),
        D_kwargs=model_config.get("D_kwargs"),
        layers_D=model_config.get("layers_D"),
        layers_D_kwargs=model_config.get("layers_D_kwargs"),
        P_kwargs=normalize_predictor_config(predictor_config),
        input_shape=tuple(model_config["input_shape"]),
    )


def run_jepa_prediction(
    *,
    decoder,
    predictor,
    dirty_encoder_latents: dict[str, torch.Tensor],
    predictor_mode: str,
):
    assert predictor_mode in PREDICTOR_MODES
    predictor_latents = None
    decoder_latents = None
    if predictor_mode == "predictor":
        assert predictor is not None
        predictor_latents = predictor(dirty_encoder_latents)
        predicted_latents = predictor_latents
    elif predictor_mode == "decoder":
        decoder_latents = decoder(dirty_encoder_latents)
        predicted_latents = decoder_latents
    else:
        assert predictor is not None
        predictor_latents = predictor(dirty_encoder_latents)
        decoder_latents = decoder(dirty_encoder_latents, skip_latents=predictor_latents)
        predicted_latents = decoder_latents
    return predicted_latents, decoder_latents, predictor_latents


def combine_feature_latents(
    left: dict[str, torch.Tensor],
    right: dict[str, torch.Tensor],
    feature_names: list[str],
) -> dict[str, torch.Tensor]:
    return {name: left[name] + right[name] for name in feature_names}


def configure_adamw_with_warmup_and_cosine_decay(module) -> torch.optim.Optimizer | dict:
    optimizer = torch.optim.AdamW(
        module.parameters(),
        lr=module.hparams.lr,
        betas=(0.9, 0.95),
        weight_decay=module.hparams.weight_decay,
    )

    trainer = getattr(module, "_trainer", None)
    if trainer is None:
        return optimizer

    max_epochs = int(trainer.max_epochs)
    assert max_epochs > 0
    warmup_epochs = int(module.hparams.warmup_epochs)
    assert 0 <= warmup_epochs <= max_epochs

    def lr_lambda(epoch: int) -> float:
        warmup = (epoch + 1) / (warmup_epochs + 1e-8)
        cosine = 0.5 * (math.cos(epoch / max_epochs * math.pi) + 1.0)
        return min(warmup, cosine)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    return {
        "optimizer": optimizer,
        "lr_scheduler": {
            "scheduler": scheduler,
            "interval": "epoch",
            "frequency": 1,
        },
    }


def feature_names_from_latents(latents: dict[str, torch.Tensor]) -> list[str]:
    return sorted_stage_keys(latents, "feat")


def mask_names_from_latents(latents: dict[str, torch.Tensor]) -> list[str]:
    return sorted_stage_keys(latents, "mask")
