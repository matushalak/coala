import torch

from cc.ml.pretraining.COALA import COALA
from cc.ml.pretraining.JEPAmodel import JEPA
from cc.ml.pretraining.LeJEPAmodel import LeJEPA
from cc.ml.pretraining.MAEmodel import MAE
from cc.ml.pretraining.common import (
    PREDICTOR_MODES,
    default_reconstruction_head_config,
    normalize_model_config,
    normalize_predictor_config,
    normalize_reconstruction_head_config,
)


def _vit_backbone_config() -> dict:
    return normalize_model_config(
        {
            "input_shape": (1, 32, 32),
            "d_layers": [32, 64],
            "layers_E": "ViT",
            "layers_E_kwargs": [
                {"depth": 1, "num_heads": 4, "transition_kernel_size": 4, "transition_stride": 4, "transition_padding": 0},
                {"depth": 1, "num_heads": 4, "transition_kernel_size": 2, "transition_stride": 2, "transition_padding": 0},
            ],
            "D_kwargs": {"densify_mode": "zero"},
        }
    )


def _predictive_backbone_config() -> dict:
    return normalize_model_config(
        {
            "input_shape": (1, 32, 32),
            "d_layers": [8, 16, 32],
            "layers_E": "ConvNet",
            "layers_E_kwargs": [
                {"depth": 1, "transition_kernel_size": 3, "transition_stride": 1, "transition_padding": 1, "block_kernel_size": 1},
                {"depth": 1, "transition_kernel_size": 3, "transition_stride": 2, "transition_padding": 1, "block_kernel_size": 1},
                {"depth": 1, "transition_kernel_size": 3, "transition_stride": 2, "transition_padding": 1, "block_kernel_size": 1},
            ],
            "E_kwargs": {"norm_type": "rmsnorm"},
            "D_kwargs": {"norm_type": "rmsnorm", "densify_mode": "zero", "use_skip": True},
        }
    )


def _rgb_backbone_config() -> dict:
    return normalize_model_config(
        {
            "input_shape": (3, 32, 32),
            "d_layers": [16, 32, 64],
            "layers_E": "ConvNet",
            "layers_E_kwargs": [
                {"depth": 1, "transition_kernel_size": 3, "transition_stride": 1, "transition_padding": 1, "block_kernel_size": 1},
                {"depth": 1, "transition_kernel_size": 3, "transition_stride": 2, "transition_padding": 1, "block_kernel_size": 1},
                {"depth": 1, "transition_kernel_size": 3, "transition_stride": 2, "transition_padding": 1, "block_kernel_size": 1},
            ],
            "E_kwargs": {"norm_type": "rmsnorm"},
            "D_kwargs": {"norm_type": "rmsnorm", "densify_mode": "zero", "use_skip": True},
        }
    )


def test_mae_supports_multiple_reconstruction_head_families():
    torch.manual_seed(0)
    imgs = torch.randn(2, 1, 32, 32)
    model_config = _vit_backbone_config()

    for head_family in ("ViT", "ConvNet", "ConvNeXt"):
        model = MAE(
            num_filters=8,
            lr=1e-3,
            mask_ratio=0.5,
            patch_size=4,
            masked_loss_weight=1.0,
            num_input_channels=1,
            image_size=32,
            model_config=model_config,
            reconstruction_head_family=head_family,
        )
        expected_head_config = normalize_reconstruction_head_config(
            default_reconstruction_head_config(
                family=head_family,
                input_shape=(8, 8),
                output_shape=(32, 32),
                feature_dim=32,
                num_output_channels=1,
            )
        )
        recon, keep_mask = model.reconstruct(imgs)

        assert recon.shape == imgs.shape
        assert keep_mask.shape == (2, 1, 32, 32)
        assert torch.isfinite(model(imgs))
        assert dict(model.hparams)["model_config"] == model_config
        assert dict(model.hparams)["reconstruction_head_config"] == expected_head_config


def test_jepa_smoke_and_logs_exact_configs():
    torch.manual_seed(0)
    imgs = torch.randn(2, 1, 32, 32)
    model_config = _predictive_backbone_config()
    predictor_config = normalize_predictor_config({"predictor_dim": 32, "depth": 1, "num_heads": 4})
    for predictor_mode in PREDICTOR_MODES:
        model = JEPA(
            num_filters=8,
            lr=1e-3,
            mask_ratio=0.5,
            patch_size=4,
            masked_loss_weight=1.0,
            num_input_channels=1,
            image_size=32,
            model_config=model_config,
            predictor_config=predictor_config,
            predictor_mode=predictor_mode,
        )

        distill_loss, metrics = model(imgs, return_metrics=True)
        _, _, _, predicted_latents, keep_masks = model._jepa_outputs(imgs)

        assert torch.isfinite(distill_loss)
        assert predicted_latents["feat0"].shape[-2:] == (32, 32)
        assert keep_masks["mask0"].shape[-2:] == (32, 32)
        assert "feat0_loss" in metrics
        assert dict(model.hparams)["model_config"] == model_config
        assert dict(model.hparams)["predictor_config"] == predictor_config
        assert dict(model.hparams)["predictor_mode"] == predictor_mode


def test_lejepa_smoke_and_logs_exact_configs():
    torch.manual_seed(0)
    imgs = torch.randn(2, 1, 32, 32)
    model_config = _predictive_backbone_config()
    predictor_config = normalize_predictor_config({"predictor_dim": 32, "depth": 1, "num_heads": 4})
    model = LeJEPA(
        num_filters=8,
        lr=1e-3,
        mask_ratio=0.5,
        patch_size=4,
        masked_loss_weight=1.0,
        num_input_channels=1,
        image_size=32,
        model_config=model_config,
        predictor_config=predictor_config,
        predictor_mode="decoder",
    )

    total_loss, distill_loss, sigreg_loss, metrics = model(imgs)

    assert torch.isfinite(total_loss)
    assert torch.isfinite(distill_loss)
    assert torch.isfinite(sigreg_loss)
    assert "feat0_loss" in metrics
    assert "feat0_sigreg" in metrics
    assert dict(model.hparams)["model_config"] == model_config
    assert dict(model.hparams)["predictor_config"] == predictor_config
    assert dict(model.hparams)["predictor_mode"] == "decoder"


def test_coala_smoke_and_logs_exact_configs():
    torch.manual_seed(0)
    imgs = torch.randn(2, 1, 32, 32)
    model_config = _predictive_backbone_config()
    predictor_config = normalize_predictor_config({"predictor_dim": 32, "depth": 1, "num_heads": 4})
    model = COALA(
        num_filters=8,
        lr=1e-3,
        mask_ratio=0.5,
        patch_size=4,
        masked_loss_weight=1.0,
        num_input_channels=1,
        image_size=32,
        model_config=model_config,
        predictor_config=predictor_config,
    )

    total_loss, distill_loss, sigreg_loss, metrics = model(imgs)
    _, _, _, _, coala_latents, keep_masks = model._jepa_outputs(imgs)

    assert torch.isfinite(total_loss)
    assert torch.isfinite(distill_loss)
    assert torch.isfinite(sigreg_loss)
    assert coala_latents["feat0"].shape[-2:] == (32, 32)
    assert keep_masks["mask0"].shape[-2:] == (32, 32)
    assert "feat0_loss" in metrics
    assert "feat0_sigreg" in metrics
    assert dict(model.hparams)["model_config"] == model_config
    assert dict(model.hparams)["predictor_config"] == predictor_config
    assert dict(model.hparams)["predictor_mode"] == "predictor+decoder"


def test_rgb32_pretraining_smoke():
    torch.manual_seed(0)
    imgs = torch.randn(2, 3, 32, 32)
    model_config = _rgb_backbone_config()
    predictor_config = normalize_predictor_config({"predictor_dim": 32, "depth": 1, "num_heads": 4})

    mae = MAE(
        num_filters=8,
        lr=1e-3,
        mask_ratio=0.5,
        patch_size=4,
        masked_loss_weight=1.0,
        num_input_channels=3,
        image_size=32,
        model_config=model_config,
    )
    recon, keep_mask = mae.reconstruct(imgs)
    assert recon.shape == imgs.shape
    assert keep_mask.shape == (2, 1, 32, 32)
    assert torch.isfinite(mae(imgs))

    coala = COALA(
        num_filters=8,
        lr=1e-3,
        mask_ratio=0.5,
        patch_size=4,
        masked_loss_weight=1.0,
        num_input_channels=3,
        image_size=32,
        model_config=model_config,
        predictor_config=predictor_config,
    )
    total_loss, distill_loss, sigreg_loss, _ = coala(imgs)
    assert torch.isfinite(total_loss)
    assert torch.isfinite(distill_loss)
    assert torch.isfinite(sigreg_loss)
