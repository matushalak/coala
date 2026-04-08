import pytest
import torch

from cc.ml.architecture.HAE import HierarchicalAutoencoder, PredictiveHierarchicalAutoencoder


EXPECTED_SPATIAL_SHAPES = {
    "ConvNet": {
        28: [(28, 28), (14, 14), (7, 7), (4, 4)],
        32: [(32, 32), (16, 16), (8, 8), (4, 4)],
        64: [(64, 64), (32, 32), (16, 16), (8, 8)],
        96: [(96, 96), (48, 48), (24, 24), (12, 12)],
    },
    "ConvNeXt": {
        28: [(28, 28), (14, 14), (7, 7), (3, 3)],
        32: [(32, 32), (16, 16), (8, 8), (4, 4)],
        64: [(64, 64), (32, 32), (16, 16), (8, 8)],
        96: [(96, 96), (48, 48), (24, 24), (12, 12)],
    },
    "ViT": {
        28: [(7, 7), (3, 3), (1, 1)],
        32: [(8, 8), (4, 4), (2, 2)],
        64: [(16, 16), (8, 8), (4, 4)],
        96: [(24, 24), (12, 12), (6, 6)],
    },
}


def _keep_mask(batch_size: int, height: int, width: int) -> torch.BoolTensor:
    keep_mask = torch.ones((batch_size, 1, height, width), dtype=torch.bool)
    keep_mask[:, :, height // 4 : height // 2, width // 4 : width // 2] = False
    return keep_mask


def _convnet_config(resolution: int) -> dict:
    return {
        "n_layers": 4,
        "d_layers": [16, 32, 64, 128],
        "E_kwargs": {"norm_type": "rmsnorm"},
        "layers_E": "ConvNet",
        "layers_E_kwargs": [
            {"depth": 1, "transition_kernel_size": 3, "transition_stride": 1, "transition_padding": 1, "block_kernel_size": 1},
            {"depth": 1, "transition_kernel_size": 3, "transition_stride": 2, "transition_padding": 1, "block_kernel_size": 1},
            {"depth": 1, "transition_kernel_size": 3, "transition_stride": 2, "transition_padding": 1, "block_kernel_size": 1},
            {"depth": 1, "transition_kernel_size": 3, "transition_stride": 2, "transition_padding": 1, "block_kernel_size": 1},
        ],
        "D_kwargs": {"densify_mode": "zero"},
        "input_shape": (1, resolution, resolution),
    }


def _convnext_config(resolution: int) -> dict:
    return {
        "n_layers": 4,
        "d_layers": [24, 48, 96, 192],
        "E_kwargs": {"norm_type": "rmsnorm"},
        "layers_E": "ConvNeXt",
        "layers_E_kwargs": [
            {"depth": 1, "transition_kernel_size": 1, "transition_stride": 1, "transition_padding": 0},
            {"depth": 1, "transition_kernel_size": 2, "transition_stride": 2, "transition_padding": 0},
            {"depth": 1, "transition_kernel_size": 2, "transition_stride": 2, "transition_padding": 0},
            {"depth": 1, "transition_kernel_size": 2, "transition_stride": 2, "transition_padding": 0},
        ],
        "D_kwargs": {"densify_mode": "zero"},
        "input_shape": (3, resolution, resolution),
    }


def _vit_config(resolution: int) -> dict:
    return {
        "n_layers": 3,
        "d_layers": [48, 96, 192],
        "layers_E": "ViT",
        "layers_E_kwargs": [
            {"depth": 1, "num_heads": 4, "transition_kernel_size": 4, "transition_stride": 4, "transition_padding": 0},
            {"depth": 1, "num_heads": 4, "transition_kernel_size": 2, "transition_stride": 2, "transition_padding": 0},
            {"depth": 1, "num_heads": 4, "transition_kernel_size": 2, "transition_stride": 2, "transition_padding": 0},
        ],
        "D_kwargs": {"densify_mode": "zero"},
        "input_shape": (3, resolution, resolution),
    }


def _config_for_family(family: str, resolution: int) -> dict:
    if family == "ConvNet":
        return _convnet_config(resolution)
    if family == "ConvNeXt":
        return _convnext_config(resolution)
    if family == "ViT":
        return _vit_config(resolution)
    assert False


@pytest.mark.parametrize("family", ["ConvNet", "ConvNeXt", "ViT"])
@pytest.mark.parametrize("resolution", [28, 32, 64, 96])
def test_hae_resolution_sweep_spatial_shapes(family: str, resolution: int):
    config = _config_for_family(family, resolution)
    model = HierarchicalAutoencoder(**config)
    expected_shapes = EXPECTED_SPATIAL_SHAPES[family][resolution]

    assert model.encoder.spatial_shapes == expected_shapes

    channels = config["input_shape"][0]
    x = torch.randn(2, channels, resolution, resolution)
    decoded, encoded = model(x, keep_mask=_keep_mask(2, resolution, resolution))

    for index, spatial_shape in enumerate(expected_shapes):
        assert encoded[f"feat{index}"].shape[-2:] == spatial_shape
        assert encoded[f"mask{index}"].shape[-2:] == spatial_shape
        assert decoded[f"feat{index}"].shape == encoded[f"feat{index}"].shape


def test_convnet_hae_shape_symmetry():
    model = HierarchicalAutoencoder(**_convnet_config(32))

    x = torch.randn(2, 1, 32, 32)
    decoded, encoded = model(x, keep_mask=_keep_mask(2, 32, 32))

    assert encoded["feat0"].shape == (2, 16, 32, 32)
    assert encoded["feat1"].shape == (2, 32, 16, 16)
    assert encoded["feat2"].shape == (2, 64, 8, 8)
    assert encoded["feat3"].shape == (2, 128, 4, 4)
    assert decoded["feat0"].shape == encoded["feat0"].shape
    assert decoded["feat1"].shape == encoded["feat1"].shape
    assert decoded["feat2"].shape == encoded["feat2"].shape
    assert decoded["feat3"].shape == encoded["feat3"].shape


def test_convnext_hae_shape_symmetry():
    model = HierarchicalAutoencoder(**_convnext_config(64))

    x = torch.randn(2, 3, 64, 64)
    decoded, encoded = model(x, keep_mask=_keep_mask(2, 64, 64))

    assert encoded["feat0"].shape == (2, 24, 64, 64)
    assert encoded["feat1"].shape == (2, 48, 32, 32)
    assert encoded["feat2"].shape == (2, 96, 16, 16)
    assert encoded["feat3"].shape == (2, 192, 8, 8)
    assert decoded["feat0"].shape == encoded["feat0"].shape
    assert decoded["feat1"].shape == encoded["feat1"].shape
    assert decoded["feat2"].shape == encoded["feat2"].shape
    assert decoded["feat3"].shape == encoded["feat3"].shape


def test_vit_hae_stops_at_first_patchified_layer():
    model = HierarchicalAutoencoder(
        n_layers=2,
        d_layers=[48, 96],
        layers_E="ViT",
        layers_E_kwargs=[
            {"depth": 1, "num_heads": 4, "transition_kernel_size": 4, "transition_stride": 4, "transition_padding": 0},
            {"depth": 1, "num_heads": 4, "transition_kernel_size": 2, "transition_stride": 2, "transition_padding": 0},
        ],
        D_kwargs={"densify_mode": "zero"},
        input_shape=(3, 32, 32),
    )

    x = torch.randn(2, 3, 32, 32)
    decoded, encoded = model(x, keep_mask=_keep_mask(2, 32, 32))

    assert encoded["feat0"].shape == (2, 48, 8, 8)
    assert encoded["feat1"].shape == (2, 96, 4, 4)
    assert decoded["feat0"].shape == encoded["feat0"].shape
    assert decoded["feat1"].shape == encoded["feat1"].shape


def test_predictive_hae_predictor_matches_encoder_shapes():
    model = PredictiveHierarchicalAutoencoder(
        n_layers=3,
        d_layers=[16, 32, 64],
        E_kwargs={"norm_type": "rmsnorm"},
        layers_E="ConvNet",
        layers_E_kwargs=[
            {"depth": 1, "transition_kernel_size": 3, "transition_stride": 1, "transition_padding": 1},
            {"depth": 1, "transition_kernel_size": 3, "transition_stride": 2, "transition_padding": 1},
            {"depth": 1, "transition_kernel_size": 3, "transition_stride": 2, "transition_padding": 1},
        ],
        D_kwargs={"densify_mode": "zero"},
        P_kwargs={"predictor_dim": 48, "depth": 1, "num_heads": 4},
        predictor_mode="predictor",
        input_shape=(1, 32, 32),
    )

    x = torch.randn(2, 1, 32, 32)
    outputs = model(x, keep_mask=_keep_mask(2, 32, 32))
    encoded = outputs["encoder_latents"]
    predicted = outputs["predicted_latents"]

    assert predicted["feat0"].shape == encoded["feat0"].shape
    assert predicted["feat1"].shape == encoded["feat1"].shape
    assert predicted["feat2"].shape == encoded["feat2"].shape
    assert outputs["decoder_latents"] is None
    assert outputs["predictor_latents"]["feat0"].shape == encoded["feat0"].shape


def test_predictive_hae_decoder_mode_uses_no_predictor():
    model = PredictiveHierarchicalAutoencoder(
        n_layers=3,
        d_layers=[16, 32, 64],
        E_kwargs={"norm_type": "rmsnorm"},
        layers_E="ConvNet",
        layers_E_kwargs=[
            {"depth": 1, "transition_kernel_size": 3, "transition_stride": 1, "transition_padding": 1},
            {"depth": 1, "transition_kernel_size": 3, "transition_stride": 2, "transition_padding": 1},
            {"depth": 1, "transition_kernel_size": 3, "transition_stride": 2, "transition_padding": 1},
        ],
        D_kwargs={"densify_mode": "zero", "use_skip": True},
        predictor_mode="decoder",
        input_shape=(1, 32, 32),
    )

    outputs = model(torch.randn(2, 1, 32, 32), keep_mask=_keep_mask(2, 32, 32))

    assert model.predictor is None
    assert outputs["predictor_latents"] is None
    assert outputs["decoder_latents"]["feat0"].shape == outputs["encoder_latents"]["feat0"].shape
    assert outputs["predicted_latents"]["feat0"].shape == outputs["encoder_latents"]["feat0"].shape


def test_predictive_hae_predictor_plus_decoder_mode_feeds_decoder():
    model = PredictiveHierarchicalAutoencoder(
        n_layers=3,
        d_layers=[16, 32, 64],
        E_kwargs={"norm_type": "rmsnorm"},
        layers_E="ConvNet",
        layers_E_kwargs=[
            {"depth": 1, "transition_kernel_size": 3, "transition_stride": 1, "transition_padding": 1},
            {"depth": 1, "transition_kernel_size": 3, "transition_stride": 2, "transition_padding": 1},
            {"depth": 1, "transition_kernel_size": 3, "transition_stride": 2, "transition_padding": 1},
        ],
        D_kwargs={"densify_mode": "zero", "use_skip": True},
        P_kwargs={"predictor_dim": 48, "depth": 1, "num_heads": 4},
        predictor_mode="predictor+decoder",
        input_shape=(1, 32, 32),
    )

    outputs = model(torch.randn(2, 1, 32, 32), keep_mask=_keep_mask(2, 32, 32))

    assert outputs["predictor_latents"] is not None
    assert outputs["decoder_latents"] is not None
    assert outputs["predicted_latents"]["feat0"].shape == outputs["encoder_latents"]["feat0"].shape
    assert outputs["decoder_latents"]["feat0"].shape == outputs["encoder_latents"]["feat0"].shape
