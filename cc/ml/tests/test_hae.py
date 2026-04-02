import torch

from cc.ml.architecture.HAE import (
    HierarchicalAutoencoder,
    PredictiveHierarchicalAutoencoder,
)


def _keep_mask(batch_size: int, height: int, width: int) -> torch.BoolTensor:
    keep_mask = torch.ones((batch_size, 1, height, width), dtype=torch.bool)
    keep_mask[:, :, height // 4 : height // 2, width // 4 : width // 2] = False
    return keep_mask


def test_convnet_hae_shape_symmetry():
    model = HierarchicalAutoencoder(
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
        input_shape=(1, 32, 32),
    )

    x = torch.randn(2, 1, 32, 32)
    decoded, encoded = model(x, keep_mask=_keep_mask(2, 32, 32))

    assert encoded["feat0"].shape == (2, 16, 32, 32)
    assert encoded["feat1"].shape == (2, 32, 16, 16)
    assert encoded["feat2"].shape == (2, 64, 8, 8)
    assert decoded["feat0"].shape == encoded["feat0"].shape
    assert decoded["feat1"].shape == encoded["feat1"].shape
    assert decoded["feat2"].shape == encoded["feat2"].shape


def test_convnext_hae_shape_symmetry():
    model = HierarchicalAutoencoder(
        n_layers=3,
        d_layers=[24, 48, 96],
        E_kwargs={"norm_type": "rmsnorm"},
        layers_E="ConvNeXt",
        layers_E_kwargs=[
            {"depth": 1, "transition_kernel_size": 1, "transition_stride": 1, "transition_padding": 0},
            {"depth": 1, "transition_kernel_size": 2, "transition_stride": 2, "transition_padding": 0},
            {"depth": 1, "transition_kernel_size": 2, "transition_stride": 2, "transition_padding": 0},
        ],
        D_kwargs={"densify_mode": "zero"},
        input_shape=(3, 64, 64),
    )

    x = torch.randn(2, 3, 64, 64)
    decoded, encoded = model(x, keep_mask=_keep_mask(2, 64, 64))

    assert encoded["feat0"].shape == (2, 24, 64, 64)
    assert encoded["feat1"].shape == (2, 48, 32, 32)
    assert encoded["feat2"].shape == (2, 96, 16, 16)
    assert decoded["feat0"].shape == encoded["feat0"].shape
    assert decoded["feat1"].shape == encoded["feat1"].shape
    assert decoded["feat2"].shape == encoded["feat2"].shape


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
        input_shape=(1, 32, 32),
    )

    x = torch.randn(2, 1, 32, 32)
    decoded, encoded, predicted = model(x, keep_mask=_keep_mask(2, 32, 32))

    assert predicted["feat0"].shape == encoded["feat0"].shape
    assert predicted["feat1"].shape == encoded["feat1"].shape
    assert predicted["feat2"].shape == encoded["feat2"].shape
    assert decoded["feat0"].shape == encoded["feat0"].shape
