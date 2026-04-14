import torch

from coala.architecture.architecture import COALANet, infer_dataset_name_from_checkpoint_path, stack_temporal_reconstructions
from coala.pretraining.common import (
    GenerativeHead,
    default_model_config,
    default_reconstruction_head_config,
    instantiate_autoencoder,
    normalize_model_config,
)


def test_coalanet_supports_modular_rgb_autoencoder():
    model_config = normalize_model_config(
        default_model_config(
            image_size=32,
            num_input_channels=3,
            num_filters=64,
            norm_type="rmsnorm",
            decoder_densify_mode="random",
            use_skip=True,
            upconv_method="upsample+conv",
        )
    )
    autoencoder = instantiate_autoencoder(model_config, predictive=False)
    reconstruction_head_config = default_reconstruction_head_config(
        family="ConvNet",
        input_shape=autoencoder.encoder.spatial_shapes[0],
        output_shape=model_config["input_shape"][1:],
        feature_dim=autoencoder.encoder.feature_dims[0],
        num_output_channels=model_config["input_shape"][0],
    )
    generative_head = GenerativeHead(
        family=reconstruction_head_config["family"],
        in_channels=autoencoder.encoder.feature_dims[0],
        input_spatial_shape=autoencoder.encoder.spatial_shapes[0],
        output_spatial_shape=model_config["input_shape"][1:],
        num_output_channels=reconstruction_head_config["num_output_channels"],
        kwargs=reconstruction_head_config["kwargs"],
    )
    model = COALANet(
        autoencoder.encoder,
        autoencoder.decoder,
        generative_head=generative_head,
        data_dims=tuple(model_config["input_shape"]),
        mode="generative",
    )
    sequence = torch.randn(2, 3, *model_config["input_shape"])
    reconstructions = stack_temporal_reconstructions(model(sequence))
    assert reconstructions.shape == (2, 3, *model_config["input_shape"])


def test_infer_dataset_name_from_checkpoint_path_supports_log_aliases():
    path = "coala/logs/MAE_logs/cifar/lightning_logs/version_39/checkpoints/epoch=134-step=47520.ckpt"
    assert infer_dataset_name_from_checkpoint_path(path) == "cifar10"
