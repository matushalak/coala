from torch import nn

from cc.ml.architecture.decoders.decoder import HierarchicalDecoder
from cc.ml.architecture.encoders.encoder import HierarchicalEncoder
from cc.ml.architecture.predictor.vit_predictor import ViTPredictor


def build_hae_from_config(config: dict):
    return HierarchicalAutoencoder(
        n_layers=config.get("n_layers"),
        d_layers=config["d_layers"],
        E_kwargs=config.get("E_kwargs"),
        layers_E=config.get("layers_E", "ConvNet"),
        layers_E_kwargs=config.get("layers_E_kwargs"),
        D_kwargs=config.get("D_kwargs"),
        layers_D=config.get("layers_D"),
        layers_D_kwargs=config.get("layers_D_kwargs"),
        input_shape=tuple(config["input_shape"]),
    )


def build_predictive_hae_from_config(config: dict):
    return PredictiveHierarchicalAutoencoder(
        n_layers=config.get("n_layers"),
        d_layers=config["d_layers"],
        E_kwargs=config.get("E_kwargs"),
        layers_E=config.get("layers_E", "ConvNet"),
        layers_E_kwargs=config.get("layers_E_kwargs"),
        D_kwargs=config.get("D_kwargs"),
        layers_D=config.get("layers_D"),
        layers_D_kwargs=config.get("layers_D_kwargs"),
        P_kwargs=config.get("P_kwargs"),
        input_shape=tuple(config["input_shape"]),
    )


class HierarchicalAutoencoder(nn.Module):
    def __init__(
        self,
        n_layers: int | None,
        d_layers: list[int],
        E_kwargs: dict | None = None,
        layers_E="ConvNet",
        layers_E_kwargs: list[dict] | None = None,
        D_kwargs: dict | None = None,
        layers_D=None,
        layers_D_kwargs: list[dict] | None = None,
        input_shape: tuple[int, int, int] | None = None,
    ):
        super().__init__()
        assert input_shape is not None
        self.n_layers = len(d_layers) if n_layers is None else n_layers
        assert self.n_layers == len(d_layers)
        self.input_shape = tuple(input_shape)
        self.feature_dims = list(d_layers)

        self.encoder = HierarchicalEncoder(
            input_shape=self.input_shape,
            d_layers=self.feature_dims,
            layers=layers_E,
            layer_kwargs=layers_E_kwargs,
            shared_kwargs=E_kwargs,
        )
        decoder_layers = layers_E if layers_D is None else layers_D
        decoder_layer_kwargs = layers_E_kwargs if layers_D_kwargs is None else layers_D_kwargs
        self.decoder = HierarchicalDecoder(
            feature_dims=self.encoder.feature_dims,
            spatial_shapes=self.encoder.spatial_shapes,
            layers=decoder_layers,
            layer_kwargs=decoder_layer_kwargs,
            shared_kwargs=D_kwargs,
        )

    def forward(self, x, keep_mask=None):
        encoder_latents = self.encoder(x, keep_mask=keep_mask)
        decoder_latents = self.decoder(encoder_latents)
        return decoder_latents, encoder_latents


class PredictiveHierarchicalAutoencoder(HierarchicalAutoencoder):
    def __init__(
        self,
        n_layers: int | None,
        d_layers: list[int],
        E_kwargs: dict | None = None,
        layers_E="ConvNet",
        layers_E_kwargs: list[dict] | None = None,
        D_kwargs: dict | None = None,
        layers_D=None,
        layers_D_kwargs: list[dict] | None = None,
        P_kwargs: dict | None = None,
        input_shape: tuple[int, int, int] | None = None,
    ):
        super().__init__(
            n_layers=n_layers,
            d_layers=d_layers,
            E_kwargs=E_kwargs,
            layers_E=layers_E,
            layers_E_kwargs=layers_E_kwargs,
            D_kwargs=D_kwargs,
            layers_D=layers_D,
            layers_D_kwargs=layers_D_kwargs,
            input_shape=input_shape,
        )
        predictor_kwargs = {} if P_kwargs is None else dict(P_kwargs)
        self.predictor = ViTPredictor(
            feature_dims=self.encoder.feature_dims,
            spatial_shapes=self.encoder.spatial_shapes,
            **predictor_kwargs,
        )

    def forward(self, x, keep_mask=None):
        encoder_latents = self.encoder(x, keep_mask=keep_mask)
        predicted_latents = self.predictor(encoder_latents)
        decoder_latents = self.decoder(encoder_latents, skip_latents=predicted_latents)
        return decoder_latents, encoder_latents, predicted_latents
