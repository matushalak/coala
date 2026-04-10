from torch import nn

from coala.architecture.decoders.decoder import HierarchicalDecoder
from coala.architecture.encoders.encoder import HierarchicalEncoder
from coala.architecture.predictor.vit_predictor import ViTPredictor

PREDICTOR_MODES = ("decoder", "predictor", "predictor+decoder")


def build_hae_modules(
    *,
    d_layers: list[int],
    E_kwargs: dict | None = None,
    layers_E="ConvNet",
    layers_E_kwargs: list[dict] | None = None,
    D_kwargs: dict | None = None,
    layers_D=None,
    layers_D_kwargs: list[dict] | None = None,
    input_shape: tuple[int, int, int] | None = None,
):
    assert input_shape is not None
    encoder = HierarchicalEncoder(
        input_shape=tuple(input_shape),
        d_layers=list(d_layers),
        layers=layers_E,
        layer_kwargs=layers_E_kwargs,
        shared_kwargs=E_kwargs,
    )
    decoder_layers = layers_E if layers_D is None else layers_D
    decoder_layer_kwargs = layers_E_kwargs if layers_D_kwargs is None else layers_D_kwargs
    decoder = HierarchicalDecoder(
        feature_dims=encoder.feature_dims,
        spatial_shapes=encoder.spatial_shapes,
        layers=decoder_layers,
        layer_kwargs=decoder_layer_kwargs,
        shared_kwargs=D_kwargs,
    )
    return encoder, decoder


def build_predictive_hae_modules(
    *,
    d_layers: list[int],
    predictor_mode: str = "predictor",
    E_kwargs: dict | None = None,
    layers_E="ConvNet",
    layers_E_kwargs: list[dict] | None = None,
    D_kwargs: dict | None = None,
    layers_D=None,
    layers_D_kwargs: list[dict] | None = None,
    P_kwargs: dict | None = None,
    input_shape: tuple[int, int, int] | None = None,
):
    assert predictor_mode in PREDICTOR_MODES
    encoder, decoder = build_hae_modules(
        d_layers=d_layers,
        E_kwargs=E_kwargs,
        layers_E=layers_E,
        layers_E_kwargs=layers_E_kwargs,
        D_kwargs=D_kwargs,
        layers_D=layers_D,
        layers_D_kwargs=layers_D_kwargs,
        input_shape=input_shape,
    )
    predictor = None
    if predictor_mode in {"predictor", "predictor+decoder"}:
        predictor_kwargs = {} if P_kwargs is None else dict(P_kwargs)
        predictor = ViTPredictor(
            feature_dims=encoder.feature_dims,
            spatial_shapes=encoder.spatial_shapes,
            **predictor_kwargs,
        )
    return encoder, decoder, predictor


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
        predictor_mode=config.get("predictor_mode", "predictor"),
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
        self.encoder, self.decoder = build_hae_modules(
            d_layers=self.feature_dims,
            E_kwargs=E_kwargs,
            layers_E=layers_E,
            layers_E_kwargs=layers_E_kwargs,
            D_kwargs=D_kwargs,
            layers_D=layers_D,
            layers_D_kwargs=layers_D_kwargs,
            input_shape=self.input_shape,
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
        predictor_mode: str = "predictor",
        input_shape: tuple[int, int, int] | None = None,
    ):
        nn.Module.__init__(self)
        assert input_shape is not None
        assert predictor_mode in PREDICTOR_MODES
        self.n_layers = len(d_layers) if n_layers is None else n_layers
        assert self.n_layers == len(d_layers)
        self.input_shape = tuple(input_shape)
        self.feature_dims = list(d_layers)
        self.predictor_mode = predictor_mode
        self.encoder, self.decoder, self.predictor = build_predictive_hae_modules(
            d_layers=self.feature_dims,
            predictor_mode=predictor_mode,
            E_kwargs=E_kwargs,
            layers_E=layers_E,
            layers_E_kwargs=layers_E_kwargs,
            D_kwargs=D_kwargs,
            layers_D=layers_D,
            layers_D_kwargs=layers_D_kwargs,
            P_kwargs=P_kwargs,
            input_shape=self.input_shape,
        )

    def encode(self, x, keep_mask=None):
        return self.encoder(x, keep_mask=keep_mask)

    def predict_from_encoded(self, encoder_latents):
        if self.predictor is None:
            return None
        return self.predictor(encoder_latents)

    def decode_from_encoded(self, encoder_latents, predictor_latents=None):
        skip_latents = predictor_latents if self.predictor_mode == "predictor+decoder" else None
        return self.decoder(encoder_latents, skip_latents=skip_latents)

    def prediction_outputs(self, x, keep_mask=None):
        encoder_latents = self.encoder(x, keep_mask=keep_mask)
        predictor_latents = self.predict_from_encoded(encoder_latents)
        decoder_latents = None
        if self.predictor_mode == "predictor":
            predicted_latents = predictor_latents
        else:
            decoder_latents = self.decode_from_encoded(encoder_latents, predictor_latents=predictor_latents)
            predicted_latents = decoder_latents
        return {
            "encoder_latents": encoder_latents,
            "predictor_latents": predictor_latents,
            "decoder_latents": decoder_latents,
            "predicted_latents": predicted_latents,
        }

    def forward(self, x, keep_mask=None):
        return self.prediction_outputs(x, keep_mask=keep_mask)
