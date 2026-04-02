import torch
import torch.nn as nn

from cc.ml.architecture.modules import resolve_module_family


def _expand_layers(layers, n_layers: int) -> list:
    if isinstance(layers, str):
        return [layers] * n_layers
    assert len(layers) == n_layers
    return list(layers)


def _expand_layer_kwargs(layer_kwargs, n_layers: int) -> list[dict]:
    if layer_kwargs is None:
        return [{} for _ in range(n_layers)]
    assert len(layer_kwargs) == n_layers
    return [dict(kwargs) for kwargs in layer_kwargs]


class HierarchicalDecoder(nn.Module):
    DENSIFY_MODES = ("random", "token", "zero")

    def __init__(
        self,
        feature_dims: list[int],
        spatial_shapes: list[tuple[int, int]],
        layers,
        layer_kwargs: list[dict] | None = None,
        shared_kwargs: dict | None = None,
    ):
        super().__init__()
        assert len(feature_dims) == len(spatial_shapes)
        self.feature_dims = list(feature_dims)
        self.spatial_shapes = list(spatial_shapes)
        self.n_layers = len(self.feature_dims)
        self.feature_names = [f"feat{i}" for i in range(self.n_layers)]
        self.mask_names = [f"mask{i}" for i in range(self.n_layers)]

        shared_kwargs = {} if shared_kwargs is None else dict(shared_kwargs)
        self.use_skip = bool(shared_kwargs.pop("use_skip", False))
        self.densify_mode = str(shared_kwargs.pop("densify_mode", "random"))
        assert self.densify_mode in self.DENSIFY_MODES
        self.shared_kwargs = shared_kwargs
        self.layer_modules = _expand_layers(layers, self.n_layers)
        self.layer_kwargs = _expand_layer_kwargs(layer_kwargs, self.n_layers)

        self.mask_tokens = nn.ParameterDict()
        for index, channels in enumerate(self.feature_dims):
            token = nn.Parameter(torch.zeros(1, channels, 1, 1))
            nn.init.normal_(token, mean=0.0, std=0.02)
            self.mask_tokens[self.feature_names[index]] = token

        stages = []
        for index in range(self.n_layers):
            output_index = self.n_layers - 1 - index
            source_index = min(self.n_layers - 1, output_index + 1)
            family = resolve_module_family(self.layer_modules[source_index])
            in_channels = self.feature_dims[source_index]
            out_channels = self.feature_dims[output_index]
            input_spatial_shape = self.spatial_shapes[source_index]
            output_spatial_shape = self.spatial_shapes[output_index]
            stage = family.make_decoder_stage(
                in_channels=in_channels,
                out_channels=out_channels,
                input_spatial_shape=input_spatial_shape,
                output_spatial_shape=output_spatial_shape,
                **self.shared_kwargs,
                **self.layer_kwargs[source_index],
            )
            stages.append(stage)
        self.stages = nn.ModuleList(stages)

    def _masked_fill(self, feat: torch.Tensor, mask_token: torch.Tensor) -> torch.Tensor:
        if self.densify_mode == "token":
            return mask_token.expand_as(feat)
        if self.densify_mode == "zero":
            return torch.zeros_like(feat)
        return torch.randn_like(feat)

    def _densify(
        self,
        feat: torch.Tensor,
        keep_mask: torch.BoolTensor,
        mask_token: torch.Tensor,
    ) -> torch.Tensor:
        fill = self._masked_fill(feat, mask_token)
        return torch.where(keep_mask.expand_as(feat), feat, fill)

    def forward(
        self,
        encoder_latents: dict[str, torch.Tensor],
        skip_latents: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        top_name = self.feature_names[-1]
        top_mask_name = self.mask_names[-1]
        x = self._densify(
            encoder_latents[top_name],
            encoder_latents[top_mask_name],
            self.mask_tokens[top_name],
        )

        decoded: dict[str, torch.Tensor] = {}
        for stage_index, stage in enumerate(self.stages):
            output_index = self.n_layers - 1 - stage_index
            skip = None
            if stage_index > 0 and self.use_skip:
                skip_name = self.feature_names[output_index]
                skip_source = encoder_latents if skip_latents is None else skip_latents
                if skip_name in skip_source:
                    skip = skip_source[skip_name]
                    mask_name = self.mask_names[output_index]
                    if mask_name in skip_source:
                        skip = self._densify(skip, skip_source[mask_name], self.mask_tokens[skip_name])
                    elif mask_name in encoder_latents:
                        skip = self._densify(skip, encoder_latents[mask_name], self.mask_tokens[skip_name])
            x = stage(x, skip=skip)
            decoded[self.feature_names[output_index]] = x
        return decoded
