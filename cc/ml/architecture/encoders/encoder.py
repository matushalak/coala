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


class HierarchicalEncoder(nn.Module):
    def __init__(
        self,
        input_shape: tuple[int, int, int],
        d_layers: list[int],
        layers,
        layer_kwargs: list[dict] | None = None,
        shared_kwargs: dict | None = None,
    ):
        super().__init__()
        assert len(input_shape) == 3
        self.input_shape = tuple(input_shape)
        self.feature_dims = list(d_layers)
        self.n_layers = len(self.feature_dims)
        self.layer_modules = _expand_layers(layers, self.n_layers)
        self.layer_kwargs = _expand_layer_kwargs(layer_kwargs, self.n_layers)
        self.shared_kwargs = {} if shared_kwargs is None else dict(shared_kwargs)
        self.feature_names = [f"feat{i}" for i in range(self.n_layers)]
        self.mask_names = [f"mask{i}" for i in range(self.n_layers)]

        stages = []
        spatial_shapes = []
        in_channels = self.input_shape[0]
        spatial_shape = self.input_shape[1:]
        for out_channels, module_name, kwargs in zip(self.feature_dims, self.layer_modules, self.layer_kwargs):
            family = resolve_module_family(module_name)
            stage = family.make_encoder_stage(
                in_channels=in_channels,
                out_channels=out_channels,
                input_spatial_shape=spatial_shape,
                **self.shared_kwargs,
                **kwargs,
            )
            stages.append(stage)
            spatial_shapes.append(stage.output_spatial_shape)
            in_channels = out_channels
            spatial_shape = stage.output_spatial_shape

        self.stages = nn.ModuleList(stages)
        self.spatial_shapes = spatial_shapes

    def forward(
        self,
        x: torch.Tensor,
        keep_mask: torch.BoolTensor | None = None,
    ) -> dict[str, torch.Tensor]:
        assert x.ndim == 4
        assert tuple(x.shape[1:]) == self.input_shape
        if keep_mask is None:
            keep_mask = torch.ones((x.shape[0], 1, *self.input_shape[1:]), device=x.device, dtype=torch.bool)
        assert keep_mask.shape == (x.shape[0], 1, *self.input_shape[1:])
        x = x * keep_mask.to(dtype=x.dtype)

        latents: dict[str, torch.Tensor] = {}
        for index, stage in enumerate(self.stages):
            x, keep_mask = stage(x, keep_mask)
            latents[self.feature_names[index]] = x
            latents[self.mask_names[index]] = keep_mask
        return latents
