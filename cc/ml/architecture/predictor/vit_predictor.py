import torch
import torch.nn as nn

from cc.ml.architecture.modules.utils import sorted_stage_keys


def _flatten_tokens(x: torch.Tensor) -> torch.Tensor:
    return x.flatten(2).transpose(1, 2)


def _restore_tokens(tokens: torch.Tensor, spatial_shape: tuple[int, int]) -> torch.Tensor:
    batch_size, _, channels = tokens.shape
    return tokens.transpose(1, 2).reshape(batch_size, channels, *spatial_shape)


class PredictorBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        assert dim % num_heads == 0
        hidden_dim = int(dim * mlp_ratio)
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, tokens: torch.Tensor, token_mask: torch.BoolTensor) -> torch.Tensor:
        y = self.norm1(tokens)
        y, _ = self.attn(y, y, y, key_padding_mask=~token_mask, need_weights=False)
        tokens = tokens + y
        tokens = tokens + self.mlp(self.norm2(tokens))
        tokens = tokens * token_mask.unsqueeze(-1).to(dtype=tokens.dtype)
        return tokens


def _build_projection(
    in_channels: int,
    out_channels: int,
    input_shape: tuple[int, int],
    output_shape: tuple[int, int],
) -> nn.Module:
    if input_shape == output_shape:
        return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
    scale_h = input_shape[0] // output_shape[0]
    scale_w = input_shape[1] // output_shape[1]
    assert input_shape[0] % output_shape[0] == 0
    assert input_shape[1] % output_shape[1] == 0
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=(scale_h, scale_w),
        stride=(scale_h, scale_w),
        padding=0,
    )


def _build_unprojection(
    in_channels: int,
    out_channels: int,
    input_shape: tuple[int, int],
    output_shape: tuple[int, int],
) -> nn.Module:
    if input_shape == output_shape:
        return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
    scale_h = output_shape[0] // input_shape[0]
    scale_w = output_shape[1] // input_shape[1]
    assert output_shape[0] % input_shape[0] == 0
    assert output_shape[1] % input_shape[1] == 0
    return nn.ConvTranspose2d(
        in_channels,
        out_channels,
        kernel_size=(scale_h, scale_w),
        stride=(scale_h, scale_w),
        padding=0,
    )


class ViTPredictor(nn.Module):
    def __init__(
        self,
        feature_dims: list[int],
        spatial_shapes: list[tuple[int, int]],
        predictor_dim: int = 256,
        depth: int = 2,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        assert len(feature_dims) == len(spatial_shapes)
        self.feature_names = [f"feat{i}" for i in range(len(feature_dims))]
        self.target_shape = spatial_shapes[-1]
        self.target_tokens = self.target_shape[0] * self.target_shape[1]

        self.input_projections = nn.ModuleDict(
            {
                name: _build_projection(
                    in_channels=channels,
                    out_channels=predictor_dim,
                    input_shape=shape,
                    output_shape=self.target_shape,
                )
                for name, channels, shape in zip(self.feature_names, feature_dims, spatial_shapes)
            }
        )
        self.merge = nn.Conv2d(
            predictor_dim * len(self.feature_names),
            predictor_dim,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.pos_embed = nn.Parameter(torch.zeros(1, self.target_tokens, predictor_dim))
        nn.init.normal_(self.pos_embed, mean=0.0, std=0.02)
        self.blocks = nn.ModuleList(
            [
                PredictorBlock(dim=predictor_dim, num_heads=num_heads, mlp_ratio=mlp_ratio)
                for _ in range(depth)
            ]
        )
        self.output_projections = nn.ModuleDict(
            {
                name: _build_unprojection(
                    in_channels=predictor_dim,
                    out_channels=channels,
                    input_shape=self.target_shape,
                    output_shape=shape,
                )
                for name, channels, shape in zip(self.feature_names, feature_dims, spatial_shapes)
            }
        )

    def forward(self, latents: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        feature_names = sorted_stage_keys(latents, "feat")
        assert feature_names == self.feature_names
        merged = torch.cat([self.input_projections[name](latents[name]) for name in feature_names], dim=1)
        merged = self.merge(merged)

        tokens = _flatten_tokens(merged) + self.pos_embed
        token_mask_name = feature_names[-1].replace("feat", "mask")
        if token_mask_name in latents:
            token_mask = latents[token_mask_name].squeeze(1).flatten(1)
        else:
            token_mask = torch.ones((tokens.shape[0], tokens.shape[1]), device=tokens.device, dtype=torch.bool)

        for block in self.blocks:
            tokens = block(tokens, token_mask)

        merged = _restore_tokens(tokens, self.target_shape)
        return {name: self.output_projections[name](merged) for name in feature_names}
