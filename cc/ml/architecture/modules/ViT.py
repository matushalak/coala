import torch
import torch.nn as nn

from cc.ml.architecture.modules.ConvNet import SparseConv2d, UpConv2d
from cc.ml.architecture.modules.utils import (
    conv_output_shape,
    convtranspose_output_padding,
    downsample_center_mask,
    pair,
)


def _flatten_tokens(x: torch.Tensor) -> torch.Tensor:
    # The ViT family in this repo uses only spatial patch tokens.
    # No CLS token is appended anywhere in the encoder or decoder blocks.
    return x.flatten(2).transpose(1, 2)


def _restore_tokens(tokens: torch.Tensor, spatial_shape: tuple[int, int]) -> torch.Tensor:
    batch_size, _, channels = tokens.shape
    return tokens.transpose(1, 2).reshape(batch_size, channels, *spatial_shape)


class SparseTransformerBlock(nn.Module):
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

    def forward(self, x: torch.Tensor, keep_mask: torch.BoolTensor) -> torch.Tensor:
        tokens = _flatten_tokens(x)
        token_mask = keep_mask.squeeze(1).flatten(1)
        tokens = tokens * token_mask.unsqueeze(-1).to(dtype=tokens.dtype)
        y = self.norm1(tokens)
        y, _ = self.attn(y, y, y, key_padding_mask=~token_mask, need_weights=False)
        tokens = tokens + y
        tokens = tokens + self.mlp(self.norm2(tokens))
        tokens = tokens * token_mask.unsqueeze(-1).to(dtype=tokens.dtype)
        return _restore_tokens(tokens, x.shape[-2:])


class DenseTransformerBlock(nn.Module):
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = _flatten_tokens(x)
        y = self.norm1(tokens)
        y, _ = self.attn(y, y, y, need_weights=False)
        tokens = tokens + y
        tokens = tokens + self.mlp(self.norm2(tokens))
        return _restore_tokens(tokens, x.shape[-2:])


class ViTEncoderStage(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        input_spatial_shape: tuple[int, int],
        depth: int = 1,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        transition_kernel_size: int = 4,
        transition_stride: int = 4,
        transition_padding: int = 0,
        downsampling: str = "conv",
    ):
        super().__init__()
        assert downsampling == "conv"
        stride = pair(transition_stride)
        assert stride[0] == stride[1]
        self.output_spatial_shape = conv_output_shape(
            input_shape=input_spatial_shape,
            kernel_size=transition_kernel_size,
            stride=transition_stride,
            padding=transition_padding,
        )
        self.transition_stride = stride[0]
        self.patchify = SparseConv2d(
            in_channels,
            out_channels,
            kernel_size=transition_kernel_size,
            stride=transition_stride,
            padding=transition_padding,
        )
        self.blocks = nn.ModuleList(
            [
                SparseTransformerBlock(dim=out_channels, num_heads=num_heads, mlp_ratio=mlp_ratio)
                for _ in range(depth)
            ]
        )

    def forward(self, x: torch.Tensor, keep_mask: torch.BoolTensor) -> tuple[torch.Tensor, torch.BoolTensor]:
        out_mask = keep_mask
        if keep_mask.shape[-2:] != self.output_spatial_shape:
            out_mask = downsample_center_mask(keep_mask, self.output_spatial_shape, stride=self.transition_stride)
        x = self.patchify(x, out_mask)
        for block in self.blocks:
            x = block(x, out_mask)
        return x, out_mask


class ViTDecoderStage(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        input_spatial_shape: tuple[int, int],
        output_spatial_shape: tuple[int, int],
        depth: int = 1,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        transition_kernel_size: int = 4,
        transition_stride: int = 4,
        transition_padding: int = 0,
        upsampling: str = "upsample+conv",
    ):
        super().__init__()
        self.output_spatial_shape = output_spatial_shape
        if input_spatial_shape == output_spatial_shape:
            self.transition = None
        else:
            output_padding = convtranspose_output_padding(
                input_shape=input_spatial_shape,
                output_shape=output_spatial_shape,
                kernel_size=transition_kernel_size,
                stride=transition_stride,
                padding=transition_padding,
            )
            self.transition = UpConv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=transition_kernel_size,
                stride=transition_stride,
                padding=transition_padding,
                output_padding=output_padding,
                method=upsampling,
            )
        self.blocks = nn.ModuleList(
            [
                DenseTransformerBlock(dim=out_channels, num_heads=num_heads, mlp_ratio=mlp_ratio)
                for _ in range(depth)
            ]
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor | None = None) -> torch.Tensor:
        if self.transition is not None:
            x = self.transition(x)
        if skip is not None:
            x = x + skip
        for block in self.blocks:
            x = block(x)
        return x


def make_encoder_stage(**kwargs) -> ViTEncoderStage:
    return ViTEncoderStage(**kwargs)


def make_decoder_stage(**kwargs) -> ViTDecoderStage:
    return ViTDecoderStage(**kwargs)
