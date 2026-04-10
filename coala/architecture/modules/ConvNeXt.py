import torch
import torch.nn as nn

import coala.architecture.modules.norms as norms
from coala.architecture.modules.ConvNet import SparseConv2d, SparseDownsamplingModule, UpConv2d
from coala.architecture.modules.utils import (
    conv_output_shape,
    convtranspose_output_padding,
    downsample_center_mask,
    pair,
)


class SparseConvNeXtBlock(nn.Module):
    def __init__(
        self,
        n_channels: int,
        spatial_dim: tuple[int, int],
        spatial_kernel_size: int = 7,
        norm_type: str = "rmsnorm",
    ):
        super().__init__()
        assert spatial_kernel_size % 2 == 1
        padding = spatial_kernel_size // 2
        self.dwconv = SparseConv2d(
            n_channels,
            n_channels,
            kernel_size=spatial_kernel_size,
            stride=1,
            padding=padding,
            groups=n_channels,
        )
        self.norm = norms.SparseNorm2d(norm_type=norm_type, normalized_shape=(n_channels, *spatial_dim))
        self.pwconv1 = SparseConv2d(n_channels, n_channels * 4, kernel_size=1, stride=1, padding=0)
        self.act = nn.GELU()
        self.grn = norms.SparseGlobalResponseNorm(n_channels * 4)
        self.pwconv2 = SparseConv2d(n_channels * 4, n_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x: torch.Tensor, keep_mask: torch.BoolTensor) -> torch.Tensor:
        mask = keep_mask.to(dtype=x.dtype)
        y = self.dwconv(x, keep_mask)
        y = self.norm(y, keep_mask)
        y = self.pwconv1(y, keep_mask)
        y = self.act(y)
        y = self.grn(y, keep_mask)
        y = y * mask
        y = self.pwconv2(y, keep_mask)
        return x + y


class DenseConvNeXtBlock(nn.Module):
    def __init__(
        self,
        n_channels: int,
        spatial_dim: tuple[int, int],
        spatial_kernel_size: int = 7,
        norm_type: str = "rmsnorm",
    ):
        super().__init__()
        assert spatial_kernel_size % 2 == 1
        padding = spatial_kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(
                n_channels,
                n_channels,
                kernel_size=spatial_kernel_size,
                stride=1,
                padding=padding,
                groups=n_channels,
            ),
            norms.DenseNorm2d(norm_type=norm_type, normalized_shape=(n_channels, *spatial_dim)),
            nn.Conv2d(n_channels, n_channels * 4, kernel_size=1, stride=1, padding=0),
            nn.GELU(),
            norms.GlobalResponseNorm(n_channels * 4),
            nn.Conv2d(n_channels * 4, n_channels, kernel_size=1, stride=1, padding=0),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class ConvNeXtEncoderStage(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        input_spatial_shape: tuple[int, int],
        depth: int = 1,
        norm_type: str = "rmsnorm",
        transition_kernel_size: int = 2,
        transition_stride: int = 2,
        transition_padding: int = 0,
        downsampling: str = "conv",
        spatial_kernel_size: int = 7,
    ):
        super().__init__()
        stride = pair(transition_stride)
        assert stride[0] == stride[1]
        self.output_spatial_shape = conv_output_shape(
            input_shape=input_spatial_shape,
            kernel_size=transition_kernel_size,
            stride=transition_stride,
            padding=transition_padding,
        )
        self.transition_stride = stride[0]
        self.transition = SparseDownsamplingModule(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=transition_kernel_size,
            stride=transition_stride,
            padding=transition_padding,
            downsampling=downsampling,
        )
        self.blocks = nn.ModuleList(
            [
                SparseConvNeXtBlock(
                    n_channels=out_channels,
                    spatial_dim=self.output_spatial_shape,
                    spatial_kernel_size=spatial_kernel_size,
                    norm_type=norm_type,
                )
                for _ in range(depth)
            ]
        )
        self.out_norm = norms.SparseNorm2d(norm_type=norm_type, normalized_shape=(out_channels, *self.output_spatial_shape))

    def forward(self, x: torch.Tensor, keep_mask: torch.BoolTensor) -> tuple[torch.Tensor, torch.BoolTensor]:
        out_mask = keep_mask
        if keep_mask.shape[-2:] != self.output_spatial_shape:
            out_mask = downsample_center_mask(keep_mask, self.output_spatial_shape, stride=self.transition_stride)
        x = self.transition(x, out_mask)
        for block in self.blocks:
            x = block(x, out_mask)
        x = self.out_norm(x, out_mask)
        return x, out_mask


class ConvNeXtDecoderStage(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        input_spatial_shape: tuple[int, int],
        output_spatial_shape: tuple[int, int],
        depth: int = 1,
        norm_type: str = "rmsnorm",
        transition_kernel_size: int = 2,
        transition_stride: int = 2,
        transition_padding: int = 0,
        upsampling: str = "upsample+conv",
        spatial_kernel_size: int = 7,
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
                DenseConvNeXtBlock(
                    n_channels=out_channels,
                    spatial_dim=output_spatial_shape,
                    spatial_kernel_size=spatial_kernel_size,
                    norm_type=norm_type,
                )
                for _ in range(depth)
            ]
        )
        self.out_norm = norms.DenseNorm2d(norm_type=norm_type, normalized_shape=(out_channels, *output_spatial_shape))

    def forward(self, x: torch.Tensor, skip: torch.Tensor | None = None) -> torch.Tensor:
        if self.transition is not None:
            x = self.transition(x)
        if skip is not None:
            x = x + skip
        for block in self.blocks:
            x = block(x)
        x = self.out_norm(x)
        return x


def make_encoder_stage(**kwargs) -> ConvNeXtEncoderStage:
    return ConvNeXtEncoderStage(**kwargs)


def make_decoder_stage(**kwargs) -> ConvNeXtDecoderStage:
    return ConvNeXtDecoderStage(**kwargs)
