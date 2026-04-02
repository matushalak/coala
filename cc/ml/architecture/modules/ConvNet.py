from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

import cc.ml.architecture.modules.norms as norms
from cc.ml.architecture.modules.utils import (
    conv_output_shape,
    convtranspose_output_padding,
    downsample_center_mask,
    pair,
    sp_conv_forward,
)


class SparseConv2d(nn.Conv2d):
    forward = sp_conv_forward


class SparseMaxPooling(nn.MaxPool2d):
    forward = sp_conv_forward


class SparseAvgPooling(nn.AvgPool2d):
    forward = sp_conv_forward


class SparseDownsamplingModule(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 2,
        padding: int = 1,
        downsampling: Literal["maxpool", "avgpool", "conv"] = "conv",
    ):
        super().__init__()
        self.downsampling = downsampling
        self.preconv = None
        if downsampling == "conv":
            self.transition = SparseConv2d(in_channels, out_channels, kernel_size, stride, padding)
        else:
            pool_cls = SparseMaxPooling if downsampling == "maxpool" else SparseAvgPooling
            self.transition = pool_cls(kernel_size, stride, padding)
            if in_channels != out_channels:
                self.preconv = SparseConv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x: torch.Tensor, keep_mask: torch.BoolTensor) -> torch.Tensor:
        if self.preconv is not None:
            x = self.preconv(x, keep_mask)
        return self.transition(x, keep_mask)


class SparseResidualConv2d(nn.Module):
    def __init__(
        self,
        n_channels: int,
        spatial_dim: tuple[int, int],
        kernel_size: int = 3,
        norm_type: str = "rmsnorm",
    ):
        super().__init__()
        assert kernel_size % 2 == 1
        padding = kernel_size // 2
        self.conv1 = SparseConv2d(n_channels, n_channels, kernel_size=kernel_size, stride=1, padding=padding)
        self.norm1 = norms.SparseNorm2d(norm_type=norm_type, normalized_shape=(n_channels, *spatial_dim))
        self.conv2 = SparseConv2d(n_channels, n_channels * 4, kernel_size=1, stride=1, padding=0)
        self.act = nn.GELU()
        self.grn = norms.SparseGlobalResponseNorm(n_channels * 4)
        self.conv3 = SparseConv2d(n_channels * 4, n_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x: torch.Tensor, keep_mask: torch.BoolTensor) -> torch.Tensor:
        mask = keep_mask.to(dtype=x.dtype)
        y = self.conv1(x, keep_mask)
        y = self.norm1(y, keep_mask)
        y = self.conv2(y, keep_mask)
        y = self.act(y)
        y = self.grn(y, keep_mask)
        y = y * mask
        y = self.conv3(y, keep_mask)
        return x + y


class ResidualConv2d(nn.Module):
    def __init__(
        self,
        n_channels: int,
        spatial_dim: tuple[int, int],
        kernel_size: int = 3,
        norm_type: str = "rmsnorm",
    ):
        super().__init__()
        assert kernel_size % 2 == 1
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(n_channels, n_channels, kernel_size=kernel_size, stride=1, padding=padding),
            norms.DenseNorm2d(norm_type=norm_type, normalized_shape=(n_channels, *spatial_dim)),
            nn.Conv2d(n_channels, n_channels * 4, kernel_size=1, stride=1, padding=0),
            nn.GELU(),
            norms.GlobalResponseNorm(n_channels * 4),
            nn.Conv2d(n_channels * 4, n_channels, kernel_size=1, stride=1, padding=0),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class SparseLocalStage(nn.Module):
    def __init__(
        self,
        n_channels: int,
        spatial_dim: tuple[int, int],
        depth: int = 1,
        use_residual: bool = True,
        kernel_size: int = 3,
        norm_type: str = "rmsnorm",
    ):
        super().__init__()
        self.pre_norm = norms.SparseNorm2d(norm_type=norm_type, normalized_shape=(n_channels, *spatial_dim))
        self.act = nn.GELU()
        self.blocks = nn.ModuleList(
            [
                SparseResidualConv2d(
                    n_channels=n_channels,
                    spatial_dim=spatial_dim,
                    kernel_size=kernel_size,
                    norm_type=norm_type,
                )
                for _ in range(depth if use_residual else 0)
            ]
        )
        self.post_norm = norms.SparseNorm2d(norm_type=norm_type, normalized_shape=(n_channels, *spatial_dim))

    def forward(self, x: torch.Tensor, keep_mask: torch.BoolTensor) -> torch.Tensor:
        x = self.pre_norm(x, keep_mask)
        x = self.act(x)
        x = x * keep_mask.to(dtype=x.dtype)
        for block in self.blocks:
            x = block(x, keep_mask)
        x = self.post_norm(x, keep_mask)
        return x


class LocalStage(nn.Module):
    def __init__(
        self,
        n_channels: int,
        spatial_dim: tuple[int, int],
        depth: int = 1,
        use_residual: bool = True,
        kernel_size: int = 3,
        norm_type: str = "rmsnorm",
    ):
        super().__init__()
        self.pre_norm = norms.DenseNorm2d(norm_type=norm_type, normalized_shape=(n_channels, *spatial_dim))
        self.act = nn.GELU()
        self.blocks = nn.ModuleList(
            [
                ResidualConv2d(
                    n_channels=n_channels,
                    spatial_dim=spatial_dim,
                    kernel_size=kernel_size,
                    norm_type=norm_type,
                )
                for _ in range(depth if use_residual else 0)
            ]
        )
        self.post_norm = norms.DenseNorm2d(norm_type=norm_type, normalized_shape=(n_channels, *spatial_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pre_norm(x)
        x = self.act(x)
        for block in self.blocks:
            x = block(x)
        x = self.post_norm(x)
        return x


class UpConv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
        output_padding: int | tuple[int, int],
        method: Literal["transposed_conv", "upsample+conv"] = "upsample+conv",
    ):
        super().__init__()
        self.kernel_size = pair(kernel_size)
        self.stride = pair(stride)
        self.padding = pair(padding)
        self.output_padding = pair(output_padding)
        self.method = method
        if method == "transposed_conv":
            self.upconv = nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=self.padding,
                output_padding=self.output_padding,
            )
        else:
            self.upconv = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=self.kernel_size,
                stride=1,
                padding=(self.kernel_size[0] // 2, self.kernel_size[1] // 2),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.method == "upsample+conv":
            target_h = (
                (x.shape[-2] - 1) * self.stride[0]
                - 2 * self.padding[0]
                + self.kernel_size[0]
                + self.output_padding[0]
            )
            target_w = (
                (x.shape[-1] - 1) * self.stride[1]
                - 2 * self.padding[1]
                + self.kernel_size[1]
                + self.output_padding[1]
            )
            x = F.interpolate(x, size=(target_h, target_w), mode="bilinear", align_corners=True)
            x = self.upconv(x)
            return x[..., :target_h, :target_w]
        return self.upconv(x)


class ConvNetEncoderStage(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        input_spatial_shape: tuple[int, int],
        depth: int = 1,
        use_residual: bool = True,
        norm_type: str = "rmsnorm",
        transition_kernel_size: int = 3,
        transition_stride: int = 2,
        transition_padding: int | None = None,
        downsampling: Literal["maxpool", "avgpool", "conv"] = "conv",
        block_kernel_size: int = 3,
    ):
        super().__init__()
        transition_padding = transition_kernel_size // 2 if transition_padding is None else transition_padding
        stride = pair(transition_stride)
        assert stride[0] == stride[1]
        self.output_spatial_shape = conv_output_shape(
            input_shape=input_spatial_shape,
            kernel_size=transition_kernel_size,
            stride=stride,
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
        self.local_stage = SparseLocalStage(
            n_channels=out_channels,
            spatial_dim=self.output_spatial_shape,
            depth=depth,
            use_residual=use_residual,
            kernel_size=block_kernel_size,
            norm_type=norm_type,
        )

    def forward(self, x: torch.Tensor, keep_mask: torch.BoolTensor) -> tuple[torch.Tensor, torch.BoolTensor]:
        out_mask = keep_mask
        if keep_mask.shape[-2:] != self.output_spatial_shape:
            out_mask = downsample_center_mask(keep_mask, self.output_spatial_shape, stride=self.transition_stride)
        x = self.transition(x, out_mask)
        x = self.local_stage(x, out_mask)
        return x, out_mask


class ConvNetDecoderStage(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        input_spatial_shape: tuple[int, int],
        output_spatial_shape: tuple[int, int],
        depth: int = 1,
        use_residual: bool = True,
        norm_type: str = "rmsnorm",
        transition_kernel_size: int = 3,
        transition_stride: int = 2,
        transition_padding: int | None = None,
        upsampling: Literal["transposed_conv", "upsample+conv"] = "upsample+conv",
        block_kernel_size: int = 3,
    ):
        super().__init__()
        transition_padding = transition_kernel_size // 2 if transition_padding is None else transition_padding
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
        self.local_stage = LocalStage(
            n_channels=out_channels,
            spatial_dim=output_spatial_shape,
            depth=depth,
            use_residual=use_residual,
            kernel_size=block_kernel_size,
            norm_type=norm_type,
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor | None = None) -> torch.Tensor:
        if self.transition is not None:
            x = self.transition(x)
        if skip is not None:
            x = x + skip
        return self.local_stage(x)


def make_encoder_stage(**kwargs) -> ConvNetEncoderStage:
    return ConvNetEncoderStage(**kwargs)


def make_decoder_stage(**kwargs) -> ConvNetDecoderStage:
    return ConvNetDecoderStage(**kwargs)
