from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

import coala.architecture.modules.norms as norms
from coala.architecture.modules.utils import (
    conv_output_shape,
    convtranspose_output_padding,
    downsample_center_mask,
    pair,
    sp_conv_forward,
)


class SparseConv2d(nn.Conv2d):
    forward = sp_conv_forward


class SparseDownsamplingModule(nn.Module):
    '''
    Downsampling convolution module (from lower visual area to higher visual area). 
        Supports masked pretraining which can simply be ignored by 
            always providing a mask full of ones.
    '''
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 2,
        padding: int = 1,
    ):
        super().__init__()
        self.transition = SparseConv2d(in_channels, out_channels, kernel_size, stride, padding)

    def forward(self, x: torch.Tensor, keep_mask: torch.BoolTensor) -> torch.Tensor:
        return self.transition(x, keep_mask)


class SparseResidualMLP(nn.Module):
    '''
    Local processing block within a cortical column (feature refinement at each spatial location by local circuits), 
        implemented as a residual convolutional block with sparse convolution and normalization.
    
    All spatial mixing is done in the Downsampling Module, this block solely performs local feature mixing with 1x1 convolutinos
        (equivalent to shared MLP across spatial locations). I.e. each cortical column is an MLP.
    '''
    def __init__(self,n_channels: int,):
        super().__init__()
        self.conv1 = SparseConv2d(n_channels, n_channels * 4, kernel_size=1, stride=1, padding=0)
        self.grn = norms.SparseGlobalResponseNorm(n_channels * 4)
        self.conv2 = SparseConv2d(n_channels * 4, n_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x: torch.Tensor, keep_mask: torch.BoolTensor) -> torch.Tensor:
        mask = keep_mask.to(dtype=x.dtype)
        y = self.conv1(x, keep_mask)
        y = self.activation_fn(y)
        y = self.grn(y, keep_mask)
        y = y * mask
        y = self.conv2(y, keep_mask)
        return x + y


class ResidualMLP(nn.Module):
    def __init__(self,n_channels: int,):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(n_channels, n_channels * 4, kernel_size=1, stride=1, padding=0),
            self.activation_fn,
            norms.GlobalResponseNorm(n_channels * 4),
            nn.Conv2d(n_channels * 4, n_channels, kernel_size=1, stride=1, padding=0)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class UpConv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
        output_padding: int | tuple[int, int],
        method: Literal["transposed_conv", "upsample+conv"] = "transposed_conv",
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
            raise NotImplementedError(f"Upsampling method {method} not supported anymore.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.upconv(x)


class ConvNetEncoderStage(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        input_spatial_shape: tuple[int, int],
        depth: int = 1,
        transition_kernel_size: int = 3,
        transition_stride: int = 2,
        transition_padding: int | None = None,
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
        )
        self.local_stage = nn.Sequential(*[
            SparseResidualMLP(out_channels) for _ in range(depth)
        ])

        self.activation_fn = nn.ReLU()

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
        local_stage:nn.Sequential,
        transition_kernel_size: int = 3,
        transition_stride: int = 2,
        transition_padding: int | None = None,
        upsampling: Literal["transposed_conv", "upsample+conv"] = "transposed_conv",
    ):
        super().__init__()
        transition_padding = transition_kernel_size // 2 if transition_padding is None else transition_padding
        self.output_spatial_shape = output_spatial_shape
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
        self.local_stage = local_stage

        self.activation_fn = nn.ReLU()

    def forward(self, x: torch.Tensor, skip: torch.Tensor | None) -> torch.Tensor:
        x = self.transition(x)
        if skip is not None:
            x = x + skip
        x = self.activation_fn(x)
        return self.local_stage(x)


def make_encoder_stage(**kwargs) -> ConvNetEncoderStage:
    return ConvNetEncoderStage(**kwargs)


def make_decoder_stage(**kwargs) -> ConvNetDecoderStage:
    return ConvNetDecoderStage(**kwargs)
