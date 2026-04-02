import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Literal

from cc.ml.architecture.modules.utils import sp_conv_forward
import cc.ml.architecture.modules.norms as norms

### Sparse versions - ENCODER side
class SparseConv2d(nn.Conv2d):
    forward = sp_conv_forward


class SparseMaxPooling(nn.MaxPool2d):
    forward = sp_conv_forward


class SparseAvgPooling(nn.AvgPool2d):
    forward = sp_conv_forward


class SparseDownsamplingModule(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, padding=1,
                 downsampling:Literal['maxpool', 'avgpool', 'conv'] = 'conv'):
        super(SparseDownsamplingModule, self).__init__()
        if downsampling == 'conv':
            self.downsample = SparseConv2d(in_channels, out_channels, kernel_size, stride, padding)
        elif 'pool' in downsampling:
            if downsampling == 'maxpool':
                downsample = SparseMaxPooling(kernel_size, stride, padding)
            elif downsampling == 'avgpool':
                downsample = SparseAvgPooling(kernel_size, stride, padding)
            # 1d conv to grow channel dimension
            if in_channels != out_channels:
                conv1x1 =  SparseConv2d(in_channels, out_channels, kernel_size=1)
                # first expand features, then pool expanded features
                downsample = nn.Sequential(conv1x1, downsample)
            self.downsample = downsample
        else:
            raise ValueError("Unsupported downsampling method")

    def forward(self, x:torch.Tensor, keepmask:torch.BoolTensor):
        return self.downsample(x, keepmask)


class SparseResidualConv2d(nn.Module):
    """
    Residual block that keeps sparse positions zeroed after each operation.
    """

    def __init__(self, n_channels: int, spatial_dim: tuple[int, int], norm_type: str = "rmsnorm"):
        super().__init__()
        self.conv1 = SparseConv2d(n_channels, n_channels, kernel_size=3, padding="same", stride=1)
        self.norm1 = norms.SparseNorm2d(norm_type=norm_type, normalized_shape=(n_channels, *spatial_dim))
        # Inverted bottleneck (convnext-style)
        self.conv2 = SparseConv2d(n_channels, n_channels*4, kernel_size=1, padding="same", stride=1)
        self.act = nn.GELU()
        self.grn = norms.SparseGlobalResponseNorm(n_channels*4) # global response norm from convnextv2 paper
        self.conv3 = SparseConv2d(n_channels*4, n_channels, kernel_size=1, padding="same", stride=1)

    def forward(self, x: torch.Tensor, keep_mask: torch.BoolTensor) -> torch.Tensor:
        mask = keep_mask.to(dtype=x.dtype)
        y = self.conv1(x, keep_mask)
        y = self.norm1(y, keep_mask)
        y = self.conv2(y, keep_mask)
        y = self.act(y)
        y = self.grn(y, keep_mask) # calibrate channels with GRN
        y = y * mask  # GELU can re-introduce non-zero values on inactive positions.
        y = self.conv3(y, keep_mask)
        return x + y # residual connection


class SparseLocalStage(nn.Module):
    """
    Local (same-resolution) sparse processing.

    Stage contract:
    - input/output keep the same spatial resolution
    - inactive positions remain exactly zero
    """

    def __init__(
        self,
        n_channels: int,
        spatial_dim: tuple[int, int],
        norm_type: str = "rmsnorm",
    ):
        super().__init__()
        self.pre_norm = norms.SparseNorm2d(norm_type=norm_type, normalized_shape=(n_channels, *spatial_dim))
        self.act = nn.GELU()
        self.block = SparseResidualConv2d(n_channels, spatial_dim=spatial_dim, norm_type=norm_type)
        self.post_norm = norms.SparseNorm2d(norm_type=norm_type, normalized_shape=(n_channels, *spatial_dim))

    def forward(self, x: torch.Tensor, keep_mask: torch.BoolTensor) -> torch.Tensor:
        x = self.pre_norm(x, keep_mask)
        x = self.act(x)
        x = x * keep_mask.to(dtype=x.dtype)
        x = self.block(x, keep_mask)
        x = self.act(x)
        x = self.post_norm(x, keep_mask)
        return x


class SparseConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding,
                 spatial_dim:tuple[int, int],
                 downsampling:Literal['maxpool', 'avgpool', 'conv']='conv'):
        super(SparseConvBlock, self).__init__()
        # Kernel size, stride and padding should lead to  spatial_dim output after downsample
        self.downsample = SparseDownsamplingModule(in_channels, out_channels, kernel_size, 
                                                   stride, padding, downsampling)
        self.local_stage = SparseLocalStage(out_channels, spatial_dim=spatial_dim)

    def forward(self, x:torch.Tensor, keepmask:torch.BoolTensor):
        x = self.downsample(x, keepmask)
        x = self.local_stage(x, keepmask)
        return x
    

### Dense versions - DECODER side (no keep_mask needed)
class UpConv2d(nn.Module):
    """
    Upsampling for decoder dense feature maps.
    Can be implemented as:
        - transposed convolution (learned projection weights from low-res to high-res)
        - upsameple (nearest / bilinear) + regular convolution (learned weights for FB inputs)
    """

    def __init__(self, in_channels: int, out_channels: int, 
                 kernel_size: int, stride: int, padding: int, output_padding: int,
                 method: Literal['transposed_conv', 'upsample+conv'] = "upsample+conv"):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        self.stride = (stride, stride) if isinstance(stride, int) else stride
        self.padding = (padding, padding) if isinstance(padding, int) else padding
        self.output_padding = (
            (output_padding, output_padding) if isinstance(output_padding, int) else output_padding
        )
        self.method = method
        if method == "transposed_conv":
            self.upconv = nn.ConvTranspose2d(
                in_channels, out_channels, kernel_size=kernel_size, padding=padding, 
                stride=stride, output_padding=output_padding
            )
        elif method == "upsample+conv":
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size//2)
            self.upconv = self.conv
        else:
            raise ValueError(f"Unsupported upsampling method: {method!r}")
        
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
        return self.upconv(x)


class ResidualConv2d(nn.Module):
    """Standard residual block for decoder dense feature maps."""
    def __init__(self, n_channels: int, spatial_dim: tuple[int, int], norm_type: str = "rmsnorm"):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(n_channels, n_channels, kernel_size=3, padding="same", stride=1),
            norms.DenseNorm2d(norm_type=norm_type, normalized_shape=(n_channels, *spatial_dim)),
            nn.Conv2d(n_channels, n_channels*4, kernel_size=1, padding="same", stride=1),
            nn.GELU(),
            norms.GlobalResponseNorm(n_channels*4), # global response norm from convnextv2 paper
            nn.Conv2d(n_channels*4, n_channels, kernel_size=1, padding="same", stride=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class LocalStage(nn.Module):
    """
    Local (same-resolution) dense processing for decoder feature maps after densification.

    Stage contract:
    - input/output keep the same spatial resolution
    - standard dense operations (e.g., LayerNorm, residuals) can be used without masking
    """

    def __init__(
        self,
        n_channels: int,
        spatial_dim: tuple[int, int],
        norm_type: str = "rmsnorm",
    ):
        super().__init__()
        self.n_channels = n_channels
        self.pre_norm = norms.DenseNorm2d(norm_type=norm_type, normalized_shape=(n_channels, *spatial_dim))
        self.act = nn.GELU()
        self.block = ResidualConv2d(n_channels, spatial_dim=spatial_dim, norm_type=norm_type)
        self.post_norm = norms.DenseNorm2d(norm_type=norm_type, normalized_shape=(n_channels, *spatial_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pre_norm(x)
        x = self.act(x)
        if self.block is not None:
            x = self.block(x)
        x = self.post_norm(x)
        return x


class UpConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding,
                 spatial_dim:tuple[int, int],
                 upsampling:Literal['transposed_conv', 'upsample+conv'] = "upsample+conv"):
        super(UpConvBlock, self).__init__()
        self.upsample = UpConv2d(in_channels, out_channels, kernel_size, stride, padding, output_padding=0, 
                                 method=upsampling)
        self.local_stage = LocalStage(out_channels, spatial_dim=spatial_dim)

    def forward(self, x:torch.Tensor):
        x = self.upsample(x)
        x = self.local_stage(x)
        return x