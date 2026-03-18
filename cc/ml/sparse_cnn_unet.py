import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Literal

def downsample_center_mask(
    keep_mask: torch.BoolTensor,
    out_size: tuple[int, int],
    stride: int = 2,
) -> torch.BoolTensor:
    """
    Downsample activity by selecting center-aligned positions (slice-based).
    For k=3, p=1, s=2 convs, output (i, j) aligns to input (2i, 2j).
    """
    out_h, out_w = out_size
    return keep_mask[:, :, ::stride, ::stride][:, :, :out_h, :out_w]


def sp_conv_forward(self, x: torch.Tensor, keep_mask: torch.BoolTensor):
    x = super(type(self), self).forward(x)
    return x * keep_mask.to(dtype=x.dtype)


def _expand_keep_mask(x: torch.Tensor, keep_mask: torch.BoolTensor) -> torch.Tensor:
    if keep_mask.ndim != x.ndim:
        raise ValueError(
            f"keep_mask must have the same rank as x, got {keep_mask.ndim} and {x.ndim}."
        )
    if keep_mask.shape[0] != x.shape[0] or keep_mask.shape[-2:] != x.shape[-2:]:
        raise ValueError(
            f"keep_mask shape {tuple(keep_mask.shape)} is incompatible with x shape {tuple(x.shape)}."
        )
    if keep_mask.shape[1] not in (1, x.shape[1]):
        raise ValueError(
            f"keep_mask channel dimension must be 1 or match x, got {keep_mask.shape[1]} and {x.shape[1]}."
        )
    return keep_mask.to(dtype=x.dtype).expand_as(x)


def _masked_mean_and_var(
    x: torch.Tensor,
    keep_mask: torch.BoolTensor,
    reduce_dims: tuple[int, ...],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mask = _expand_keep_mask(x, keep_mask)
    masked_x = x * mask
    count = mask.sum(dim=reduce_dims, keepdim=True).clamp_min(1.0)
    mean = masked_x.sum(dim=reduce_dims, keepdim=True) / count
    var = ((x - mean).pow(2) * mask).sum(dim=reduce_dims, keepdim=True) / count
    return mask, mean, var


def _resolve_eps(eps: float | None, x: torch.Tensor) -> float:
    return torch.finfo(x.dtype).eps if eps is None else eps


class GlobalResponseNorm(nn.Module):
    """
    Global response normalization (GRN) as used in ConvNeXtV2.
        GRN normalizes each channel by the global L2 norm across spatial dimensions, 
        with learnable scaling and bias.
    
    Assumes (B, C, H, W) input shape.
    """

    def __init__(self, n_channels: int, eps: float = 1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, n_channels, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, n_channels, 1, 1))
        self.eps = eps

    def forward(self, x: torch.Tensor, keep_mask: torch.BoolTensor | None = None) -> torch.Tensor:
        if keep_mask is None:
            mask = 1.0
        else:
            mask = keep_mask.to(dtype=x.dtype)

        # L2 norm "pooling" across spatial dimensions.
        gx = torch.sqrt((x.pow(2) * mask).sum(dim=[2, 3], keepdim=True))
        # Competition across channels.
        nx = gx / (gx.mean(dim=1, keepdim=True) + self.eps)
        # Apply scaling and bias
        return self.gamma * (x * nx) + self.beta + x
    

class SparseGlobalResponseNorm(GlobalResponseNorm):
    def forward(self, x: torch.Tensor, keep_mask: torch.BoolTensor) -> torch.Tensor:
        return super().forward(x, keep_mask=keep_mask) * keep_mask.to(dtype=x.dtype)
        

class SparseConv2d(nn.Conv2d):
    forward = sp_conv_forward


class SparseMaxPooling(nn.MaxPool2d):
    forward = sp_conv_forward


class SparseAvgPooling(nn.AvgPool2d):
    forward = sp_conv_forward


# legacy layernorm with incorrect masking
# class SparseLayerNorm2d(nn.LayerNorm):
#     '''
#     Legacy incorrect SparseLayerNorm2d to check older models against
#     '''
#     forward = sp_conv_forward

# correct
class SparseLayerNorm2d(nn.LayerNorm):
    def forward(self, x: torch.Tensor, keep_mask: torch.BoolTensor) -> torch.Tensor:
        reduce_dims = tuple(range(x.ndim - len(self.normalized_shape), x.ndim))
        mask, mean, var = _masked_mean_and_var(x, keep_mask, reduce_dims)
        eps = _resolve_eps(self.eps, x)
        y = (x - mean) / torch.sqrt(var + eps)
        if self.elementwise_affine:
            y = y * self.weight + self.bias
        return y * mask


class SparseRMSNorm2d(nn.RMSNorm):
    def forward(self, x: torch.Tensor, keep_mask: torch.BoolTensor) -> torch.Tensor:
        reduce_dims = tuple(range(x.ndim - len(self.normalized_shape), x.ndim))
        mask = _expand_keep_mask(x, keep_mask)
        count = mask.sum(dim=reduce_dims, keepdim=True).clamp_min(1.0)
        eps = _resolve_eps(self.eps, x)
        rms = torch.rsqrt((x.pow(2) * mask).sum(dim=reduce_dims, keepdim=True) / count + eps)
        y = x * rms
        if self.elementwise_affine:
            y = y * self.weight
        return y * mask


class SparseNorm2d(nn.Module):
    def __init__(self, norm_type: str = "rmsnorm", *args, **kwargs):
        super().__init__()
        norm_type = norm_type.lower()
        if norm_type == "layernorm":
            self.norm = SparseLayerNorm2d(*args, **kwargs)
        elif norm_type == "rmsnorm":
            self.norm = SparseRMSNorm2d(*args, **kwargs)
        else:
            raise ValueError(f"Unsupported norm_type {norm_type!r}. Expected 'layernorm' or 'rmsnorm'.")

    def forward(self, x: torch.Tensor, keep_mask: torch.BoolTensor) -> torch.Tensor:
        return self.norm(x, keep_mask)


class DenseNorm2d(nn.Module):
    def __init__(self, norm_type: str = "rmsnorm", *args, **kwargs):
        super().__init__()
        norm_type = norm_type.lower()
        if norm_type == "layernorm":
            self.norm = nn.LayerNorm(*args, **kwargs)
        elif norm_type == "rmsnorm":
            self.norm = nn.RMSNorm(*args, **kwargs)
        else:
            raise ValueError(f"Unsupported norm_type {norm_type!r}. Expected 'layernorm' or 'rmsnorm'.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x)


class SparseResidualConv2d(nn.Module):
    """Residual block that keeps sparse positions zeroed after each operation."""

    def __init__(self, n_channels: int, spatial_dim: tuple[int, int], kernel_size: int = 3,
                 norm_type: str = "rmsnorm"):
        super().__init__()
        self.conv1 = SparseConv2d(n_channels, n_channels, kernel_size=kernel_size, padding="same", stride=1)
        self.norm1 = SparseNorm2d(norm_type=norm_type, normalized_shape=(n_channels, *spatial_dim))
        self.act = nn.GELU()
        self.grn = SparseGlobalResponseNorm(n_channels) # global response norm from convnextv2 paper
        self.conv2 = SparseConv2d(n_channels, n_channels, kernel_size=kernel_size, padding="same", stride=1)

    def forward(self, x: torch.Tensor, keep_mask: torch.BoolTensor) -> torch.Tensor:
        mask = keep_mask.to(dtype=x.dtype)
        y = self.conv1(x, keep_mask)
        y = self.norm1(y, keep_mask)
        y = self.act(y)
        y = self.grn(y, keep_mask) # calibrate channels with GRN
        y = y * mask  # GELU can re-introduce non-zero values on inactive positions.
        y = self.conv2(y, keep_mask)
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
        use_residual: bool = True,
        kernel_size: int = 3,
        norm_type: str = "rmsnorm",
    ):
        super().__init__()
        self.pre_norm = SparseNorm2d(norm_type=norm_type, normalized_shape=(n_channels, *spatial_dim))
        self.act = nn.GELU()
        self.block = (
            SparseResidualConv2d(n_channels, spatial_dim=spatial_dim, kernel_size=kernel_size, norm_type=norm_type)
            if use_residual
            else None
        )
        self.post_norm = SparseNorm2d(norm_type=norm_type, normalized_shape=(n_channels, *spatial_dim))

    def forward(self, x: torch.Tensor, keep_mask: torch.BoolTensor) -> torch.Tensor:
        x = self.pre_norm(x, keep_mask)
        x = self.act(x)
        x = x * keep_mask.to(dtype=x.dtype)
        if self.block is not None:
            x = self.block(x, keep_mask)
        x = self.post_norm(x, keep_mask)
        return x


class DenseUpConv2d(nn.Module):
    """
    Upsampling for decoder dense feature maps.
    Can be implemented as:
        - transposed convolution (learned projection weights from low-res to high-res)
        - upsameple (nearest / bilinear) + regular convolution (learned weights for FB inputs)
    """

    def __init__(self, in_channels: int, out_channels: int, 
                 kernel_size: int = 3, stride: int = 2, padding: int = 1, output_padding: int = 1,
                 method: Literal['transposed_conv', 'upsample+conv'] = "transposed_conv"):
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
                in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, output_padding=output_padding
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


class DenseResidualConv2d(nn.Module):
    """Standard residual block for decoder dense feature maps."""

    def __init__(self, n_channels: int, spatial_dim: tuple[int, int], kernel_size: int = 3, norm_type: str = "rmsnorm"):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(n_channels, n_channels, kernel_size=kernel_size, padding="same", stride=1),
            DenseNorm2d(norm_type=norm_type, normalized_shape=(n_channels, *spatial_dim)),
            nn.GELU(),
            GlobalResponseNorm(n_channels), # global response norm from convnextv2 paper
            nn.Conv2d(n_channels, n_channels, kernel_size=kernel_size, padding="same", stride=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class DenseLocalStage(nn.Module):
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
        use_residual: bool = True,
        kernel_size: int = 3,
        norm_type: str = "rmsnorm",
    ):
        super().__init__()
        self.pre_norm = DenseNorm2d(norm_type=norm_type, normalized_shape=(n_channels, *spatial_dim))
        self.act = nn.GELU()
        self.block = (
            DenseResidualConv2d(n_channels, spatial_dim=spatial_dim, kernel_size=kernel_size, norm_type=norm_type)
            if use_residual
            else None
        )
        self.post_norm = DenseNorm2d(norm_type=norm_type, normalized_shape=(n_channels, *spatial_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pre_norm(x)
        x = self.act(x)
        if self.block is not None:
            x = self.block(x)
        x = self.post_norm(x)
        return x


class SparseCNNEncoder(nn.Module):
    """
    MNIST-sized sparse encoder (28 -> 14 -> 7 -> 4).

    Stage convention:
    - resolution transition is explicit (downsample conv)
    - local processing is explicit (norm/act/residual at fixed resolution)
    - returned feats are post-local / pre-next-transition states
    """

    def __init__(self, num_input_channels: int = 1, num_filters: int = 32, norm_type: str = "rmsnorm"):
        super().__init__()
        # V1
        self.down28_to_28 = SparseConv2d(num_input_channels, num_filters // 2, kernel_size=3, padding=1, stride=1)
        self.local28 = SparseLocalStage(num_filters // 2, spatial_dim=(28, 28), use_residual=True, kernel_size=3, norm_type=norm_type)

        # V2
        self.down28_to_14 = SparseConv2d(num_filters // 2, num_filters, kernel_size=3, padding=1, stride=2)
        self.local14 = SparseLocalStage(num_filters, spatial_dim=(14, 14), use_residual=True, kernel_size=3, norm_type=norm_type)

        # V3
        self.down14_to_7 = SparseConv2d(num_filters, 2 * num_filters, kernel_size=3, padding=1, stride=2)
        self.local7 = SparseLocalStage(2 * num_filters, spatial_dim=(7, 7), use_residual=True, kernel_size=3, norm_type=norm_type)

        # V4
        self.down7_to_4 = SparseConv2d(2 * num_filters, 4 * num_filters, kernel_size=3, padding=1, stride=2)
        self.local4 = SparseLocalStage(4 * num_filters, spatial_dim=(4, 4), use_residual=False, kernel_size=3, norm_type=norm_type)

    def forward(self, x: torch.Tensor, keep_mask: torch.BoolTensor) -> dict[str, torch.Tensor]:
        x = x.float()
        keep_mask = keep_mask.bool()
        x = x * keep_mask.to(dtype=x.dtype)

        x = self.down28_to_28(x, keep_mask)
        x28 = self.local28(x, keep_mask)

        mask14 = downsample_center_mask(keep_mask, (14, 14), stride=2)
        x14 = self.down28_to_14(x28, mask14)
        x14 = self.local14(x14, mask14)

        mask7 = downsample_center_mask(mask14, (7, 7), stride=2)
        x7 = self.down14_to_7(x14, mask7)
        x7 = self.local7(x7, mask7)

        mask4 = downsample_center_mask(mask7, (4, 4), stride=2)
        x4 = self.down7_to_4(x7, mask4)
        x4 = self.local4(x4, mask4)
        
        return {
            "feat28": x28,
            "feat14": x14,
            "feat7": x7,
            "feat4": x4,
            "mask28":keep_mask,
            "mask14": mask14,
            "mask7": mask7,
            "mask4": mask4,
        }


class SparseCNNDecoder(nn.Module):
    """
    Dense decoder with UNet skips.
    Sparse positions from the encoder are densified before local processing.
    """

    DENSIFY_MODES = ("random", "token", "zero")

    def __init__(
        self,
        num_output_channels: int = 1,
        num_filters: int = 32,
        densify_mode: str = "random",
        conv_method: Literal['transposed_conv', 'upsample+conv'] = "transposed_conv",
        use_skip:bool = True,
        norm_type: str = "rmsnorm"
    ):
        super().__init__()
        self.conv_method = conv_method
        self.use_skip = use_skip

        c28 = num_filters // 2
        c14 = num_filters
        c7 = 2 * num_filters
        c4 = 4 * num_filters
        self.set_densify_mode(densify_mode)

        self.mask_token28 = nn.Parameter(torch.zeros(1, c28, 1, 1))
        self.mask_token14 = nn.Parameter(torch.zeros(1, c14, 1, 1))
        self.mask_token7 = nn.Parameter(torch.zeros(1, c7, 1, 1))
        self.mask_token4 = nn.Parameter(torch.zeros(1, c4, 1, 1))
        nn.init.normal_(self.mask_token28, mean=0.0, std=0.02)
        nn.init.normal_(self.mask_token14, mean=0.0, std=0.02)
        nn.init.normal_(self.mask_token7, mean=0.0, std=0.02)
        nn.init.normal_(self.mask_token4, mean=0.0, std=0.02)

        # V4
        self.local4 = DenseLocalStage(c4, spatial_dim=(4, 4), use_residual=False, kernel_size=3, norm_type=norm_type)
        
        # V3
        # self.up4_to_7 = nn.ConvTranspose2d(c4, c7, kernel_size=3, output_padding=0, padding=1, stride=2)
        self.up4_to_7 = DenseUpConv2d(c4, c7, kernel_size=3, padding=1, stride=2, output_padding=0, method=conv_method)
        self.local7 = DenseLocalStage(c7, spatial_dim=(7, 7), use_residual=True, kernel_size=3, norm_type=norm_type)

        # V2
        # self.up7_to_14 = nn.ConvTranspose2d(c7, c14, kernel_size=3, output_padding=1, padding=1, stride=2)
        self.up7_to_14 = DenseUpConv2d(c7, c14, kernel_size=3, padding=1, stride=2, output_padding=1, method=conv_method)
        self.local14 = DenseLocalStage(c14, spatial_dim=(14, 14), use_residual=True, kernel_size=3, norm_type=norm_type)

        # V1
        # self.up14_to_28 = nn.ConvTranspose2d(c14,c28,kernel_size=3,output_padding=1,padding=1,stride=2,)
        self.up14_to_28 = DenseUpConv2d(c14, c28, kernel_size=3, padding=1, stride=2, output_padding=1, method=conv_method)
        self.local28 = DenseLocalStage(c28, spatial_dim=(28, 28), use_residual=True, kernel_size=3, norm_type=norm_type)

        # Predict output (retina)        
        self.up28_to_out = nn.Conv2d(c28, num_output_channels, kernel_size=3, padding=1, stride=1)
        nn.init.xavier_uniform_(self.up28_to_out.weight)

    def set_densify_mode(self, mode: str) -> None:
        if mode not in self.DENSIFY_MODES:
            raise ValueError(f"densify_mode must be one of {self.DENSIFY_MODES}, got {mode!r}.")
        self.densify_mode = mode

    def _masked_fill(self, feat: torch.Tensor, mask_token: torch.Tensor) -> torch.Tensor:
        if self.densify_mode == "token":
            return mask_token.expand_as(feat)
        if self.densify_mode == "zero":
            return torch.zeros_like(feat)
        if self.densify_mode == "random":
            # Random noise on pixels of each masked patch
            # NOTE: are these values too large?
            return torch.randn_like(feat)# * 0.1
        raise RuntimeError(f"Unsupported densify_mode: {self.densify_mode}")

    def _densify(self, feat: torch.Tensor, keep_mask: torch.BoolTensor, mask_token: torch.Tensor) -> torch.Tensor:
        fill = self._masked_fill(feat, mask_token)
        return torch.where(keep_mask.expand_as(feat), feat, fill)

    def forward(self, enc_out: dict[str, torch.Tensor]) -> torch.Tensor:
        dense4 = self._densify(enc_out["feat4"], enc_out["mask4"], self.mask_token4)
        x = self.local4(dense4)
        x = self.up4_to_7(x)
        
        if self.use_skip:
            dense7_skip = self._densify(enc_out["feat7"], enc_out["mask7"], self.mask_token7)
            x = x + dense7_skip
        x = self.local7(x)
        x = self.up7_to_14(x)
        
        if self.use_skip:
            dense14_skip = self._densify(enc_out["feat14"], enc_out["mask14"], self.mask_token14)
            x = x + dense14_skip
        x = self.local14(x)
        x = self.up14_to_28(x)
        
        if self.use_skip:
            dense28_skip = self._densify(enc_out["feat28"], enc_out["mask28"], self.mask_token28)
            x = x + dense28_skip
        x = self.local28(x)
        x = self.up28_to_out(x)
        return x

    @property
    def device(self):
        return next(self.parameters()).device


class SparseCNNUNet(nn.Module):
    """Convenience wrapper matching a single forward(x, keep_mask) call."""

    def __init__(
        self,
        num_input_channels: int = 1,
        num_output_channels: int = 1,
        num_filters: int = 32,
        decoder_densify_mode: str = "random",
        use_skip:bool = True,
        upconv_method: Literal['transposed_conv', 'upsample+conv'] = "upsample+conv",
        norm_type: str = "rmsnorm"
    ):
        super().__init__()
        self.encoder = SparseCNNEncoder(num_input_channels=num_input_channels, 
                                        num_filters=num_filters, norm_type=norm_type)
        self.decoder = SparseCNNDecoder(
            num_output_channels=num_output_channels,
            num_filters=num_filters,
            densify_mode=decoder_densify_mode,
            use_skip=use_skip,
            conv_method=upconv_method,
            norm_type=norm_type
            )

    def forward(self, x: torch.Tensor, keep_mask: torch.BoolTensor) -> torch.Tensor:
        enc_out = self.encoder(x, keep_mask=keep_mask)
        return self.decoder(enc_out)

    def set_decoder_densify_mode(self, mode: str) -> None:
        self.decoder.set_densify_mode(mode)
