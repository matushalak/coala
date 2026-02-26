import torch
import torch.nn as nn


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


def sp_ln_forward(self, x: torch.Tensor, keep_mask: torch.BoolTensor):
    x = super(type(self), self).forward(x)
    return x * keep_mask.to(dtype=x.dtype)


class SparseConv2d(nn.Conv2d):
    forward = sp_conv_forward


class SparseMaxPooling(nn.MaxPool2d):
    forward = sp_conv_forward


class SparseAvgPooling(nn.AvgPool2d):
    forward = sp_conv_forward


class SparseLayerNorm2d(nn.LayerNorm):
    forward = sp_ln_forward


class SparseResidualConv2d(nn.Module):
    """Residual block that keeps sparse positions zeroed after each operation."""

    def __init__(self, n_channels: int, spatial_dim: tuple[int, int], kernel_size: int = 3):
        super().__init__()
        self.conv1 = SparseConv2d(n_channels, n_channels, kernel_size=kernel_size, padding="same", stride=1)
        self.norm1 = SparseLayerNorm2d((n_channels, *spatial_dim))
        self.act = nn.GELU()
        self.conv2 = SparseConv2d(n_channels, n_channels, kernel_size=kernel_size, padding="same", stride=1)

    def forward(self, x: torch.Tensor, keep_mask: torch.BoolTensor) -> torch.Tensor:
        mask = keep_mask.to(dtype=x.dtype)
        y = self.conv1(x, keep_mask)
        y = self.norm1(y, keep_mask)
        y = self.act(y)
        y = y * mask  # GELU can re-introduce non-zero values on inactive positions.
        y = self.conv2(y, keep_mask)
        return x + y


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
    ):
        super().__init__()
        self.norm = SparseLayerNorm2d((n_channels, *spatial_dim))
        self.act = nn.GELU()
        self.block = (
            SparseResidualConv2d(n_channels, spatial_dim=spatial_dim, kernel_size=kernel_size)
            if use_residual
            else None
        )

    def forward(self, x: torch.Tensor, keep_mask: torch.BoolTensor) -> torch.Tensor:
        x = self.norm(x, keep_mask)
        x = self.act(x)
        x = x * keep_mask.to(dtype=x.dtype)
        if self.block is not None:
            x = self.block(x, keep_mask)
        return x


class DenseResidualConv2d(nn.Module):
    """Standard residual block for decoder dense feature maps."""

    def __init__(self, n_channels: int, spatial_dim: tuple[int, int], kernel_size: int = 3):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(n_channels, n_channels, kernel_size=kernel_size, padding="same", stride=1),
            nn.LayerNorm((n_channels, *spatial_dim)),
            nn.GELU(),
            nn.Conv2d(n_channels, n_channels, kernel_size=kernel_size, padding="same", stride=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class SparseCNNEncoder(nn.Module):
    """
    MNIST-sized sparse encoder (28 -> 14 -> 7 -> 4).

    Stage convention:
    - resolution transition is explicit (downsample conv)
    - local processing is explicit (norm/act/residual at fixed resolution)
    - returned feats are post-local / pre-next-transition states
    """

    def __init__(self, num_input_channels: int = 1, num_filters: int = 32):
        super().__init__()
        self.down28_to_14 = SparseConv2d(num_input_channels, num_filters, kernel_size=3, padding=1, stride=2)
        self.local14 = SparseLocalStage(num_filters, spatial_dim=(14, 14), use_residual=True, kernel_size=3)

        self.down14_to_7 = SparseConv2d(num_filters, 2 * num_filters, kernel_size=3, padding=1, stride=2)
        self.local7 = SparseLocalStage(2 * num_filters, spatial_dim=(7, 7), use_residual=True, kernel_size=3)

        self.down7_to_4 = SparseConv2d(2 * num_filters, 2 * num_filters, kernel_size=3, padding=1, stride=2)
        self.local4 = SparseLocalStage(2 * num_filters, spatial_dim=(4, 4), use_residual=False, kernel_size=3)

    def forward(self, x: torch.Tensor, keep_mask: torch.BoolTensor) -> dict[str, torch.Tensor]:
        x = x.float()
        keep_mask = keep_mask.bool()
        x = x * keep_mask.to(dtype=x.dtype)

        mask14 = downsample_center_mask(keep_mask, (14, 14), stride=2)
        x14 = self.down28_to_14(x, mask14)
        x14 = self.local14(x14, mask14)

        mask7 = downsample_center_mask(mask14, (7, 7), stride=2)
        x7 = self.down14_to_7(x14, mask7)
        x7 = self.local7(x7, mask7)

        mask4 = downsample_center_mask(mask7, (4, 4), stride=2)
        x4 = self.down7_to_4(x7, mask4)
        x4 = self.local4(x4, mask4)
        breakpoint()
        return {
            "feat14": x14,
            "feat7": x7,
            "feat4": x4,
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
    ):
        super().__init__()
        c14 = num_filters
        c7 = 2 * num_filters
        c4 = 2 * num_filters
        self.set_densify_mode(densify_mode)

        self.mask_token14 = nn.Parameter(torch.zeros(1, c14, 1, 1))
        self.mask_token7 = nn.Parameter(torch.zeros(1, c7, 1, 1))
        self.mask_token4 = nn.Parameter(torch.zeros(1, c4, 1, 1))
        nn.init.normal_(self.mask_token14, mean=0.0, std=0.02)
        nn.init.normal_(self.mask_token7, mean=0.0, std=0.02)
        nn.init.normal_(self.mask_token4, mean=0.0, std=0.02)

        # Explicit stage split: local processing at fixed resolution vs resolution transitions.
        self.local4 = nn.Sequential(nn.LayerNorm((c4, 4, 4)), nn.GELU())
        self.up4_to_7 = nn.ConvTranspose2d(c4, c7, kernel_size=3, output_padding=0, padding=1, stride=2)
        self.local7 = nn.Sequential(
            nn.LayerNorm((c7, 7, 7)),
            nn.GELU(),
            DenseResidualConv2d(c7, spatial_dim=(7, 7), kernel_size=3),
        )

        self.up7_to_14 = nn.ConvTranspose2d(c7, c14, kernel_size=3, output_padding=1, padding=1, stride=2)
        self.local14 = nn.Sequential(
            nn.LayerNorm((c14, 14, 14)),
            nn.GELU(),
            DenseResidualConv2d(c14, spatial_dim=(14, 14), kernel_size=3),
        )

        self.up14_to_28 = nn.ConvTranspose2d(
            c14,
            num_output_channels,
            kernel_size=3,
            output_padding=1,
            padding=1,
            stride=2,
        )

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
            return torch.randn_like(feat)
        raise RuntimeError(f"Unsupported densify_mode: {self.densify_mode}")

    def _densify(self, feat: torch.Tensor, keep_mask: torch.BoolTensor, mask_token: torch.Tensor) -> torch.Tensor:
        fill = self._masked_fill(feat, mask_token)
        return torch.where(keep_mask.expand_as(feat), feat, fill)

    def forward(self, enc_out: dict[str, torch.Tensor]) -> torch.Tensor:
        dense4 = self._densify(enc_out["feat4"], enc_out["mask4"], self.mask_token4)
        dense7_skip = self._densify(enc_out["feat7"], enc_out["mask7"], self.mask_token7)
        dense14_skip = self._densify(enc_out["feat14"], enc_out["mask14"], self.mask_token14)

        x = self.local4(dense4)
        x = self.up4_to_7(x)
        x = x + dense7_skip
        x = self.local7(x)
        x = self.up7_to_14(x)
        x = x + dense14_skip
        x = self.local14(x)
        return self.up14_to_28(x)

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
    ):
        super().__init__()
        self.encoder = SparseCNNEncoder(num_input_channels=num_input_channels, num_filters=num_filters)
        self.decoder = SparseCNNDecoder(
            num_output_channels=num_output_channels,
            num_filters=num_filters,
            densify_mode=decoder_densify_mode,
        )

    def forward(self, x: torch.Tensor, keep_mask: torch.BoolTensor) -> torch.Tensor:
        enc_out = self.encoder(x, keep_mask=keep_mask)
        return self.decoder(enc_out)

    def set_decoder_densify_mode(self, mode: str) -> None:
        self.decoder.set_densify_mode(mode)
