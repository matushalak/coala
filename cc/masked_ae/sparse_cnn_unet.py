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
    """MNIST-sized sparse encoder (28 -> 14 -> 7 -> 4)."""

    def __init__(self, num_input_channels: int = 1, num_filters: int = 32):
        super().__init__()
        self.down14 = SparseConv2d(num_input_channels, num_filters, kernel_size=3, padding=1, stride=2)
        self.norm14 = SparseLayerNorm2d((num_filters, 14, 14))
        self.block14 = SparseResidualConv2d(num_filters, spatial_dim=(14, 14), kernel_size=3)

        self.down7 = SparseConv2d(num_filters, 2 * num_filters, kernel_size=3, padding=1, stride=2)
        self.norm7 = SparseLayerNorm2d((2 * num_filters, 7, 7))
        self.block7 = SparseResidualConv2d(2 * num_filters, spatial_dim=(7, 7), kernel_size=3)

        self.down4 = SparseConv2d(2 * num_filters, 2 * num_filters, kernel_size=3, padding=1, stride=2)
        self.norm4 = SparseLayerNorm2d((2 * num_filters, 4, 4))
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor, keep_mask: torch.BoolTensor) -> dict[str, torch.Tensor]:
        x = x.float()
        keep_mask = keep_mask.bool()
        x = x * keep_mask.to(dtype=x.dtype)

        mask14 = downsample_center_mask(keep_mask, (14, 14), stride=2)
        x14 = self.down14(x, mask14)
        x14 = self.norm14(x14, mask14)
        x14 = self.act(x14)
        x14 = x14 * mask14.to(dtype=x14.dtype)
        x14 = self.block14(x14, mask14)

        mask7 = downsample_center_mask(mask14, (7, 7), stride=2)
        x7 = self.down7(x14, mask7)
        x7 = self.norm7(x7, mask7)
        x7 = self.act(x7)
        x7 = x7 * mask7.to(dtype=x7.dtype)
        x7 = self.block7(x7, mask7)

        mask4 = downsample_center_mask(mask7, (4, 4), stride=2)
        x4 = self.down4(x7, mask4)
        x4 = self.norm4(x4, mask4)
        x4 = self.act(x4)
        x4 = x4 * mask4.to(dtype=x4.dtype)

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
    Sparse positions from the encoder are filled with learned mask tokens first.
    """

    def __init__(self, num_output_channels: int = 1, num_filters: int = 32):
        super().__init__()
        c14 = num_filters
        c7 = 2 * num_filters
        c4 = 2 * num_filters

        self.mask_token14 = nn.Parameter(torch.zeros(1, c14, 1, 1))
        self.mask_token7 = nn.Parameter(torch.zeros(1, c7, 1, 1))
        self.mask_token4 = nn.Parameter(torch.zeros(1, c4, 1, 1))
        nn.init.normal_(self.mask_token14, mean=0.0, std=0.02)
        nn.init.normal_(self.mask_token7, mean=0.0, std=0.02)
        nn.init.normal_(self.mask_token4, mean=0.0, std=0.02)

        self.pre4 = nn.Sequential(nn.LayerNorm((c4, 4, 4)), nn.GELU())
        self.up7 = nn.ConvTranspose2d(c4, c7, kernel_size=3, output_padding=0, padding=1, stride=2)
        self.post7 = nn.Sequential(
            nn.LayerNorm((c7, 7, 7)),
            nn.GELU(),
            DenseResidualConv2d(c7, spatial_dim=(7, 7), kernel_size=3),
        )

        self.up14 = nn.ConvTranspose2d(c7, c14, kernel_size=3, output_padding=1, padding=1, stride=2)
        self.post14 = nn.Sequential(
            nn.LayerNorm((c14, 14, 14)),
            nn.GELU(),
            DenseResidualConv2d(c14, spatial_dim=(14, 14), kernel_size=3),
        )

        self.up28 = nn.ConvTranspose2d(c14, num_output_channels, kernel_size=3, output_padding=1, padding=1, stride=2)

    @staticmethod
    def _densify(feat: torch.Tensor, keep_mask: torch.BoolTensor, mask_token: torch.Tensor) -> torch.Tensor:
        dense_token = mask_token.expand_as(feat)
        return torch.where(keep_mask.expand_as(feat), feat, dense_token)

    def forward(self, enc_out: dict[str, torch.Tensor]) -> torch.Tensor:
        # dense4 = self._densify(enc_out["feat4"], enc_out["mask4"], self.mask_token4)
        # dense7_skip = self._densify(enc_out["feat7"], enc_out["mask7"], self.mask_token7)
        # dense14_skip = self._densify(enc_out["feat14"], enc_out["mask14"], self.mask_token14)

        dense4 = self._densify(enc_out["feat4"], enc_out["mask4"], torch.randn_like(self.mask_token4))
        dense7_skip = self._densify(enc_out["feat7"], enc_out["mask7"], torch.randn_like(self.mask_token7))
        dense14_skip = self._densify(enc_out["feat14"], enc_out["mask14"], torch.randn_like(self.mask_token14))

        x = self.pre4(dense4)
        x = self.up7(x)
        x = x + dense7_skip
        x = self.post7(x)
        x = self.up14(x)
        x = x + dense14_skip
        x = self.post14(x)
        return self.up28(x)

    @property
    def device(self):
        return next(self.parameters()).device


class SparseCNNUNet(nn.Module):
    """Convenience wrapper matching a single forward(x, keep_mask) call."""

    def __init__(self, num_input_channels: int = 1, num_output_channels: int = 1, num_filters: int = 32):
        super().__init__()
        self.encoder = SparseCNNEncoder(num_input_channels=num_input_channels, num_filters=num_filters)
        self.decoder = SparseCNNDecoder(num_output_channels=num_output_channels, num_filters=num_filters)

    def forward(self, x: torch.Tensor, keep_mask: torch.BoolTensor) -> torch.Tensor:
        enc_out = self.encoder(x, keep_mask=keep_mask)
        return self.decoder(enc_out)
