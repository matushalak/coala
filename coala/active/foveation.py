from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from PIL import Image


def _ensure_batched(image: torch.Tensor) -> tuple[torch.Tensor, bool]:
    if image.ndim == 3:
        return image.unsqueeze(0), True
    if image.ndim == 4:
        return image, False
    raise ValueError(f"Expected image with shape (C,H,W) or (N,C,H,W), got {tuple(image.shape)}.")


def _restore_batch_dim(image: torch.Tensor, squeezed: bool) -> torch.Tensor:
    return image.squeeze(0) if squeezed else image


def _to_rgb(image: torch.Tensor) -> torch.Tensor:
    if image.ndim != 3:
        raise ValueError(f"Expected image with shape (C,H,W), got {tuple(image.shape)}.")
    if image.shape[0] == 3:
        return image
    if image.shape[0] == 1:
        return image.expand(3, -1, -1)
    raise ValueError(f"Expected 1 or 3 channels, got {image.shape[0]}.")


def _fixation_pixels(image: torch.Tensor, fixation_x: float, fixation_y: float) -> tuple[float, float]:
    _, _, height, width = image.shape
    x_coord = max(0.0, min(width - 1.0, fixation_x * (width - 1.0)))
    y_coord = max(0.0, min(height - 1.0, fixation_y * (height - 1.0)))
    return x_coord, y_coord


def _meshgrid_pixels(height: int, width: int, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    y_coords = torch.arange(height, device=device, dtype=dtype)
    x_coords = torch.arange(width, device=device, dtype=dtype)
    return torch.meshgrid(y_coords, x_coords, indexing="ij")


def gaussian_blur(image: torch.Tensor, sigma: float) -> torch.Tensor:
    image, squeezed = _ensure_batched(image)
    if sigma <= 0.0:
        return _restore_batch_dim(image, squeezed)

    radius = max(1, int(round(3.0 * sigma)))
    coords = torch.arange(-radius, radius + 1, device=image.device, dtype=image.dtype)
    kernel_1d = torch.exp(-(coords.square()) / (2.0 * sigma * sigma))
    kernel_1d = kernel_1d / kernel_1d.sum().clamp_min(1e-8)

    kernel_x = kernel_1d.view(1, 1, 1, -1).expand(image.shape[1], 1, 1, -1)
    kernel_y = kernel_1d.view(1, 1, -1, 1).expand(image.shape[1], 1, -1, 1)
    blurred = F.conv2d(F.pad(image, (radius, radius, 0, 0), mode="reflect"), kernel_x, groups=image.shape[1])
    blurred = F.conv2d(F.pad(blurred, (0, 0, radius, radius), mode="reflect"), kernel_y, groups=image.shape[1])
    return _restore_batch_dim(blurred, squeezed)


def _cosine_square_belt(x: torch.Tensor) -> torch.Tensor:
    lower = torch.cos(math.pi * (x + 0.25)).square()
    upper = 1.0 - torch.cos(math.pi * (x - 0.75)).square()
    zeros = torch.zeros_like(x)
    ones = torch.ones_like(x)
    return (
        torch.where((x <= -0.25) & (x > -0.75), lower, zeros)
        + torch.where((x >= 0.25) & (x <= 0.75), upper, zeros)
        + torch.where((x < 0.25) & (x > -0.25), ones, zeros)
    )


def _fovea_basis(eccentricity_deg: torch.Tensor, spacing: float, e0_deg: float) -> torch.Tensor:
    spacing_t = torch.tensor(spacing, device=eccentricity_deg.device, dtype=eccentricity_deg.dtype)
    e0_t = torch.tensor(e0_deg, device=eccentricity_deg.device, dtype=eccentricity_deg.dtype)
    safe_ecc = eccentricity_deg.clamp_min(1e-6)
    preinput = (torch.log(safe_ecc) - torch.log(e0_t)) / spacing_t
    preinput = torch.clamp(preinput, 0.0, 1.0)
    return _cosine_square_belt(preinput)


def _ring_basis(eccentricity_deg: torch.Tensor, ring_index: int, spacing: float, e0_deg: float) -> torch.Tensor:
    spacing_t = torch.tensor(spacing, device=eccentricity_deg.device, dtype=eccentricity_deg.dtype)
    e0_t = torch.tensor(e0_deg, device=eccentricity_deg.device, dtype=eccentricity_deg.dtype)
    safe_ecc = eccentricity_deg.clamp_min(1e-6)
    preinput = (torch.log(safe_ecc) - (torch.log(e0_t) + (ring_index + 1) * spacing_t)) / spacing_t
    return _cosine_square_belt(preinput)


def perceptual_foveate(
    image: torch.Tensor,
    fixation_x: float,
    fixation_y: float,
    *,
    kerw_coef: float = 0.06,
    fovea_radius_ratio: float = 0.12,
    ring_spacing: float = 0.2,
    parafoveal_noise_std: float = 0.0,
    field_of_view_degrees: float = 20.0,
) -> torch.Tensor:
    image, squeezed = _ensure_batched(image)
    if kerw_coef < 0.0:
        raise ValueError("kerw_coef must be >= 0.")
    if fovea_radius_ratio <= 0.0:
        raise ValueError("fovea_radius_ratio must be > 0.")
    if ring_spacing <= 0.0:
        raise ValueError("ring_spacing must be > 0.")
    if parafoveal_noise_std < 0.0:
        raise ValueError("parafoveal_noise_std must be >= 0.")

    batch_size, _, height, width = image.shape
    x_coord, y_coord = _fixation_pixels(image, fixation_x, fixation_y)
    grid_y, grid_x = _meshgrid_pixels(height, width, image.device, image.dtype)

    deg_per_pix = field_of_view_degrees / math.sqrt(height * height + width * width)
    eccentricity_px = torch.sqrt((grid_x - x_coord).square() + (grid_y - y_coord).square())
    eccentricity_deg = eccentricity_px * deg_per_pix

    fovea_radius_px = max(1.0, fovea_radius_ratio * min(height, width))
    e0_deg = max(1e-4, fovea_radius_px * deg_per_pix)
    maxecc_deg = max(
        math.sqrt(max(x_coord, width - x_coord) ** 2 + max(y_coord, height - y_coord) ** 2) * deg_per_pix,
        e0_deg,
    )
    num_rings = max(0, int(math.ceil((math.log(maxecc_deg) - math.log(e0_deg)) / ring_spacing)))

    fovea_mask = _fovea_basis(eccentricity_deg, ring_spacing, e0_deg)
    result = fovea_mask[None, None, :, :] * image
    for ring_index in range(num_rings):
        ring_mask = _ring_basis(eccentricity_deg, ring_index, ring_spacing, e0_deg)[None, None, :, :]
        mean_dev_deg = math.exp(math.log(e0_deg) + (ring_index + 1) * ring_spacing)
        sigma_px = kerw_coef * mean_dev_deg / deg_per_pix
        blurred = gaussian_blur(image, sigma=sigma_px)
        result = result + ring_mask * blurred

    if parafoveal_noise_std > 0.0:
        parafoveal_weight = (1.0 - fovea_mask).pow(1.5)[None, None, :, :]
        corr_sigma_px = max(1.0, 0.35 * fovea_radius_px)
        noise_texture = gaussian_blur(torch.randn_like(result), sigma=corr_sigma_px)
        noise_texture = noise_texture - noise_texture.mean(dim=(-2, -1), keepdim=True)
        noise_texture = noise_texture / noise_texture.std(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
        luminance_scale = 0.35 + 0.65 * result.detach()
        noise = parafoveal_noise_std * noise_texture * luminance_scale
        result = result + parafoveal_weight * noise

    result = result.clamp(0.0, 1.0)
    return _restore_batch_dim(result, squeezed)


def cortical_magnify(
    image: torch.Tensor,
    fixation_x: float,
    fixation_y: float,
    *,
    magnif_fov_ratio: float = 0.22,
    magnif_k_ratio: float = 0.22,
    cover_ratio: float = 1.0,
) -> torch.Tensor:
    image, squeezed = _ensure_batched(image)
    if magnif_fov_ratio <= 0.0:
        raise ValueError("magnif_fov_ratio must be > 0.")
    if magnif_k_ratio < 0.0:
        raise ValueError("magnif_k_ratio must be >= 0.")
    if cover_ratio <= 0.0:
        raise ValueError("cover_ratio must be > 0.")

    _, _, height, width = image.shape
    x_coord, y_coord = _fixation_pixels(image, fixation_x, fixation_y)
    half_h = height // 2
    half_w = width // 2

    grid_y, grid_x = torch.meshgrid(
        torch.arange(-half_h + 0.5, half_h + 0.5, device=image.device, dtype=image.dtype),
        torch.arange(-half_w + 0.5, half_w + 0.5, device=image.device, dtype=image.dtype),
        indexing="ij",
    )
    eccentricity = torch.sqrt(grid_x.square() + grid_y.square())
    eccentricity_safe = eccentricity.clamp_min(1e-6)

    magnif_fov_px = max(1.0, magnif_fov_ratio * min(height, width))
    magnif_k_px = max(0.0, magnif_k_ratio * min(height, width))
    ecc_tfm = torch.where(
        eccentricity < magnif_fov_px,
        eccentricity,
        ((eccentricity + magnif_k_px).square() / (2.0 * (magnif_fov_px + magnif_k_px)))
        + magnif_fov_px
        - (magnif_fov_px + magnif_k_px) / 2.0,
    )

    maxdist = math.sqrt(max(height - y_coord, y_coord) ** 2 + max(width - x_coord, x_coord) ** 2)
    coef = maxdist / float(ecc_tfm.max().item())
    coef = coef * math.sqrt(cover_ratio)

    sample_x = x_coord + coef * ecc_tfm * (grid_x / eccentricity_safe)
    sample_y = y_coord + coef * ecc_tfm * (grid_y / eccentricity_safe)
    sample_x = sample_x.clamp(0.0, width - 1.0)
    sample_y = sample_y.clamp(0.0, height - 1.0)

    norm_x = (sample_x / max(width - 1, 1)) * 2.0 - 1.0
    norm_y = (sample_y / max(height - 1, 1)) * 2.0 - 1.0
    grid = torch.stack((norm_x, norm_y), dim=-1).unsqueeze(0).expand(image.shape[0], -1, -1, -1)

    magnified = F.grid_sample(image, grid, mode="bilinear", padding_mode="border", align_corners=True)
    magnified = magnified.clamp(0.0, 1.0)
    return _restore_batch_dim(magnified, squeezed)


def apply_foveation(
    image: torch.Tensor,
    fixation_x: float,
    fixation_y: float,
    *,
    kerw_coef: float = 0.06,
    fovea_radius_ratio: float = 0.12,
    ring_spacing: float = 0.2,
    parafoveal_noise_std: float = 0.0,
    magnif_fov_ratio: float = 0.22,
    magnif_k_ratio: float = 0.22,
    cover_ratio: float = 1.0,
    field_of_view_degrees: float = 20.0,
) -> torch.Tensor:
    perceptual = perceptual_foveate(
        image,
        fixation_x,
        fixation_y,
        kerw_coef=kerw_coef,
        fovea_radius_ratio=fovea_radius_ratio,
        ring_spacing=ring_spacing,
        parafoveal_noise_std=parafoveal_noise_std,
        field_of_view_degrees=field_of_view_degrees,
    )
    combined = cortical_magnify(
        perceptual,
        fixation_x,
        fixation_y,
        magnif_fov_ratio=magnif_fov_ratio,
        magnif_k_ratio=magnif_k_ratio,
        cover_ratio=cover_ratio,
    )
    return combined


def render_foveation_bundle(
    image: torch.Tensor,
    fixation_x: float,
    fixation_y: float,
    *,
    kerw_coef: float = 0.06,
    fovea_radius_ratio: float = 0.12,
    ring_spacing: float = 0.2,
    parafoveal_noise_std: float = 0.0,
    magnif_fov_ratio: float = 0.22,
    magnif_k_ratio: float = 0.22,
    cover_ratio: float = 1.0,
    field_of_view_degrees: float = 20.0,
) -> dict[str, torch.Tensor]:
    original = image
    overlay = overlay_fixation_marker(image, fixation_x, fixation_y)
    perceptual = perceptual_foveate(
        image,
        fixation_x,
        fixation_y,
        kerw_coef=kerw_coef,
        fovea_radius_ratio=fovea_radius_ratio,
        ring_spacing=ring_spacing,
        parafoveal_noise_std=parafoveal_noise_std,
        field_of_view_degrees=field_of_view_degrees,
    )
    combined = cortical_magnify(
        perceptual,
        fixation_x,
        fixation_y,
        magnif_fov_ratio=magnif_fov_ratio,
        magnif_k_ratio=magnif_k_ratio,
        cover_ratio=cover_ratio,
    )
    return {
        "original": original,
        "overlay": overlay,
        "perceptual": _to_rgb(perceptual),
        "combined": _to_rgb(combined),
    }


def overlay_fixation_marker(
    image: torch.Tensor,
    fixation_x: float,
    fixation_y: float,
    *,
    marker_radius: int = 3,
    marker_color: tuple[float, float, float] = (1.0, 0.2, 0.1),
) -> torch.Tensor:
    rgb = _to_rgb(image).clone()
    _, height, width = rgb.shape
    x_coord = max(0, min(width - 1, int(round(fixation_x * (width - 1)))))
    y_coord = max(0, min(height - 1, int(round(fixation_y * (height - 1)))))
    radius = max(1, int(marker_radius))

    yy, xx = torch.meshgrid(
        torch.arange(height, device=rgb.device),
        torch.arange(width, device=rgb.device),
        indexing="ij",
    )
    dist = torch.sqrt((xx - x_coord).float().square() + (yy - y_coord).float().square())
    ring = (dist >= max(0.0, radius - 1.25)) & (dist <= radius + 0.25)
    cross = ((xx == x_coord) & (torch.abs(yy - y_coord) <= radius + 1)) | (
        (yy == y_coord) & (torch.abs(xx - x_coord) <= radius + 1)
    )
    marker = ring | cross
    color = torch.tensor(marker_color, device=rgb.device, dtype=rgb.dtype).view(3, 1, 1)
    return torch.where(marker.unsqueeze(0), color, rgb)


def tensor_to_pil_image(image: torch.Tensor) -> Image.Image:
    rgb = _to_rgb(image.detach().cpu().clamp(0.0, 1.0))
    rgb = rgb.mul(255.0).round().to(dtype=torch.uint8).permute(1, 2, 0).numpy()
    return Image.fromarray(rgb, mode="RGB")
