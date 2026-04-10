from __future__ import annotations

import torch


MASKING_STRATEGIES = ("random", "multi-block", "mixed")
DEFAULT_MULTI_BLOCK_SCALE_RANGE = (0.1, 0.6)
DEFAULT_MULTI_BLOCK_ASPECT_RATIO_RANGE = (0.5, 1.5)
DEFAULT_MULTI_BLOCK_SQUARE_ASPECT_RATIO = 1
MULTI_BLOCK_MIN_GAP = 1
MULTI_BLOCK_POSITION_CANDIDATES = 32
RANDOM_BANK_SIZE = 4096
MULTI_BLOCK_BANK_SIZE = 16384
MULTI_BLOCK_PIXEL_BANK_MAX_ELEMENTS = 16_000_000
MIXED_BATCH_BANK_SIZE = 512
MIXED_BATCH_PIXEL_BANK_MAX_ELEMENTS = 64_000_000

_MULTI_BLOCK_PATCH_BANK_CACHE: dict[tuple, torch.BoolTensor] = {}
_MULTI_BLOCK_DEVICE_BANK_CACHE: dict[tuple, torch.BoolTensor] = {}
_RANDOM_PATCH_BANK_CACHE: dict[tuple, torch.BoolTensor] = {}
_RANDOM_DEVICE_BANK_CACHE: dict[tuple, torch.BoolTensor] = {}
_MIXED_PATCH_BATCH_BANK_CACHE: dict[tuple, torch.BoolTensor] = {}
_MIXED_DEVICE_BATCH_BANK_CACHE: dict[tuple, torch.BoolTensor] = {}


def add_masking_arguments(
    parser,
    *,
    default_mask_ratio: float = 0.6,
    default_patch_size: int = 4,
    default_masking_strategy: str = "random",
    default_denoise_sigma: float = 1.0,
):
    parser.add_argument("--mask_ratio", default=default_mask_ratio, type=float, help="Fraction of patches to hide.")
    parser.add_argument("--patch_size", default=default_patch_size, type=int, help="Patch size used for masking.")
    parser.add_argument(
        "--masking_strategy",
        default=default_masking_strategy,
        choices=MASKING_STRATEGIES,
        type=str,
        help=(
            "Mask sampler to use. 'random' matches the original patchwise masking, "
            "'multi-block' masks one square block plus two rectangular blocks, and "
            "'mixed' uses an even mix of both within each batch."
        ),
    )
    parser.add_argument(
        "--denoise_sigma",
        default=default_denoise_sigma,
        type=float,
        help="Standard deviation of the zero-mean Gaussian noise used by denoising objectives.",
    )
    parser.add_argument(
        "--multi_block_scale_min",
        default=DEFAULT_MULTI_BLOCK_SCALE_RANGE[0],
        type=float,
        help=(
            "Minimum raw area scale for each multi-block mask, expressed as a fraction of the "
            "full image before the sampled block scales are renormalized to match mask_ratio."
        ),
    )
    parser.add_argument(
        "--multi_block_scale_max",
        default=DEFAULT_MULTI_BLOCK_SCALE_RANGE[1],
        type=float,
        help=(
            "Maximum raw area scale for each multi-block mask, expressed as a fraction of the "
            "full image before the sampled block scales are renormalized to match mask_ratio."
        ),
    )
    parser.add_argument(
        "--multi_block_aspect_ratio_min",
        default=DEFAULT_MULTI_BLOCK_ASPECT_RATIO_RANGE[0],
        type=float,
        help="Minimum aspect ratio used for the rectangular multi-block masks. Matches I-JEPA by default.",
    )
    parser.add_argument(
        "--multi_block_aspect_ratio_max",
        default=DEFAULT_MULTI_BLOCK_ASPECT_RATIO_RANGE[1],
        type=float,
        help="Maximum aspect ratio used for the rectangular multi-block masks. Matches I-JEPA by default.",
    )
    parser.add_argument(
        "--multi_block_square_aspect_ratio",
        default=DEFAULT_MULTI_BLOCK_SQUARE_ASPECT_RATIO,
        type=float,
        help="Aspect ratio used for the square-style multi-block mask.",
    )


def masking_kwargs_from_args(args) -> dict[str, float | str]:
    return {
        "masking_strategy": args.masking_strategy,
        "denoise_sigma": args.denoise_sigma,
        "multi_block_scale_min": args.multi_block_scale_min,
        "multi_block_scale_max": args.multi_block_scale_max,
        "multi_block_aspect_ratio_min": args.multi_block_aspect_ratio_min,
        "multi_block_aspect_ratio_max": args.multi_block_aspect_ratio_max,
        "multi_block_square_aspect_ratio": args.multi_block_square_aspect_ratio,
    }


def clear_mask_bank_caches() -> None:
    _MULTI_BLOCK_PATCH_BANK_CACHE.clear()
    _MULTI_BLOCK_DEVICE_BANK_CACHE.clear()
    _RANDOM_PATCH_BANK_CACHE.clear()
    _RANDOM_DEVICE_BANK_CACHE.clear()
    _MIXED_PATCH_BATCH_BANK_CACHE.clear()
    _MIXED_DEVICE_BATCH_BANK_CACHE.clear()


def sample_keep_mask(
    imgs: torch.Tensor,
    *,
    patch_size: int,
    mask_ratio: float,
    masking_strategy: str = "random",
    multi_block_scale_min: float = DEFAULT_MULTI_BLOCK_SCALE_RANGE[0],
    multi_block_scale_max: float = DEFAULT_MULTI_BLOCK_SCALE_RANGE[1],
    multi_block_aspect_ratio_min: float = DEFAULT_MULTI_BLOCK_ASPECT_RATIO_RANGE[0],
    multi_block_aspect_ratio_max: float = DEFAULT_MULTI_BLOCK_ASPECT_RATIO_RANGE[1],
    multi_block_square_aspect_ratio: float = DEFAULT_MULTI_BLOCK_SQUARE_ASPECT_RATIO,
) -> torch.BoolTensor:
    if masking_strategy == "random":
        return sample_random_keep_mask(imgs, patch_size=patch_size, mask_ratio=mask_ratio)
    if masking_strategy == "multi-block":
        return sample_multiblock_keep_mask(
            imgs,
            patch_size=patch_size,
            mask_ratio=mask_ratio,
            multi_block_scale_min=multi_block_scale_min,
            multi_block_scale_max=multi_block_scale_max,
            multi_block_aspect_ratio_min=multi_block_aspect_ratio_min,
            multi_block_aspect_ratio_max=multi_block_aspect_ratio_max,
            multi_block_square_aspect_ratio=multi_block_square_aspect_ratio,
        )
    if masking_strategy != "mixed":
        raise ValueError(f"masking_strategy must be one of {MASKING_STRATEGIES}, got {masking_strategy!r}.")

    return _sample_mixed_keep_mask_from_batch_bank(
        batch_size=imgs.shape[0],
        height=imgs.shape[-2],
        width=imgs.shape[-1],
        patch_size=patch_size,
        mask_ratio=mask_ratio,
        device=imgs.device,
        multi_block_scale_min=multi_block_scale_min,
        multi_block_scale_max=multi_block_scale_max,
        multi_block_aspect_ratio_min=multi_block_aspect_ratio_min,
        multi_block_aspect_ratio_max=multi_block_aspect_ratio_max,
        multi_block_square_aspect_ratio=multi_block_square_aspect_ratio,
    )


def sample_mixed_strategy_labels(batch_size: int, device: torch.device) -> torch.LongTensor:
    if batch_size <= 0:
        raise ValueError(f"batch_size must be > 0, got {batch_size}.")
    if batch_size == 1:
        return torch.randint(0, 2, (1,), device=device, dtype=torch.long)

    labels = torch.ones(batch_size, device=device, dtype=torch.long)
    indices = torch.randperm(batch_size, device=device)
    num_random = batch_size // 2
    if batch_size % 2 == 1 and bool(torch.rand((), device=device) < 0.5):
        num_random += 1
    labels[indices[:num_random]] = 0
    return labels


def sample_random_keep_mask(
    imgs: torch.Tensor,
    *,
    patch_size: int,
    mask_ratio: float,
) -> torch.BoolTensor:
    batch_size, _, _, patch_h, patch_w, _, num_keep, _ = _mask_setup(
        imgs,
        patch_size=patch_size,
        mask_ratio=mask_ratio,
    )
    num_patches = patch_h * patch_w
    noise = torch.rand(batch_size, num_patches, device=imgs.device)
    keep_idx = noise.topk(k=num_keep, dim=1, largest=False, sorted=False).indices
    keep_patch = torch.zeros(batch_size, num_patches, device=imgs.device, dtype=torch.bool)
    keep_patch.scatter_(1, keep_idx, True)
    keep_patch = keep_patch.view(batch_size, 1, patch_h, patch_w)
    return _patch_to_pixel_mask(keep_patch, patch_size)


def sample_multiblock_keep_mask(
    imgs: torch.Tensor,
    *,
    patch_size: int,
    mask_ratio: float,
    multi_block_scale_min: float = DEFAULT_MULTI_BLOCK_SCALE_RANGE[0],
    multi_block_scale_max: float = DEFAULT_MULTI_BLOCK_SCALE_RANGE[1],
    multi_block_aspect_ratio_min: float = DEFAULT_MULTI_BLOCK_ASPECT_RATIO_RANGE[0],
    multi_block_aspect_ratio_max: float = DEFAULT_MULTI_BLOCK_ASPECT_RATIO_RANGE[1],
    multi_block_square_aspect_ratio: float = DEFAULT_MULTI_BLOCK_SQUARE_ASPECT_RATIO,
) -> torch.BoolTensor:
    batch_size, _, _, patch_h, patch_w, _, _, num_mask = _mask_setup(
        imgs,
        patch_size=patch_size,
        mask_ratio=mask_ratio,
    )
    _validate_multi_block_ranges(
        multi_block_scale_min=multi_block_scale_min,
        multi_block_scale_max=multi_block_scale_max,
        multi_block_aspect_ratio_min=multi_block_aspect_ratio_min,
        multi_block_aspect_ratio_max=multi_block_aspect_ratio_max,
        multi_block_square_aspect_ratio=multi_block_square_aspect_ratio,
    )

    if num_mask == 0:
        keep_patch = torch.ones((batch_size, 1, patch_h, patch_w), device=imgs.device, dtype=torch.bool)
        return _patch_to_pixel_mask(keep_patch, patch_size)

    return _sample_multiblock_keep_mask_from_bank(
        batch_size=batch_size,
        height=imgs.shape[-2],
        width=imgs.shape[-1],
        patch_size=patch_size,
        mask_ratio=mask_ratio,
        device=imgs.device,
        multi_block_scale_min=multi_block_scale_min,
        multi_block_scale_max=multi_block_scale_max,
        multi_block_aspect_ratio_min=multi_block_aspect_ratio_min,
        multi_block_aspect_ratio_max=multi_block_aspect_ratio_max,
        multi_block_square_aspect_ratio=multi_block_square_aspect_ratio,
    )


def _sample_multiblock_keep_mask_from_bank(
    *,
    batch_size: int,
    height: int,
    width: int,
    patch_size: int,
    mask_ratio: float,
    device: torch.device,
    multi_block_scale_min: float,
    multi_block_scale_max: float,
    multi_block_aspect_ratio_min: float,
    multi_block_aspect_ratio_max: float,
    multi_block_square_aspect_ratio: float,
) -> torch.BoolTensor:
    patch_h = height // patch_size
    patch_w = width // patch_size
    num_patches = patch_h * patch_w
    num_keep = max(1, int(round((1.0 - mask_ratio) * num_patches)))
    num_keep = min(num_patches, num_keep)
    num_mask = num_patches - num_keep
    if num_mask == 0:
        keep_patch = torch.ones((batch_size, 1, patch_h, patch_w), device=device, dtype=torch.bool)
        return _patch_to_pixel_mask(keep_patch, patch_size)

    keep_mask_bank = _get_multiblock_keep_mask_bank(
        patch_h=patch_h,
        patch_w=patch_w,
        patch_size=patch_size,
        num_mask=num_mask,
        device=device,
        multi_block_scale_min=multi_block_scale_min,
        multi_block_scale_max=multi_block_scale_max,
        multi_block_aspect_ratio_min=multi_block_aspect_ratio_min,
        multi_block_aspect_ratio_max=multi_block_aspect_ratio_max,
        multi_block_square_aspect_ratio=multi_block_square_aspect_ratio,
    )
    sample_idx = torch.randint(keep_mask_bank.shape[0], (batch_size,), device=device)
    return keep_mask_bank.index_select(0, sample_idx)


def _mask_setup(
    imgs: torch.Tensor,
    *,
    patch_size: int,
    mask_ratio: float,
) -> tuple[int, int, int, int, int, int, int, int]:
    if imgs.ndim != 4:
        raise ValueError(f"Expected imgs to have shape (B, C, H, W), got {tuple(imgs.shape)}.")
    if patch_size <= 0:
        raise ValueError(f"patch_size must be > 0, got {patch_size}.")
    if not (0.0 <= mask_ratio <= 1.0):
        raise ValueError(f"mask_ratio must be in [0, 1], got {mask_ratio}.")

    batch_size, _, height, width = imgs.shape
    if height % patch_size != 0 or width % patch_size != 0:
        raise ValueError(f"Image size ({height}, {width}) must be divisible by patch_size={patch_size}.")

    patch_h = height // patch_size
    patch_w = width // patch_size
    num_patches = patch_h * patch_w
    num_keep = max(1, int(round((1.0 - mask_ratio) * num_patches)))
    num_keep = min(num_patches, num_keep)
    num_mask = num_patches - num_keep
    return batch_size, height, width, patch_h, patch_w, num_patches, num_keep, num_mask


def _validate_multi_block_ranges(
    *,
    multi_block_scale_min: float,
    multi_block_scale_max: float,
    multi_block_aspect_ratio_min: float,
    multi_block_aspect_ratio_max: float,
    multi_block_square_aspect_ratio: float,
) -> None:
    if multi_block_scale_min <= 0.0 or multi_block_scale_max <= 0.0:
        raise ValueError("Multi-block scales must be > 0.")
    if multi_block_scale_min > multi_block_scale_max:
        raise ValueError("multi_block_scale_min must be <= multi_block_scale_max.")
    if multi_block_aspect_ratio_min <= 0.0 or multi_block_aspect_ratio_max <= 0.0:
        raise ValueError("Multi-block aspect ratios must be > 0.")
    if multi_block_aspect_ratio_min > multi_block_aspect_ratio_max:
        raise ValueError("multi_block_aspect_ratio_min must be <= multi_block_aspect_ratio_max.")
    if multi_block_square_aspect_ratio <= 0.0:
        raise ValueError("multi_block_square_aspect_ratio must be > 0.")


def _multiblock_bank_key(
    *,
    patch_h: int,
    patch_w: int,
    patch_size: int,
    num_mask: int,
    multi_block_scale_min: float,
    multi_block_scale_max: float,
    multi_block_aspect_ratio_min: float,
    multi_block_aspect_ratio_max: float,
    multi_block_square_aspect_ratio: float,
) -> tuple:
    return (
        patch_h,
        patch_w,
        patch_size,
        num_mask,
        float(multi_block_scale_min),
        float(multi_block_scale_max),
        float(multi_block_aspect_ratio_min),
        float(multi_block_aspect_ratio_max),
        float(multi_block_square_aspect_ratio),
        MULTI_BLOCK_BANK_SIZE,
    )


def _mixed_batch_bank_key(
    *,
    batch_size: int,
    patch_h: int,
    patch_w: int,
    patch_size: int,
    num_keep: int,
    num_mask: int,
    multi_block_scale_min: float,
    multi_block_scale_max: float,
    multi_block_aspect_ratio_min: float,
    multi_block_aspect_ratio_max: float,
    multi_block_square_aspect_ratio: float,
) -> tuple:
    return (
        batch_size,
        patch_h,
        patch_w,
        patch_size,
        num_keep,
        num_mask,
        float(multi_block_scale_min),
        float(multi_block_scale_max),
        float(multi_block_aspect_ratio_min),
        float(multi_block_aspect_ratio_max),
        float(multi_block_square_aspect_ratio),
        MIXED_BATCH_BANK_SIZE,
    )


def _random_bank_key(
    *,
    patch_h: int,
    patch_w: int,
    patch_size: int,
    num_keep: int,
) -> tuple:
    return (
        patch_h,
        patch_w,
        patch_size,
        num_keep,
        RANDOM_BANK_SIZE,
    )


def _patch_to_pixel_mask(keep_patch: torch.BoolTensor, patch_size: int) -> torch.BoolTensor:
    return keep_patch.repeat_interleave(patch_size, dim=2).repeat_interleave(patch_size, dim=3)


def _sample_random_keep_mask_from_bank(
    *,
    batch_size: int,
    height: int,
    width: int,
    patch_size: int,
    mask_ratio: float,
    device: torch.device,
) -> torch.BoolTensor:
    patch_h = height // patch_size
    patch_w = width // patch_size
    num_patches = patch_h * patch_w
    num_keep = max(1, int(round((1.0 - mask_ratio) * num_patches)))
    num_keep = min(num_patches, num_keep)
    keep_mask_bank = _get_random_keep_mask_bank(
        patch_h=patch_h,
        patch_w=patch_w,
        patch_size=patch_size,
        num_keep=num_keep,
        device=device,
    )
    sample_idx = torch.randint(keep_mask_bank.shape[0], (batch_size,), device=device)
    return keep_mask_bank.index_select(0, sample_idx)


def _get_random_keep_mask_bank(
    *,
    patch_h: int,
    patch_w: int,
    patch_size: int,
    num_keep: int,
    device: torch.device,
) -> torch.BoolTensor:
    key = _random_bank_key(
        patch_h=patch_h,
        patch_w=patch_w,
        patch_size=patch_size,
        num_keep=num_keep,
    )
    if key not in _RANDOM_PATCH_BANK_CACHE:
        _RANDOM_PATCH_BANK_CACHE[key] = _build_random_patch_bank(
            patch_h=patch_h,
            patch_w=patch_w,
            num_keep=num_keep,
        )

    device_key = key + (str(device),)
    if device_key not in _RANDOM_DEVICE_BANK_CACHE:
        patch_bank = _RANDOM_PATCH_BANK_CACHE[key].to(device=device)
        num_pixels = RANDOM_BANK_SIZE * patch_h * patch_size * patch_w * patch_size
        if num_pixels <= MULTI_BLOCK_PIXEL_BANK_MAX_ELEMENTS:
            keep_mask_bank = _patch_to_pixel_mask(patch_bank.unsqueeze(1), patch_size)
        else:
            keep_mask_bank = patch_bank.unsqueeze(1)
        _RANDOM_DEVICE_BANK_CACHE[device_key] = keep_mask_bank

    keep_mask_bank = _RANDOM_DEVICE_BANK_CACHE[device_key]
    if keep_mask_bank.shape[-2:] == (patch_h, patch_w):
        return _patch_to_pixel_mask(keep_mask_bank, patch_size)
    return keep_mask_bank


def _get_random_patch_bank(
    *,
    patch_h: int,
    patch_w: int,
    patch_size: int,
    num_keep: int,
) -> torch.BoolTensor:
    key = _random_bank_key(
        patch_h=patch_h,
        patch_w=patch_w,
        patch_size=patch_size,
        num_keep=num_keep,
    )
    if key not in _RANDOM_PATCH_BANK_CACHE:
        _RANDOM_PATCH_BANK_CACHE[key] = _build_random_patch_bank(
            patch_h=patch_h,
            patch_w=patch_w,
            num_keep=num_keep,
        )
    return _RANDOM_PATCH_BANK_CACHE[key]


def _build_random_patch_bank(
    *,
    patch_h: int,
    patch_w: int,
    num_keep: int,
) -> torch.BoolTensor:
    chunk_size = min(512, RANDOM_BANK_SIZE)
    num_patches = patch_h * patch_w
    banks = []
    remaining = RANDOM_BANK_SIZE
    while remaining > 0:
        current = min(chunk_size, remaining)
        noise = torch.rand(current, num_patches)
        keep_idx = noise.topk(k=num_keep, dim=1, largest=False, sorted=False).indices
        keep_patch = torch.zeros(current, num_patches, dtype=torch.bool)
        keep_patch.scatter_(1, keep_idx, True)
        banks.append(keep_patch.view(current, patch_h, patch_w))
        remaining -= current
    return torch.cat(banks, dim=0).contiguous()


def _get_multiblock_keep_mask_bank(
    *,
    patch_h: int,
    patch_w: int,
    patch_size: int,
    num_mask: int,
    device: torch.device,
    multi_block_scale_min: float,
    multi_block_scale_max: float,
    multi_block_aspect_ratio_min: float,
    multi_block_aspect_ratio_max: float,
    multi_block_square_aspect_ratio: float,
) -> torch.BoolTensor:
    key = _multiblock_bank_key(
        patch_h=patch_h,
        patch_w=patch_w,
        patch_size=patch_size,
        num_mask=num_mask,
        multi_block_scale_min=multi_block_scale_min,
        multi_block_scale_max=multi_block_scale_max,
        multi_block_aspect_ratio_min=multi_block_aspect_ratio_min,
        multi_block_aspect_ratio_max=multi_block_aspect_ratio_max,
        multi_block_square_aspect_ratio=multi_block_square_aspect_ratio,
    )
    if key not in _MULTI_BLOCK_PATCH_BANK_CACHE:
        _MULTI_BLOCK_PATCH_BANK_CACHE[key] = _build_multiblock_patch_bank(
            patch_h=patch_h,
            patch_w=patch_w,
            num_mask=num_mask,
            multi_block_scale_min=multi_block_scale_min,
            multi_block_scale_max=multi_block_scale_max,
            multi_block_aspect_ratio_min=multi_block_aspect_ratio_min,
            multi_block_aspect_ratio_max=multi_block_aspect_ratio_max,
            multi_block_square_aspect_ratio=multi_block_square_aspect_ratio,
        )

    device_key = key + (str(device),)
    if device_key not in _MULTI_BLOCK_DEVICE_BANK_CACHE:
        patch_bank = _MULTI_BLOCK_PATCH_BANK_CACHE[key].to(device=device)
        num_pixels = MULTI_BLOCK_BANK_SIZE * patch_h * patch_size * patch_w * patch_size
        if num_pixels <= MULTI_BLOCK_PIXEL_BANK_MAX_ELEMENTS:
            keep_mask_bank = _patch_to_pixel_mask(patch_bank.unsqueeze(1), patch_size)
        else:
            keep_mask_bank = patch_bank.unsqueeze(1)
        _MULTI_BLOCK_DEVICE_BANK_CACHE[device_key] = keep_mask_bank

    keep_mask_bank = _MULTI_BLOCK_DEVICE_BANK_CACHE[device_key]
    if keep_mask_bank.shape[-2:] == (patch_h, patch_w):
        return _patch_to_pixel_mask(keep_mask_bank, patch_size)
    return keep_mask_bank


def _get_multiblock_patch_bank(
    *,
    patch_h: int,
    patch_w: int,
    patch_size: int,
    num_mask: int,
    multi_block_scale_min: float,
    multi_block_scale_max: float,
    multi_block_aspect_ratio_min: float,
    multi_block_aspect_ratio_max: float,
    multi_block_square_aspect_ratio: float,
) -> torch.BoolTensor:
    key = _multiblock_bank_key(
        patch_h=patch_h,
        patch_w=patch_w,
        patch_size=patch_size,
        num_mask=num_mask,
        multi_block_scale_min=multi_block_scale_min,
        multi_block_scale_max=multi_block_scale_max,
        multi_block_aspect_ratio_min=multi_block_aspect_ratio_min,
        multi_block_aspect_ratio_max=multi_block_aspect_ratio_max,
        multi_block_square_aspect_ratio=multi_block_square_aspect_ratio,
    )
    if key not in _MULTI_BLOCK_PATCH_BANK_CACHE:
        _MULTI_BLOCK_PATCH_BANK_CACHE[key] = _build_multiblock_patch_bank(
            patch_h=patch_h,
            patch_w=patch_w,
            num_mask=num_mask,
            multi_block_scale_min=multi_block_scale_min,
            multi_block_scale_max=multi_block_scale_max,
            multi_block_aspect_ratio_min=multi_block_aspect_ratio_min,
            multi_block_aspect_ratio_max=multi_block_aspect_ratio_max,
            multi_block_square_aspect_ratio=multi_block_square_aspect_ratio,
        )
    return _MULTI_BLOCK_PATCH_BANK_CACHE[key]


def _sample_mixed_keep_mask_from_batch_bank(
    *,
    batch_size: int,
    height: int,
    width: int,
    patch_size: int,
    mask_ratio: float,
    device: torch.device,
    multi_block_scale_min: float,
    multi_block_scale_max: float,
    multi_block_aspect_ratio_min: float,
    multi_block_aspect_ratio_max: float,
    multi_block_square_aspect_ratio: float,
) -> torch.BoolTensor:
    patch_h = height // patch_size
    patch_w = width // patch_size
    num_patches = patch_h * patch_w
    num_keep = max(1, int(round((1.0 - mask_ratio) * num_patches)))
    num_keep = min(num_patches, num_keep)
    num_mask = num_patches - num_keep
    key = _mixed_batch_bank_key(
        batch_size=batch_size,
        patch_h=patch_h,
        patch_w=patch_w,
        patch_size=patch_size,
        num_keep=num_keep,
        num_mask=num_mask,
        multi_block_scale_min=multi_block_scale_min,
        multi_block_scale_max=multi_block_scale_max,
        multi_block_aspect_ratio_min=multi_block_aspect_ratio_min,
        multi_block_aspect_ratio_max=multi_block_aspect_ratio_max,
        multi_block_square_aspect_ratio=multi_block_square_aspect_ratio,
    )
    if key not in _MIXED_PATCH_BATCH_BANK_CACHE:
        _MIXED_PATCH_BATCH_BANK_CACHE[key] = _build_mixed_patch_batch_bank(
            batch_size=batch_size,
            patch_h=patch_h,
            patch_w=patch_w,
            patch_size=patch_size,
            num_keep=num_keep,
            num_mask=num_mask,
            multi_block_scale_min=multi_block_scale_min,
            multi_block_scale_max=multi_block_scale_max,
            multi_block_aspect_ratio_min=multi_block_aspect_ratio_min,
            multi_block_aspect_ratio_max=multi_block_aspect_ratio_max,
            multi_block_square_aspect_ratio=multi_block_square_aspect_ratio,
        )

    device_key = key + (str(device),)
    if device_key not in _MIXED_DEVICE_BATCH_BANK_CACHE:
        patch_batch_bank = _MIXED_PATCH_BATCH_BANK_CACHE[key].to(device=device)
        num_pixels = MIXED_BATCH_BANK_SIZE * batch_size * patch_h * patch_size * patch_w * patch_size
        if num_pixels <= MIXED_BATCH_PIXEL_BANK_MAX_ELEMENTS:
            flat_patch_bank = patch_batch_bank.reshape(-1, 1, patch_h, patch_w)
            keep_mask_bank = _patch_to_pixel_mask(flat_patch_bank, patch_size).view(
                MIXED_BATCH_BANK_SIZE,
                batch_size,
                1,
                height,
                width,
            )
        else:
            keep_mask_bank = patch_batch_bank.unsqueeze(2)
        _MIXED_DEVICE_BATCH_BANK_CACHE[device_key] = keep_mask_bank

    keep_mask_bank = _MIXED_DEVICE_BATCH_BANK_CACHE[device_key]
    sample_idx = torch.randint(keep_mask_bank.shape[0], (1,), device=device)
    keep_mask = keep_mask_bank.index_select(0, sample_idx).squeeze(0)
    if keep_mask.shape[-2:] == (patch_h, patch_w):
        return _patch_to_pixel_mask(keep_mask, patch_size)
    return keep_mask


def _build_mixed_patch_batch_bank(
    *,
    batch_size: int,
    patch_h: int,
    patch_w: int,
    patch_size: int,
    num_keep: int,
    num_mask: int,
    multi_block_scale_min: float,
    multi_block_scale_max: float,
    multi_block_aspect_ratio_min: float,
    multi_block_aspect_ratio_max: float,
    multi_block_square_aspect_ratio: float,
) -> torch.BoolTensor:
    random_patch_bank = _get_random_patch_bank(
        patch_h=patch_h,
        patch_w=patch_w,
        patch_size=patch_size,
        num_keep=num_keep,
    )
    multi_block_patch_bank = _get_multiblock_patch_bank(
        patch_h=patch_h,
        patch_w=patch_w,
        patch_size=patch_size,
        num_mask=num_mask,
        multi_block_scale_min=multi_block_scale_min,
        multi_block_scale_max=multi_block_scale_max,
        multi_block_aspect_ratio_min=multi_block_aspect_ratio_min,
        multi_block_aspect_ratio_max=multi_block_aspect_ratio_max,
        multi_block_square_aspect_ratio=multi_block_square_aspect_ratio,
    )

    mixed_batches = []
    for _ in range(MIXED_BATCH_BANK_SIZE):
        num_random = batch_size // 2
        if batch_size % 2 == 1 and bool(torch.rand(()) < 0.5):
            num_random += 1
        num_multi_block = batch_size - num_random

        entries = []
        if num_random > 0:
            random_idx = torch.randint(random_patch_bank.shape[0], (num_random,))
            entries.append(random_patch_bank.index_select(0, random_idx))
        if num_multi_block > 0:
            multi_idx = torch.randint(multi_block_patch_bank.shape[0], (num_multi_block,))
            entries.append(multi_block_patch_bank.index_select(0, multi_idx))

        keep_patch = torch.cat(entries, dim=0)
        order = torch.randperm(batch_size)
        mixed_batches.append(keep_patch.index_select(0, order))

    return torch.stack(mixed_batches, dim=0).contiguous()


def _build_multiblock_patch_bank(
    *,
    patch_h: int,
    patch_w: int,
    num_mask: int,
    multi_block_scale_min: float,
    multi_block_scale_max: float,
    multi_block_aspect_ratio_min: float,
    multi_block_aspect_ratio_max: float,
    multi_block_square_aspect_ratio: float,
) -> torch.BoolTensor:
    chunk_size = min(512, MULTI_BLOCK_BANK_SIZE)
    banks = []
    remaining = MULTI_BLOCK_BANK_SIZE
    while remaining > 0:
        current = min(chunk_size, remaining)
        banks.append(
            _sample_multiblock_keep_patch_direct(
                batch_size=current,
                patch_h=patch_h,
                patch_w=patch_w,
                num_mask=num_mask,
                device=torch.device("cpu"),
                multi_block_scale_min=multi_block_scale_min,
                multi_block_scale_max=multi_block_scale_max,
                multi_block_aspect_ratio_min=multi_block_aspect_ratio_min,
                multi_block_aspect_ratio_max=multi_block_aspect_ratio_max,
                multi_block_square_aspect_ratio=multi_block_square_aspect_ratio,
            )
        )
        remaining -= current
    return torch.cat(banks, dim=0).contiguous()


def _sample_multiblock_keep_patch_direct(
    *,
    batch_size: int,
    patch_h: int,
    patch_w: int,
    num_mask: int,
    device: torch.device,
    multi_block_scale_min: float,
    multi_block_scale_max: float,
    multi_block_aspect_ratio_min: float,
    multi_block_aspect_ratio_max: float,
    multi_block_square_aspect_ratio: float,
) -> torch.BoolTensor:
    keep_patch = torch.ones((batch_size, patch_h, patch_w), device=device, dtype=torch.bool)
    if num_mask == 0:
        return keep_patch

    num_blocks = min(3, num_mask)
    raw_scales = torch.empty(batch_size, num_blocks, device=device).uniform_(
        multi_block_scale_min,
        multi_block_scale_max,
    )
    target_areas = _allocate_block_areas(raw_scales, total_area=num_mask)
    aspect_ratios = _sample_block_aspect_ratios(
        batch_size=batch_size,
        num_blocks=num_blocks,
        device=device,
        multi_block_aspect_ratio_min=multi_block_aspect_ratio_min,
        multi_block_aspect_ratio_max=multi_block_aspect_ratio_max,
        multi_block_square_aspect_ratio=multi_block_square_aspect_ratio,
    )
    block_h, block_w = _sample_block_shapes(
        target_areas=target_areas,
        aspect_ratios=aspect_ratios,
        max_h=patch_h,
        max_w=patch_w,
    )
    top, left = _sample_block_positions(
        block_h=block_h,
        block_w=block_w,
        grid_h=patch_h,
        grid_w=patch_w,
        min_gap=MULTI_BLOCK_MIN_GAP,
    )
    keep_patch[:] = ~_rectangles_to_mask(
        top=top,
        left=left,
        block_h=block_h,
        block_w=block_w,
        grid_h=patch_h,
        grid_w=patch_w,
    )
    return keep_patch


def _allocate_block_areas(raw_scales: torch.Tensor, *, total_area: int) -> torch.LongTensor:
    num_blocks = raw_scales.shape[-1]
    if num_blocks == 0:
        return torch.zeros_like(raw_scales, dtype=torch.long)

    base = torch.ones_like(raw_scales, dtype=torch.long)
    remaining = total_area - num_blocks
    if remaining <= 0:
        return base

    weights = raw_scales / raw_scales.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    extra_float = weights * remaining
    extra = extra_float.floor().to(dtype=torch.long)
    leftover = remaining - extra.sum(dim=-1)
    if bool((leftover > 0).any()):
        residual = extra_float - extra.to(dtype=extra_float.dtype)
        order = residual.argsort(dim=-1, descending=True)
        rank = torch.arange(num_blocks, device=raw_scales.device).unsqueeze(0)
        add = (rank < leftover.unsqueeze(1)).to(dtype=extra.dtype)
        extra.scatter_add_(dim=-1, index=order, src=add)
    return base + extra


def _sample_block_aspect_ratios(
    *,
    batch_size: int,
    num_blocks: int,
    device: torch.device,
    multi_block_aspect_ratio_min: float,
    multi_block_aspect_ratio_max: float,
    multi_block_square_aspect_ratio: float,
) -> torch.Tensor:
    aspect_ratios = torch.empty(batch_size, num_blocks, device=device).uniform_(
        multi_block_aspect_ratio_min,
        multi_block_aspect_ratio_max,
    )
    aspect_ratios[:, 0] = float(multi_block_square_aspect_ratio)
    return aspect_ratios


def _sample_block_shapes(
    *,
    target_areas: torch.LongTensor,
    aspect_ratios: torch.Tensor,
    max_h: int,
    max_w: int,
) -> tuple[torch.LongTensor, torch.LongTensor]:
    target = target_areas.to(dtype=aspect_ratios.dtype).clamp_min(1.0)
    block_h = torch.sqrt(target / aspect_ratios.clamp_min(1e-8)).round().to(dtype=torch.long)
    block_w = torch.sqrt(target * aspect_ratios).round().to(dtype=torch.long)
    block_h.clamp_(1, max_h)
    block_w.clamp_(1, max_w)

    for _ in range(max_h + max_w):
        area = block_h * block_w
        too_big = area > target_areas
        if not bool(too_big.any()):
            break
        shrink_h = too_big & (block_h >= block_w) & (block_h > 1)
        shrink_w = too_big & ~shrink_h & (block_w > 1)
        fallback_h = too_big & ~(shrink_h | shrink_w) & (block_h > 1)
        fallback_w = too_big & ~(shrink_h | shrink_w | fallback_h) & (block_w > 1)
        block_h = block_h - shrink_h.to(dtype=block_h.dtype) - fallback_h.to(dtype=block_h.dtype)
        block_w = block_w - shrink_w.to(dtype=block_w.dtype) - fallback_w.to(dtype=block_w.dtype)

    return block_h.clamp_min(1), block_w.clamp_min(1)


def _sample_block_positions(
    *,
    block_h: torch.LongTensor,
    block_w: torch.LongTensor,
    grid_h: int,
    grid_w: int,
    min_gap: int = 1,
) -> tuple[torch.LongTensor, torch.LongTensor]:
    batch_size, num_blocks = block_h.shape
    device = block_h.device
    top = torch.zeros_like(block_h)
    left = torch.zeros_like(block_w)
    total_positions = grid_h * grid_w
    num_candidates = min(total_positions, MULTI_BLOCK_POSITION_CANDIDATES)
    if total_positions <= MULTI_BLOCK_POSITION_CANDIDATES:
        all_top = torch.arange(grid_h, device=device).view(1, grid_h, 1).expand(1, grid_h, grid_w).reshape(1, -1)
        all_left = torch.arange(grid_w, device=device).view(1, 1, grid_w).expand(1, grid_h, grid_w).reshape(1, -1)
    else:
        all_top = None
        all_left = None

    for block_idx in range(num_blocks):
        max_top = (grid_h - block_h[:, block_idx] + 1).clamp_min(1)
        max_left = (grid_w - block_w[:, block_idx] + 1).clamp_min(1)
        if block_idx == 0:
            top[:, block_idx] = (torch.rand(batch_size, device=device) * max_top.to(dtype=torch.float32)).floor().to(dtype=top.dtype)
            left[:, block_idx] = (torch.rand(batch_size, device=device) * max_left.to(dtype=torch.float32)).floor().to(dtype=left.dtype)
            continue

        if all_top is not None:
            cand_top = all_top.expand(batch_size, -1)
            cand_left = all_left.expand(batch_size, -1)
            valid = (cand_top < max_top.unsqueeze(1)) & (cand_left < max_left.unsqueeze(1))
        else:
            cand_top = (torch.rand(batch_size, num_candidates, device=device) * max_top.unsqueeze(1).to(dtype=torch.float32)).floor().to(dtype=top.dtype)
            cand_left = (torch.rand(batch_size, num_candidates, device=device) * max_left.unsqueeze(1).to(dtype=torch.float32)).floor().to(dtype=left.dtype)
            valid = torch.ones((batch_size, num_candidates), device=device, dtype=torch.bool)

        score = torch.rand(valid.shape, device=device)

        cand_bottom = cand_top + block_h[:, block_idx].unsqueeze(1)
        cand_right = cand_left + block_w[:, block_idx].unsqueeze(1)
        cand_center_y = cand_top.to(dtype=torch.float32) + 0.5 * block_h[:, block_idx].to(dtype=torch.float32).unsqueeze(1)
        cand_center_x = cand_left.to(dtype=torch.float32) + 0.5 * block_w[:, block_idx].to(dtype=torch.float32).unsqueeze(1)
        min_distance = torch.full_like(score, float("inf"))
        valid_nogap = valid.clone()
        valid_gap = valid.clone()

        for prev_idx in range(block_idx):
            prev_top = top[:, prev_idx].unsqueeze(1)
            prev_left = left[:, prev_idx].unsqueeze(1)
            prev_bottom = prev_top + block_h[:, prev_idx].unsqueeze(1)
            prev_right = prev_left + block_w[:, prev_idx].unsqueeze(1)

            valid_nogap &= (
                (cand_bottom <= prev_top)
                | (cand_top >= prev_bottom)
                | (cand_right <= prev_left)
                | (cand_left >= prev_right)
            )
            valid_gap &= (
                (cand_bottom + min_gap <= prev_top)
                | (cand_top >= prev_bottom + min_gap)
                | (cand_right + min_gap <= prev_left)
                | (cand_left >= prev_right + min_gap)
            )

            prev_center_y = prev_top.to(dtype=torch.float32) + 0.5 * block_h[:, prev_idx].to(dtype=torch.float32).unsqueeze(1)
            prev_center_x = prev_left.to(dtype=torch.float32) + 0.5 * block_w[:, prev_idx].to(dtype=torch.float32).unsqueeze(1)
            distance = torch.sqrt((cand_center_y - prev_center_y).square() + (cand_center_x - prev_center_x).square())
            min_distance = torch.minimum(min_distance, distance)

        score = (
            valid_gap.to(dtype=score.dtype) * 1_000_000.0
            + valid_nogap.to(dtype=score.dtype) * 100_000.0
            + min_distance
            + score
        )
        no_non_overlap = ~valid_nogap.any(dim=1)
        if bool(no_non_overlap.any()):
            fallback_score = torch.where(
                valid[no_non_overlap],
                min_distance[no_non_overlap] + torch.rand_like(score[no_non_overlap]),
                score[no_non_overlap].new_full((), float("-inf")),
            )
            score[no_non_overlap] = fallback_score

        score = torch.where(valid, score, score.new_full((), float("-inf")))
        best_idx = score.argmax(dim=1)
        top[:, block_idx] = cand_top.gather(1, best_idx.unsqueeze(1)).squeeze(1)
        left[:, block_idx] = cand_left.gather(1, best_idx.unsqueeze(1)).squeeze(1)

    return top, left


def _rectangles_to_mask(
    *,
    top: torch.LongTensor,
    left: torch.LongTensor,
    block_h: torch.LongTensor,
    block_w: torch.LongTensor,
    grid_h: int,
    grid_w: int,
) -> torch.BoolTensor:
    batch_size, num_blocks = top.shape
    yy = torch.arange(grid_h, device=top.device).view(1, 1, grid_h, 1)
    xx = torch.arange(grid_w, device=top.device).view(1, 1, 1, grid_w)
    top = top.view(batch_size, num_blocks, 1, 1)
    left = left.view(batch_size, num_blocks, 1, 1)
    bottom = top + block_h.view(batch_size, num_blocks, 1, 1)
    right = left + block_w.view(batch_size, num_blocks, 1, 1)
    return ((yy >= top) & (yy < bottom) & (xx >= left) & (xx < right)).any(dim=1)
