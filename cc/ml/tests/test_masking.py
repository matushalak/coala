from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import torch

from cc.ml.masking import (
    add_masking_arguments,
    sample_keep_mask,
    sample_mixed_strategy_labels,
    sample_multiblock_keep_mask,
    sample_random_keep_mask,
)


def _normalize_image_size(image_size: int | tuple[int, int]) -> tuple[int, int]:
    if isinstance(image_size, int):
        return image_size, image_size
    return image_size


def _expected_masked_patches(*, image_size: int | tuple[int, int], patch_size: int, mask_ratio: float) -> int:
    height, width = _normalize_image_size(image_size)
    num_patches = (height // patch_size) * (width // patch_size)
    num_keep = max(1, int(round((1.0 - mask_ratio) * num_patches)))
    return num_patches - num_keep


def _masked_patch_counts(keep_mask: torch.BoolTensor, *, patch_size: int) -> torch.Tensor:
    masked_patch = ~keep_mask[:, 0, ::patch_size, ::patch_size]
    return masked_patch.sum(dim=(1, 2))


def _has_adjacent_masked_patches(masked_patch: torch.BoolTensor) -> bool:
    horizontal = bool((masked_patch[:, 1:] & masked_patch[:, :-1]).any())
    vertical = bool((masked_patch[1:, :] & masked_patch[:-1, :]).any())
    return horizontal or vertical


def _rectangle_components(masked_patch: torch.BoolTensor) -> list[set[tuple[int, int]]]:
    coords = {
        (int(row_idx), int(col_idx))
        for row_idx, col_idx in zip(*masked_patch.nonzero(as_tuple=True))
    }
    components: list[set[tuple[int, int]]] = []
    while coords:
        start = coords.pop()
        stack = [start]
        component = {start}
        while stack:
            row_idx, col_idx = stack.pop()
            for next_row, next_col in (
                (row_idx - 1, col_idx),
                (row_idx + 1, col_idx),
                (row_idx, col_idx - 1),
                (row_idx, col_idx + 1),
            ):
                if (next_row, next_col) in coords:
                    coords.remove((next_row, next_col))
                    component.add((next_row, next_col))
                    stack.append((next_row, next_col))
        components.append(component)
    return components


def _assert_components_are_rectangles(masked_patch: torch.BoolTensor) -> None:
    for component in _rectangle_components(masked_patch):
        rows = [row_idx for row_idx, _ in component]
        cols = [col_idx for _, col_idx in component]
        row_min, row_max = min(rows), max(rows)
        col_min, col_max = min(cols), max(cols)
        expected_component = {
            (row_idx, col_idx)
            for row_idx in range(row_min, row_max + 1)
            for col_idx in range(col_min, col_max + 1)
        }
        assert component == expected_component


def _assert_multiblock_budget_is_reasonable(masked_counts: torch.Tensor, *, expected: int) -> None:
    tolerance = max(3, math.ceil(0.25 * expected))
    lower_bound = max(1, expected - tolerance)
    assert int(masked_counts.max().item()) <= expected
    assert int(masked_counts.min().item()) >= lower_bound


def _make_demo_images(
    *,
    num_images: int,
    image_size: tuple[int, int],
    num_channels: int,
) -> torch.Tensor:
    height, width = image_size
    ys = torch.linspace(0.0, 1.0, height)
    xs = torch.linspace(0.0, 1.0, width)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    checker = ((torch.floor(xx * 8.0) + torch.floor(yy * 8.0)) % 2.0).float()

    imgs = []
    for idx in range(num_images):
        phase = idx / max(1, num_images - 1)
        freq_x = 2 + (idx % 3)
        freq_y = 3 + ((idx + 1) % 4)
        carrier = (
            0.35 * torch.sin(freq_x * math.pi * xx + phase * math.pi)
            + 0.35 * torch.cos(freq_y * math.pi * yy - phase * math.pi / 2.0)
            + 0.15 * checker
            + 0.15 * (xx * yy)
        )
        carrier = carrier.sub(carrier.amin()).div((carrier.amax() - carrier.amin()).clamp_min(1e-8))

        if num_channels == 1:
            img = carrier.unsqueeze(0)
        elif num_channels == 3:
            img = torch.stack([xx, yy, carrier], dim=0)
        else:
            channels = [carrier]
            for channel_idx in range(1, num_channels):
                shifted = torch.roll(
                    carrier,
                    shifts=(2 * channel_idx, -3 * channel_idx),
                    dims=(0, 1),
                )
                channels.append(shifted)
            img = torch.stack(channels[:num_channels], dim=0)
        imgs.append(img)

    return torch.stack(imgs, dim=0)


def _expand_to_rgb(imgs: torch.Tensor) -> torch.Tensor:
    if imgs.shape[1] == 1:
        return imgs.expand(-1, 3, -1, -1)
    if imgs.shape[1] >= 3:
        return imgs[:, :3]
    return torch.cat([imgs, imgs[:, :1]], dim=1)


def _mask_overlay(imgs: torch.Tensor, keep_mask: torch.BoolTensor) -> torch.Tensor:
    base = _expand_to_rgb(imgs)
    masked = (~keep_mask).expand(-1, 3, -1, -1)
    tint = base.new_tensor((1.0, 0.2, 0.2)).view(1, 3, 1, 1)
    return torch.where(masked, 0.3 * base + 0.7 * tint, base)


def _mask_map(keep_mask: torch.BoolTensor) -> torch.Tensor:
    masked = (~keep_mask).float()
    visible = keep_mask.float()
    return torch.cat([masked, 0.8 * visible, 0.8 * visible], dim=1)


def _sample_keep_mask_and_labels(
    imgs: torch.Tensor,
    *,
    patch_size: int,
    mask_ratio: float,
    masking_strategy: str,
    multi_block_scale_min: float,
    multi_block_scale_max: float,
    multi_block_aspect_ratio_min: float,
    multi_block_aspect_ratio_max: float,
    multi_block_square_aspect_ratio: float,
) -> tuple[torch.BoolTensor, list[str]]:
    if masking_strategy != "mixed":
        keep_mask = sample_keep_mask(
            imgs,
            patch_size=patch_size,
            mask_ratio=mask_ratio,
            masking_strategy=masking_strategy,
            multi_block_scale_min=multi_block_scale_min,
            multi_block_scale_max=multi_block_scale_max,
            multi_block_aspect_ratio_min=multi_block_aspect_ratio_min,
            multi_block_aspect_ratio_max=multi_block_aspect_ratio_max,
            multi_block_square_aspect_ratio=multi_block_square_aspect_ratio,
        )
        return keep_mask, [masking_strategy] * imgs.shape[0]

    labels = sample_mixed_strategy_labels(imgs.shape[0], imgs.device)
    random_idx = (labels == 0).nonzero(as_tuple=False).squeeze(1)
    multiblock_idx = (labels == 1).nonzero(as_tuple=False).squeeze(1)
    keep_mask = torch.empty((imgs.shape[0], 1, imgs.shape[-2], imgs.shape[-1]), dtype=torch.bool, device=imgs.device)

    if random_idx.numel() > 0:
        keep_mask[random_idx] = sample_random_keep_mask(
            imgs[random_idx],
            patch_size=patch_size,
            mask_ratio=mask_ratio,
        )
    if multiblock_idx.numel() > 0:
        keep_mask[multiblock_idx] = sample_multiblock_keep_mask(
            imgs[multiblock_idx],
            patch_size=patch_size,
            mask_ratio=mask_ratio,
            multi_block_scale_min=multi_block_scale_min,
            multi_block_scale_max=multi_block_scale_max,
            multi_block_aspect_ratio_min=multi_block_aspect_ratio_min,
            multi_block_aspect_ratio_max=multi_block_aspect_ratio_max,
            multi_block_square_aspect_ratio=multi_block_square_aspect_ratio,
        )

    strategy_names = ["random" if label.item() == 0 else "multi-block" for label in labels]
    return keep_mask, strategy_names


def _plot_tensor(ax, img: torch.Tensor, *, patch_size: int) -> None:
    if img.shape[0] == 1:
        ax.imshow(img[0].detach().cpu(), cmap="gray", interpolation="nearest", vmin=0.0, vmax=1.0)
    else:
        ax.imshow(img.movedim(0, -1).detach().cpu().clamp(0.0, 1.0), interpolation="nearest", vmin=0.0, vmax=1.0)
    height, width = img.shape[-2:]
    ax.set_xticks([x - 0.5 for x in range(0, width + 1, patch_size)], minor=True)
    ax.set_yticks([y - 0.5 for y in range(0, height + 1, patch_size)], minor=True)
    ax.grid(which="minor", color="white", linewidth=0.4, alpha=0.6)
    ax.tick_params(which="both", bottom=False, left=False, labelbottom=False, labelleft=False)


def _render_mask_figure(
    *,
    imgs: torch.Tensor,
    keep_mask: torch.BoolTensor,
    strategy_names: list[str],
    patch_size: int,
    save_path: str | Path | None,
    show: bool,
) -> None:
    import matplotlib.pyplot as plt

    display_rows = [
        ("Original", imgs),
        ("Masked Input", torch.where(keep_mask, imgs, imgs.new_full((), 0.1))),
        ("Overlay", _mask_overlay(imgs, keep_mask)),
        ("Mask Map", _mask_map(keep_mask)),
    ]
    masked_counts = _masked_patch_counts(keep_mask, patch_size=patch_size).tolist()
    total_patches = (imgs.shape[-2] // patch_size) * (imgs.shape[-1] // patch_size)

    fig, axes = plt.subplots(
        len(display_rows),
        imgs.shape[0],
        squeeze=False,
        figsize=(3.0 * imgs.shape[0], 3.0 * len(display_rows)),
    )
    for row_idx, (row_name, row_imgs) in enumerate(display_rows):
        for col_idx in range(imgs.shape[0]):
            ax = axes[row_idx][col_idx]
            _plot_tensor(ax, row_imgs[col_idx], patch_size=patch_size)
            if row_idx == 0:
                masked_count = masked_counts[col_idx]
                ax.set_title(
                    f"{strategy_names[col_idx]}\n{masked_count}/{total_patches} masked",
                    fontsize=10,
                )
            if col_idx == 0:
                ax.set_ylabel(row_name, fontsize=10)

    fig.tight_layout()
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=180, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def visualize_masks(args: argparse.Namespace) -> None:
    image_size = _parse_image_size_arg(args.image_size)
    torch.manual_seed(args.seed)

    imgs = _make_demo_images(
        num_images=args.num_images,
        image_size=image_size,
        num_channels=args.num_channels,
    )
    keep_mask, strategy_names = _sample_keep_mask_and_labels(
        imgs,
        patch_size=args.patch_size,
        mask_ratio=args.mask_ratio,
        masking_strategy=args.masking_strategy,
        multi_block_scale_min=args.multi_block_scale_min,
        multi_block_scale_max=args.multi_block_scale_max,
        multi_block_aspect_ratio_min=args.multi_block_aspect_ratio_min,
        multi_block_aspect_ratio_max=args.multi_block_aspect_ratio_max,
        multi_block_square_aspect_ratio=args.multi_block_square_aspect_ratio,
    )

    masked_counts = _masked_patch_counts(keep_mask, patch_size=args.patch_size)
    total_patches = (image_size[0] // args.patch_size) * (image_size[1] // args.patch_size)
    expected_masked = _expected_masked_patches(
        image_size=image_size,
        patch_size=args.patch_size,
        mask_ratio=args.mask_ratio,
    )

    print(
        f"image_size={image_size[0]}x{image_size[1]} | patch_size={args.patch_size} | "
        f"mask_ratio={args.mask_ratio:.3f} | strategy={args.masking_strategy}"
    )
    print(f"expected masked patches per image: {expected_masked}/{total_patches}")
    for idx, (strategy_name, masked_count) in enumerate(zip(strategy_names, masked_counts.tolist())):
        print(
            f"sample {idx}: strategy={strategy_name}, "
            f"masked_patches={masked_count}/{total_patches}, "
            f"masked_ratio={masked_count / total_patches:.3f}"
        )

    _render_mask_figure(
        imgs=imgs,
        keep_mask=keep_mask,
        strategy_names=strategy_names,
        patch_size=args.patch_size,
        save_path=args.save_path,
        show=args.show,
    )


def _parse_image_size_arg(values: list[int]) -> tuple[int, int]:
    if len(values) == 1:
        return values[0], values[0]
    if len(values) == 2:
        return values[0], values[1]
    raise ValueError("--image_size expects one integer (square image) or two integers (height width).")


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--image_size",
        nargs="+",
        default=[28],
        type=int,
        help="Synthetic image size. Pass one value for square images or two values for height width.",
    )
    parser.add_argument("--num_images", default=6, type=int, help="Number of sample masks to visualize.")
    parser.add_argument("--num_channels", default=1, type=int, help="Number of channels in the synthetic demo image.")
    parser.add_argument("--seed", default=0, type=int, help="Random seed used for mask sampling.")
    parser.add_argument("--save_path", default=None, type=str, help="Optional PNG path for saving the rendered figure.")
    parser.add_argument("--show", dest="show", action="store_true", help="Display the rendered figure.")
    parser.add_argument("--no_show", dest="show", action="store_false", help="Skip interactive figure display.")
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Time mask sampling instead of rendering visualizations.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        type=str,
        help="Torch device used for benchmarking, for example cpu or cuda.",
    )
    parser.add_argument(
        "--warmup_iters",
        default=25,
        type=int,
        help="Warmup iterations to run before timing when --benchmark is set.",
    )
    parser.add_argument(
        "--benchmark_iters",
        default=250,
        type=int,
        help="Measured iterations to run per masking strategy when --benchmark is set.",
    )
    parser.set_defaults(show=True)
    add_masking_arguments(parser)
    return parser


def _synchronize_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def benchmark_masks(args: argparse.Namespace) -> None:
    image_size = _parse_image_size_arg(args.image_size)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA benchmarking requested but torch.cuda.is_available() is False.")

    torch.manual_seed(args.seed)
    imgs = _make_demo_images(
        num_images=args.num_images,
        image_size=image_size,
        num_channels=args.num_channels,
    ).to(device)

    strategies = ("random", "multi-block", "mixed")
    timings_ms: dict[str, float] = {}

    common_kwargs = {
        "patch_size": args.patch_size,
        "mask_ratio": args.mask_ratio,
        "multi_block_scale_min": args.multi_block_scale_min,
        "multi_block_scale_max": args.multi_block_scale_max,
        "multi_block_aspect_ratio_min": args.multi_block_aspect_ratio_min,
        "multi_block_aspect_ratio_max": args.multi_block_aspect_ratio_max,
        "multi_block_square_aspect_ratio": args.multi_block_square_aspect_ratio,
    }

    print(
        f"benchmarking on device={device} | batch_size={args.num_images} | "
        f"image_size={image_size[0]}x{image_size[1]} | patch_size={args.patch_size} | "
        f"mask_ratio={args.mask_ratio:.3f}"
    )

    with torch.no_grad():
        for strategy in strategies:
            for _ in range(args.warmup_iters):
                sample_keep_mask(imgs, masking_strategy=strategy, **common_kwargs)
            _synchronize_device(device)

            start = time.perf_counter()
            for _ in range(args.benchmark_iters):
                sample_keep_mask(imgs, masking_strategy=strategy, **common_kwargs)
            _synchronize_device(device)
            elapsed = time.perf_counter() - start
            timings_ms[strategy] = 1000.0 * elapsed / args.benchmark_iters

    baseline_ms = timings_ms["random"]
    for strategy in strategies:
        slowdown = timings_ms[strategy] / baseline_ms if baseline_ms > 0.0 else float("inf")
        print(f"{strategy:11s} {timings_ms[strategy]:8.3f} ms/iter | slowdown x{slowdown:.3f}")


def test_random_mask_sampler_matches_patch_budget():
    torch.manual_seed(0)
    imgs = torch.randn(4, 1, 28, 28)
    keep_mask = sample_random_keep_mask(imgs, patch_size=4, mask_ratio=0.6)

    expected = _expected_masked_patches(image_size=28, patch_size=4, mask_ratio=0.6)
    masked_counts = _masked_patch_counts(keep_mask, patch_size=4)

    assert keep_mask.shape == imgs.shape[:1] + (1, 28, 28)
    assert torch.equal(masked_counts, torch.full_like(masked_counts, expected))


def test_multiblock_mask_sampler_matches_patch_budget_and_keeps_blocks_contiguous():
    torch.manual_seed(0)
    imgs = torch.randn(4, 1, 28, 28)
    keep_mask = sample_multiblock_keep_mask(imgs, patch_size=4, mask_ratio=0.6)

    expected = _expected_masked_patches(image_size=28, patch_size=4, mask_ratio=0.6)
    masked_counts = _masked_patch_counts(keep_mask, patch_size=4)

    _assert_multiblock_budget_is_reasonable(masked_counts, expected=expected)
    masked_patch = ~keep_mask[0, 0, ::4, ::4]
    assert _has_adjacent_masked_patches(masked_patch)
    _assert_components_are_rectangles(masked_patch)


def test_mixed_strategy_labels_split_batches_evenly():
    torch.manual_seed(0)
    labels = sample_mixed_strategy_labels(batch_size=8, device=torch.device("cpu"))

    assert labels.shape == (8,)
    assert int((labels == 0).sum().item()) == 4
    assert int((labels == 1).sum().item()) == 4


def test_mae_supports_mixed_masking_strategy():
    from cc.ml.MAEmodel import MAE

    torch.manual_seed(0)
    model = MAE(
        num_filters=8,
        lr=1e-3,
        mask_ratio=0.6,
        patch_size=4,
        masked_loss_weight=1.0,
        masking_strategy="mixed",
    )
    imgs = torch.randn(6, 1, 28, 28)

    keep_mask = model._mask(imgs)
    expected = _expected_masked_patches(image_size=28, patch_size=4, mask_ratio=0.6)
    masked_counts = _masked_patch_counts(keep_mask, patch_size=4)

    _assert_multiblock_budget_is_reasonable(masked_counts, expected=expected)
    assert keep_mask.dtype == torch.bool
    assert keep_mask.shape == (6, 1, 28, 28)


def test_shared_mask_sampler_accepts_mixed_strategy():
    torch.manual_seed(0)
    imgs = torch.randn(6, 1, 28, 28)
    keep_mask = sample_keep_mask(imgs, patch_size=4, mask_ratio=0.6, masking_strategy="mixed")

    expected = _expected_masked_patches(image_size=28, patch_size=4, mask_ratio=0.6)
    _assert_multiblock_budget_is_reasonable(_masked_patch_counts(keep_mask, patch_size=4), expected=expected)


if __name__ == "__main__":
    parsed_args = _build_arg_parser().parse_args()
    if parsed_args.benchmark:
        benchmark_masks(parsed_args)
    else:
        visualize_masks(parsed_args)
