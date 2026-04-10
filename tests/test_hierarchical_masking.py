from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import sys
import tempfile

import torch

# Allow direct script execution from inside this folder.
if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[3]))

from coala.architecture.modules.utils import downsample_center_mask
from coala.utils.masking import add_masking_arguments, sample_keep_mask


def _parse_image_size_arg(values: list[int]) -> tuple[int, int]:
    if len(values) == 1:
        return values[0], values[0]
    if len(values) == 2:
        return values[0], values[1]
    raise ValueError("--image_size expects one integer (square image) or two integers (height width).")


def _next_out_size(spatial_size: tuple[int, int], stride: int) -> tuple[int, int]:
    height, width = spatial_size
    return math.ceil(height / stride), math.ceil(width / stride)


def _sample_sites_for_next_level(
    keep_mask: torch.BoolTensor,
    *,
    out_size: tuple[int, int],
    stride: int,
) -> torch.BoolTensor:
    out_h, out_w = out_size
    row_idx = torch.arange(out_h, device=keep_mask.device) * stride
    col_idx = torch.arange(out_w, device=keep_mask.device) * stride
    grid_rows, grid_cols = torch.meshgrid(row_idx, col_idx, indexing="ij")

    sample_sites = torch.zeros_like(keep_mask)
    sample_sites[:, :, grid_rows.reshape(-1), grid_cols.reshape(-1)] = True
    return sample_sites


def _build_mask_hierarchy(
    keep_mask: torch.BoolTensor,
    *,
    num_layers: int,
    stride: int,
) -> tuple[list[torch.BoolTensor], list[torch.BoolTensor]]:
    if num_layers < 0:
        raise ValueError(f"num_layers must be >= 0, got {num_layers}.")
    if stride <= 0:
        raise ValueError(f"stride must be > 0, got {stride}.")

    level_masks = [keep_mask.bool()]
    sample_site_masks: list[torch.BoolTensor] = []

    for _ in range(num_layers):
        current_mask = level_masks[-1]
        out_size = _next_out_size(current_mask.shape[-2:], stride)
        sample_site_masks.append(
            _sample_sites_for_next_level(
                current_mask,
                out_size=out_size,
                stride=stride,
            )
        )
        level_masks.append(
            downsample_center_mask(
                current_mask,
                out_size=out_size,
                stride=stride,
            )
        )

    return level_masks, sample_site_masks


def _sample_initial_keep_mask(
    *,
    image_size: tuple[int, int],
    patch_size: int,
    mask_ratio: float,
    masking_strategy: str,
    multi_block_scale_min: float,
    multi_block_scale_max: float,
    multi_block_aspect_ratio_min: float,
    multi_block_aspect_ratio_max: float,
    multi_block_square_aspect_ratio: float,
) -> torch.BoolTensor:
    height, width = image_size
    imgs = torch.zeros((1, 1, height, width), dtype=torch.float32)
    return sample_keep_mask(
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


def _mask_counts(keep_mask: torch.BoolTensor) -> tuple[int, int, int]:
    total = keep_mask.shape[-2] * keep_mask.shape[-1]
    keep = int(keep_mask[0, 0].sum().item())
    masked = total - keep
    return keep, masked, total


def _mask_rgb(keep_mask: torch.BoolTensor) -> torch.Tensor:
    keep_color = torch.tensor((0.93, 0.93, 0.93), dtype=torch.float32, device=keep_mask.device).view(1, 3, 1, 1)
    masked_color = torch.tensor((0.95, 0.28, 0.28), dtype=torch.float32, device=keep_mask.device).view(1, 3, 1, 1)
    return torch.where(keep_mask.expand(-1, 3, -1, -1), keep_color, masked_color)


def _sample_overlay(keep_mask: torch.BoolTensor, sample_sites: torch.BoolTensor) -> torch.Tensor:
    base = _mask_rgb(keep_mask)
    sample_color = torch.tensor((0.18, 0.45, 1.0), dtype=torch.float32, device=keep_mask.device).view(1, 3, 1, 1)
    return torch.where(sample_sites.expand_as(base), sample_color, base)


def _plot_tensor(ax, img: torch.Tensor) -> None:
    if img.shape[0] == 1:
        ax.imshow(img[0].detach().cpu(), cmap="gray", interpolation="nearest", vmin=0.0, vmax=1.0)
    else:
        ax.imshow(img.movedim(0, -1).detach().cpu().clamp(0.0, 1.0), interpolation="nearest", vmin=0.0, vmax=1.0)

    height, width = img.shape[-2:]
    if max(height, width) <= 32:
        ax.set_xticks([x - 0.5 for x in range(width + 1)], minor=True)
        ax.set_yticks([y - 0.5 for y in range(height + 1)], minor=True)
        ax.grid(which="minor", color="black", linewidth=0.3, alpha=0.25)
    ax.tick_params(which="both", bottom=False, left=False, labelbottom=False, labelleft=False)


def _render_hierarchy_figure(
    *,
    level_masks: list[torch.BoolTensor],
    sample_site_masks: list[torch.BoolTensor],
    masking_strategy: str,
    patch_size: int,
    stride: int,
    save_path: str | Path | None,
    show: bool,
) -> None:
    import matplotlib.pyplot as plt

    num_cols = len(level_masks)
    fig, axes = plt.subplots(
        2,
        num_cols,
        squeeze=False,
        figsize=(3.1 * num_cols, 6.1),
    )

    for level_idx, keep_mask in enumerate(level_masks):
        keep, masked, total = _mask_counts(keep_mask)
        height, width = keep_mask.shape[-2:]

        mask_ax = axes[0][level_idx]
        _plot_tensor(mask_ax, _mask_rgb(keep_mask)[0])
        mask_ax.set_title(
            f"L{level_idx}\n{height}x{width} | masked {masked}/{total}",
            fontsize=10,
        )
        if level_idx == 0:
            mask_ax.set_ylabel("Mask", fontsize=10)

        sample_ax = axes[1][level_idx]
        if level_idx < len(sample_site_masks):
            next_height, next_width = level_masks[level_idx + 1].shape[-2:]
            _plot_tensor(sample_ax, _sample_overlay(keep_mask, sample_site_masks[level_idx])[0])
            sample_ax.set_title(
                f"sample -> {next_height}x{next_width}\nchild[y, x] = parent[{stride}y, {stride}x]",
                fontsize=9,
            )
        else:
            _plot_tensor(sample_ax, _mask_rgb(keep_mask)[0])
            sample_ax.set_title("final level", fontsize=10)
        if level_idx == 0:
            sample_ax.set_ylabel("Parent Sites", fontsize=10)

    fig.suptitle(
        f"downsample_center_mask hierarchy | strategy={masking_strategy} | patch_size={patch_size} | stride={stride}",
        fontsize=12,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=180, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def run_hierarchical_mask_demo(args: argparse.Namespace) -> tuple[list[torch.BoolTensor], list[torch.BoolTensor]]:
    image_size = _parse_image_size_arg(args.image_size)
    torch.manual_seed(args.seed)

    keep_mask = _sample_initial_keep_mask(
        image_size=image_size,
        patch_size=args.patch_size,
        mask_ratio=args.mask_ratio,
        masking_strategy=args.masking_strategy,
        multi_block_scale_min=args.multi_block_scale_min,
        multi_block_scale_max=args.multi_block_scale_max,
        multi_block_aspect_ratio_min=args.multi_block_aspect_ratio_min,
        multi_block_aspect_ratio_max=args.multi_block_aspect_ratio_max,
        multi_block_square_aspect_ratio=args.multi_block_square_aspect_ratio,
    )
    level_masks, sample_site_masks = _build_mask_hierarchy(
        keep_mask,
        num_layers=args.num_layers,
        stride=args.stride,
    )

    print(
        f"start={image_size[0]}x{image_size[1]} | layers={args.num_layers} | stride={args.stride} | "
        f"patch_size={args.patch_size} | mask_ratio={args.mask_ratio:.3f} | strategy={args.masking_strategy}"
    )
    for level_idx, level_mask in enumerate(level_masks):
        keep, masked, total = _mask_counts(level_mask)
        height, width = level_mask.shape[-2:]
        print(
            f"level {level_idx}: {height}x{width} | keep={keep}/{total} | "
            f"masked={masked}/{total} | masked_ratio={masked / total:.3f}"
        )

    _render_hierarchy_figure(
        level_masks=level_masks,
        sample_site_masks=sample_site_masks,
        masking_strategy=args.masking_strategy,
        patch_size=args.patch_size,
        stride=args.stride,
        save_path=args.save_path,
        show=args.show,
    )
    return level_masks, sample_site_masks


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--image_size",
        nargs="+",
        default=[28],
        type=int,
        help="Starting keep-mask resolution. Pass one value for square inputs or two values for height width.",
    )
    parser.add_argument(
        "--num_layers",
        default=3,
        type=int,
        help="Number of times to apply downsample_center_mask.",
    )
    parser.add_argument(
        "--stride",
        default=2,
        type=int,
        help="Stride passed to downsample_center_mask. stride=2 matches the current sparse UNet downsampling path.",
    )
    parser.add_argument("--seed", default=0, type=int, help="Random seed used for mask sampling.")
    parser.add_argument("--save_path", default=None, type=str, help="Optional PNG path for saving the rendered figure.")
    parser.add_argument("--show", dest="show", action="store_true", help="Display the rendered figure.")
    parser.add_argument("--no_show", dest="show", action="store_false", help="Skip interactive figure display.")
    parser.set_defaults(show=True)
    add_masking_arguments(parser)
    return parser


def test_downsample_center_mask_hierarchy_matches_stride_slice():
    keep_mask = torch.tensor(
        [[
            [[
                True, False, True, False, True, False,
            ], [
                False, True, False, True, False, True,
            ], [
                True, True, False, False, True, True,
            ], [
                False, False, True, True, False, False,
            ], [
                True, False, False, True, True, False,
            ]]
        ]],
        dtype=torch.bool,
    )

    level_masks, sample_site_masks = _build_mask_hierarchy(keep_mask, num_layers=1, stride=2)

    expected_next = keep_mask[:, :, ::2, ::2][:, :, :3, :3]
    expected_sites = torch.zeros_like(keep_mask)
    expected_sites[:, :, ::2, ::2] = True

    assert torch.equal(level_masks[1], expected_next)
    assert torch.equal(sample_site_masks[0], expected_sites)


def test_hierarchical_masking_visualization_smoke(tmp_path):
    pytest = __import__("pytest")
    pytest.importorskip("matplotlib")

    out_path = tmp_path / "hierarchical_masking.png"
    args = _build_arg_parser().parse_args(
        [
            "--image_size", "28",
            "--num_layers", "3",
            "--mask_ratio", "0.6",
            "--masking_strategy", "random",
            "--patch_size", "1",
            "--save_path", str(out_path),
            "--no_show",
        ]
    )
    level_masks, sample_site_masks = run_hierarchical_mask_demo(args)

    assert len(level_masks) == 4
    assert len(sample_site_masks) == 3
    assert level_masks[0].shape[-2:] == (28, 28)
    assert level_masks[1].shape[-2:] == (14, 14)
    assert level_masks[2].shape[-2:] == (7, 7)
    assert level_masks[3].shape[-2:] == (4, 4)
    assert out_path.exists() and out_path.stat().st_size > 0


if __name__ == "__main__":
    parsed_args = _build_arg_parser().parse_args()
    run_hierarchical_mask_demo(parsed_args)
