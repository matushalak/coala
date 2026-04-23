import argparse
from pathlib import Path

import torch

from coala import DATADIR
from coala.datasets.msmnist import msmnist


def _parse_masked_fill(value: str) -> str | float:
    if value == "random":
        return value
    return float(value)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data_dir", type=str, default=str(DATADIR))
    parser.add_argument("--download", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--patch_size", type=int, default=4)
    parser.add_argument("--mask_ratio", type=float, default=0.5)
    parser.add_argument("--masked_fill", type=str, default="0.0")
    parser.add_argument("--noise_sigma", type=float, default=1.0)
    parser.add_argument("--number_of_masks", type=int, default=4)
    parser.add_argument("--timesteps_per_mask", type=int, default=1)
    parser.add_argument("--num_examples", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output_path",
        type=str,
        default="tests/msmnist_contrastive_gallery.png",
    )
    parser.add_argument("--no_show", action="store_true")
    return parser


def _imshow(ax, image: torch.Tensor) -> None:
    image = image.detach().cpu()
    ax.imshow(image[0], cmap="gray", interpolation="nearest", vmin=-1.0, vmax=1.0)
    ax.set_xticks([])
    ax.set_yticks([])


def main() -> None:
    args = build_argparser().parse_args()
    if args.no_show:
        import matplotlib

        matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    torch.manual_seed(args.seed)
    train_loader, _, _ = msmnist(
        root=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        download=args.download,
        patch_size=args.patch_size,
        mask_ratio=args.mask_ratio,
        mask_pattern="structured",
        masked_fill=_parse_masked_fill(args.masked_fill),
        noise_sigma=args.noise_sigma,
        number_of_masks=args.number_of_masks,
        timesteps_per_mask=args.timesteps_per_mask,
        target_type="both",
        contrastive=True,
    )

    masked_imgs, targets = next(iter(train_loader))
    pair_count = min(args.num_examples, masked_imgs.shape[0] // 2)
    n_steps = masked_imgs.shape[1]
    paired_inputs = masked_imgs.view(-1, 2, *masked_imgs.shape[1:])[:pair_count]
    paired_targets = targets["image"].view(-1, 2, *targets["image"].shape[1:])[:pair_count]
    labels = targets["label"].view(-1, 2, targets["label"].shape[-1])[:pair_count]

    fig, axes = plt.subplots(
        pair_count * 3,
        n_steps,
        squeeze=False,
        figsize=(1.8 * n_steps, 1.8 * pair_count * 3),
    )
    row_titles = ("primary", "inverse", "clean")
    for pair_idx in range(pair_count):
        sample_label = int(labels[pair_idx, 0, 0].item())
        for time_idx in range(n_steps):
            panels = (
                paired_inputs[pair_idx, 0, time_idx],
                paired_inputs[pair_idx, 1, time_idx],
                paired_targets[pair_idx, 0, time_idx],
            )
            for row_offset, panel in enumerate(panels):
                ax = axes[pair_idx * 3 + row_offset][time_idx]
                _imshow(ax, panel)
                if time_idx == 0:
                    ax.set_ylabel(f"{row_titles[row_offset]}\nlabel={sample_label}", rotation=0, labelpad=28)
                if pair_idx == 0:
                    ax.set_title(f"t={time_idx}")

    fig.suptitle(
        "MSMNIST contrastive structured-mask views\nrows: primary masked view, inverse masked view, clean target",
        fontsize=14,
    )
    fig.tight_layout()

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    print(f"Saved contrastive MSMNIST gallery to {output_path.resolve()}")
    print("Interleaved batch order: sample0_view0, sample0_view1, sample1_view0, sample1_view1, ...")
    print(f"contrastive_positive_index: {targets['contrastive_positive_index'][: 2 * pair_count].tolist()}")

    if not args.no_show:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
