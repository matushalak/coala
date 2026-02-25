import argparse
from pathlib import Path

import torch
from torchvision.utils import save_image

from mnist import combine_grayscale_levels_mnist, mnist
from train_pl import VAE
from utils import visualize_reconstructions


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_LOGS_ROOT = SCRIPT_DIR / "VAE_logs" / "lightning_logs"

def resolve_checkpoint_path(logs_root, run_subdir, checkpoint_name=None):
    ckpt_dir = Path(logs_root) / run_subdir / "checkpoints"
    if checkpoint_name is not None:
        ckpt_path = ckpt_dir / checkpoint_name
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        return ckpt_path

    candidates = sorted(ckpt_dir.glob("*.ckpt"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(f"No checkpoint found in: {ckpt_dir}")
    return candidates[0]


def _grayscale_kwargs(n_grayscale_levels, level_digits, default_digits=None):
    kwargs = {}
    if default_digits is None:
        default_digits = list(range(10))
    for level in range(n_grayscale_levels):
        kwargs[f"level_{level}_digits"] = level_digits.get(level, default_digits)
    return kwargs


def _parse_level_digits_spec(spec, n_grayscale_levels):
    if ":" not in spec:
        raise ValueError(f"Invalid --level_digits '{spec}'. Use format LEVEL:D1,D2,...")

    level_str, digits_str = spec.split(":", 1)
    level = int(level_str)
    if level < 0 or level >= n_grayscale_levels:
        raise ValueError(
            f"Level {level} out of range for n_grayscale_levels={n_grayscale_levels}. "
            f"Valid levels: 0..{n_grayscale_levels - 1}"
        )

    digits_str = digits_str.strip()
    if digits_str.lower() in {"all", "*"}:
        digits = list(range(10))
    else:
        digits = sorted({int(d) for d in digits_str.split(",") if d.strip() != ""})
        if not digits:
            raise ValueError(f"No digits provided in --level_digits '{spec}'")
        if any(d < 0 or d > 9 for d in digits):
            raise ValueError(f"Digits must be in 0..9 in --level_digits '{spec}'")

    name = f"level_{level}_digits_" + "_".join(str(d) for d in digits)
    return name, level, digits


def build_custom_loaders(base_test_loader, n_grayscale_levels, level_specs):
    loaders = {}
    if not level_specs:
        level_specs = ["0:all"]

    for spec in level_specs:
        loader_name, level, digits = _parse_level_digits_spec(spec, n_grayscale_levels)
        # Build one loader per LEVEL:DIGITS spec:
        # selected level keeps requested digits, all other levels are empty.
        level_digits = {level: digits}
        kwargs = _grayscale_kwargs(
            n_grayscale_levels=n_grayscale_levels,
            level_digits=level_digits,
            default_digits=[],
        )
        loaders[loader_name] = combine_grayscale_levels_mnist(
            base_test_loader,
            n_grayscale_levels=n_grayscale_levels,
            batch_size=base_test_loader.batch_size,
            shuffle=True,
            num_workers=base_test_loader.num_workers,
            drop_last=False,
            **kwargs,
        )
    return loaders


def main(args):
    checkpoint_path = resolve_checkpoint_path(args.logs_root, args.run_subdir, args.checkpoint_name)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model = VAE.load_from_checkpoint(str(checkpoint_path), map_location=device)
    model.to(device)
    model.eval()

    _, _, base_test_loader = mnist(
        root=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        download=args.download,
    )
    custom_loaders = build_custom_loaders(
        base_test_loader=base_test_loader,
        n_grayscale_levels=args.n_grayscale_levels,
        level_specs=args.level_digits,
    )

    output_dir = Path(args.output_dir) if args.output_dir else checkpoint_path.parent.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    for loader_name, loader in custom_loaders.items():
        grid = visualize_reconstructions(model, loader, n_images=args.n_images)
        out_path = output_dir / f"recon_{loader_name}.png"
        save_image(grid, str(out_path), normalize=False)
        print(f"Saved: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--run_subdir", type=str, required=True, help="Sub-folder in VAE_logs/lightning_logs.")
    parser.add_argument(
        "--checkpoint_name",
        type=str,
        default=None,
        help="Checkpoint filename in <run_subdir>/checkpoints. If omitted, newest .ckpt is used.",
    )
    parser.add_argument("--logs_root", type=str, default=str(DEFAULT_LOGS_ROOT), help="Path to VAE logs root.")
    parser.add_argument("--data_dir", type=str, default="../data/", help="MNIST data root.")
    parser.add_argument("--batch_size", type=int, default=128, help="Data loader batch size.")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of dataloader workers.")
    parser.add_argument("--n_images", type=int, default=8, help="Number of reconstructions to visualize.")
    parser.add_argument(
        "--n_grayscale_levels",
        type=int,
        default=6,
        help="Number of grayscale levels in combined_grayscale_levels_mnist.",
    )
    parser.add_argument(
        "--level_digits",
        nargs="*",
        default=[],
        help=(
            "One or more LEVEL:D1,D2,... specs. "
            "Each spec creates one test loader for that level only "
            "(all other levels are empty). "
            "Example: --level_digits 5:1,2,3 2:0,9"
        ),
    )
    parser.add_argument("--download", action="store_true", help="Download MNIST if missing.")
    parser.add_argument("--cpu", action="store_true", help="Force CPU.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Where to save reconstruction images. Default: same run folder as checkpoint.",
    )
    main(parser.parse_args())
