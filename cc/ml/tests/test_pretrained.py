from __future__ import annotations

import argparse
from pathlib import Path
import sys

# Allow direct script execution from inside this folder: `python test_pretrained.py`.
if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[3]))

from cc.ml import MAE_logs, FM_logs

def _load_checkpoint(torch, checkpoint_path: Path) -> dict[str, object]:
    return torch.load(checkpoint_path, map_location="cpu", weights_only=False)


def _load_checkpoint_hparams(checkpoint: dict[str, object]) -> dict[str, object]:
    return dict(checkpoint.get("hyper_parameters", {}))


def _compute_batch_metrics(torch, recon, imgs, keep_mask) -> dict[str, object]:
    per_pixel_mse = (recon - imgs).pow(2).mean(dim=1)
    visible_mask = keep_mask.squeeze(1)
    masked_mask = ~visible_mask
    visible_mask_f = visible_mask.to(dtype=per_pixel_mse.dtype)
    masked_mask_f = masked_mask.to(dtype=per_pixel_mse.dtype)
    return {
        "visible_sum": (per_pixel_mse * visible_mask_f).sum(),
        "visible_count": visible_mask_f.sum(),
        "masked_sum": (per_pixel_mse * masked_mask_f).sum(),
        "masked_count": masked_mask_f.sum(),
        "full_sum": per_pixel_mse.sum(),
        "full_count": torch.tensor(per_pixel_mse.numel(), device=per_pixel_mse.device, dtype=per_pixel_mse.dtype),
        "num_images": torch.tensor(imgs.shape[0], device=per_pixel_mse.device, dtype=per_pixel_mse.dtype),
    }


def _build_reconstruction_grid(torch, torchvision, imgs, masked_imgs, recon, num_images: int):
    grey = 0.0
    num_images = max(1, min(num_images, imgs.shape[0]))
    original = imgs[:num_images].float()
    masked = masked_imgs[:num_images].float()
    reconstructed = recon[:num_images].float()
    panel = torch.cat([original, masked, reconstructed], dim=0).detach().cpu()
    return torchvision.utils.make_grid(panel, nrow=num_images, normalize=False, pad_value=grey)


def _save_or_show_grid(grid, *, save_path: str | Path | None, show: bool) -> None:
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torchvision = __import__("torchvision")
        torchvision.utils.save_image(grid, str(save_path))
    if show:
        import matplotlib.pyplot as plt

        # grid = grid.detach().cpu().mul(255).add_(0.5).clamp_(0, 255).to(__import__("torch").uint8)
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.imshow(grid[0], cmap="gray", interpolation="nearest", vmin=-1, vmax=1)
        ax.set_title("Original / Masked / Reconstruction")
        ax.axis("off")
        fig.tight_layout()
        plt.show()


def evaluate_pretrained_checkpoint(
    checkpoint_path: str | Path,
    *,
    eval_split: str,
    batch_size: int,
    num_batches: int,
    give_mask: bool = True,
    seed: int = 2026,
    num_show_images: int = 8,
    mask_ratio: float | None = None,
    masked_fill:str|float|None = None,
    visible_corrupt: bool = False,
    patch_size: int | None = None,
    masked_loss_weight: float | None = None,
    decoder_densify_mode: str | None = None,
    upconv_method: str | None = None,
    use_skip: bool | None = None,
    num_filters: int | None = None,
    num_input_channels: int | None = None,
    norm_type: str | None = None,
) -> dict[str, object]:
    torch = __import__("torch")
    torchvision = __import__("torchvision")

    from cc import DATADIR
    from cc.datasets.mnist import mnist
    from cc.ml.MAEmodel import MAE

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing pretrained checkpoint: {checkpoint_path}")

    checkpoint = _load_checkpoint(torch, checkpoint_path)
    hparams = _load_checkpoint_hparams(checkpoint)
    mask_ratio = float(hparams["mask_ratio"] if mask_ratio is None else mask_ratio)
    patch_size = int(hparams["patch_size"] if patch_size is None else patch_size)
    masked_loss_weight = float(hparams["masked_loss_weight"] if masked_loss_weight is None else masked_loss_weight)
    decoder_densify_mode = str(hparams["decoder_densify_mode"] if decoder_densify_mode is None else decoder_densify_mode)
    upconv_method = str(hparams["upconv_method"] if upconv_method is None else upconv_method)
    use_skip = bool(hparams["use_skip"] if use_skip is None else use_skip)
    num_filters = int(hparams["num_filters"] if num_filters is None else num_filters)
    num_input_channels = int(hparams["num_input_channels"] if num_input_channels is None else num_input_channels)
    # Older MAE checkpoints predate this hparam and used LayerNorm throughout.
    norm_type = str(hparams.get("norm_type", "layernorm") if norm_type is None else norm_type)

    train_loader, val_loader, test_loader = mnist(
        root=DATADIR,
        batch_size=batch_size,
        num_workers=0,
        download=False,
    )
    dataloaders = {"train": train_loader, "val": val_loader, "test": test_loader}
    if eval_split not in dataloaders:
        raise ValueError(f"eval_split must be one of {tuple(dataloaders)}, got {eval_split!r}.")
    loader = dataloaders[eval_split]

    model = MAE(
        num_filters=num_filters,
        lr=float(hparams["lr"]),
        mask_ratio=mask_ratio,
        patch_size=patch_size,
        masked_loss_weight=masked_loss_weight,
        num_input_channels=num_input_channels,
        decoder_densify_mode=decoder_densify_mode,
        use_skip=use_skip,
        upconv_method=upconv_method,
        norm_type=norm_type,
    )
    model.load_state_dict(checkpoint.get("state_dict", checkpoint), strict=False)
    model.eval()
    param_ref = next(model.parameters())
    totals = None
    first_batch = None

    processed_batches = 0
    torch.manual_seed(seed)
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if num_batches > 0 and batch_idx >= num_batches:
                break
            processed_batches += 1
            imgs = batch[0].to(device=param_ref.device, dtype=param_ref.dtype)
            actual_keep_mask = model._mask(imgs)
            masked_imgs = imgs * actual_keep_mask.to(dtype=imgs.dtype)
            noise = torch.randn_like(imgs)
            if masked_fill is not None:
                if isinstance(masked_fill, str) and masked_fill == "random":
                    masked_imgs = torch.where(actual_keep_mask, 
                                              masked_imgs + noise if visible_corrupt else masked_imgs, 
                                              noise)
                else:
                    masked_imgs = torch.where(actual_keep_mask, 
                                              masked_imgs + noise if visible_corrupt else masked_imgs, 
                                              masked_fill.to(device=imgs.device, dtype=imgs.dtype))

            model_keep_mask = actual_keep_mask if give_mask else torch.ones_like(actual_keep_mask)
            recon, _ = model.reconstruct(masked_imgs, keep_mask=model_keep_mask)
            batch_loss = model._reconstruction_loss(recon, imgs, actual_keep_mask)
            batch_metrics = _compute_batch_metrics(torch, recon, imgs, actual_keep_mask)
            batch_metrics["weighted_loss_sum"] = batch_loss.detach() * imgs.shape[0]
            if totals is None:
                totals = {key: value.detach().clone() for key, value in batch_metrics.items()}
                first_batch = (imgs, actual_keep_mask, recon)
            else:
                for key, value in batch_metrics.items():
                    totals[key] = totals[key] + value.detach()

    if totals is None or first_batch is None:
        raise RuntimeError("No evaluation batches were processed.")

    imgs0, keep_mask0, recon0 = first_batch
    metrics = {
        "visible_mse": totals["visible_sum"] / totals["visible_count"].clamp_min(1.0),
        "masked_mse": totals["masked_sum"] / totals["masked_count"].clamp_min(1.0),
        "full_mse": totals["full_sum"] / totals["full_count"].clamp_min(1.0),
        "weighted_loss": totals["weighted_loss_sum"] / totals["num_images"].clamp_min(1.0),
    }
    return {
        "metrics": metrics,
        "masked_loss_weight": masked_loss_weight,
        "mask_ratio": mask_ratio,
        "patch_size": patch_size,
        "give_mask": give_mask,
        "eval_split": eval_split,
        "num_batches": processed_batches,
        "norm_type": norm_type,
        "grid": _build_reconstruction_grid(
            torch,
            torchvision,
            imgs0,
            masked_imgs,
            recon0,
            num_images=num_show_images,
        ),
        "first_imgs": imgs0,
        "first_keep_mask": keep_mask0,
        "first_recon": recon0,
        "torch": torch,
    }



# Command-line interface
DEFAULT_CHECKPOINT_PATH = (
    # MAE models
    # f"{MAE_logs}/lightning_logs/version_13/checkpoints/epoch=19-step=8440.ckpt"
    # f"{MAE_logs}/lightning_logs/version_18/checkpoints/epoch=19-step=8440.ckpt" # improved MAE
    
    # dMAE models
    # f"{MAE_logs}/lightning_logs/version_26/checkpoints/epoch=49-step=21100.ckpt" # denoising MAE
    f"{MAE_logs}/lightning_logs/version_21/checkpoints/epoch=19-step=8440.ckpt" # denoising MAE, more noise
    # f"{MAE_logs}/lightning_logs/version_27/checkpoints/epoch=50-step=21522.ckpt" # "FM"-type noise dMAE pretraining, also good
    
    # FM models
    # f"{FM_logs}/lightning_logs/version_1/checkpoints/epoch=18-step=16036.ckpt" # FM model, pretrained for 21 epochs
)
DEFAULT_SAVE_PATH = Path(__file__).with_name("pretrained_reconstructions.png")

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Core test params
    parser.add_argument("--checkpoint-path", default=str(DEFAULT_CHECKPOINT_PATH), type=str)
    parser.add_argument("--eval-split", default="test", choices=("train", "val", "test"), type=str)
    parser.add_argument("--batch-size", default=128, type=int)
    parser.add_argument("--num-batches", default=0, type=int, help="0 means evaluate the full split.")
    
    # Masking params
    parser.add_argument("--mask-ratio", default=None, type=float)
    parser.add_argument("--masked_fill", default="random", type=str, help="Masked pixel fill value or 'random'.")
    parser.add_argument("--visible_corrupt", action='store_true', help="Whether to corrupt visible pixels.")
    parser.add_argument("--give-mask", dest="give_mask", action="store_true")
    parser.add_argument("--no-give-mask", dest="give_mask", action="store_false")
    parser.set_defaults(give_mask=True)
    parser.add_argument("--patch-size", default=None, type=int)
    
    # Other test and model params
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--num-show-images", default=16, type=int)
    parser.add_argument("--save-path", default=str(DEFAULT_SAVE_PATH), type=str)
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--masked-loss-weight", default=None, type=float)
    parser.add_argument("--num-filters", default=None, type=int)
    parser.add_argument("--num-input-channels", default=None, type=int)
    parser.add_argument("--decoder-densify-mode", default=None, choices=("random", "token", "zero"),type=str,)
    parser.add_argument("--upconv-method",default=None,choices=("transposed_conv", "upsample+conv"),type=str,)
    parser.add_argument("--norm-type",default=None,choices=("layernorm", "rmsnorm"),type=str,)
    parser.add_argument("--use-skip", dest="use_skip", action="store_true", default=None)
    parser.add_argument("--no-skip", dest="use_skip", action="store_false")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    results = evaluate_pretrained_checkpoint(
        checkpoint_path=args.checkpoint_path,
        eval_split=args.eval_split,
        batch_size=args.batch_size,
        num_batches=args.num_batches,
        give_mask=args.give_mask,
        seed=args.seed,
        num_show_images=args.num_show_images,
        mask_ratio=args.mask_ratio,
        patch_size=args.patch_size,
        masked_fill=args.masked_fill,
        visible_corrupt=args.visible_corrupt,
        masked_loss_weight=args.masked_loss_weight,
        decoder_densify_mode=args.decoder_densify_mode,
        upconv_method=args.upconv_method,
        use_skip=args.use_skip,
        num_filters=args.num_filters,
        num_input_channels=args.num_input_channels,
        norm_type=args.norm_type,
    )
    _save_or_show_grid(results["grid"], save_path=args.save_path, show=args.show)
    print(f"checkpoint: {args.checkpoint_path}")
    print(f"eval_split: {results['eval_split']}")
    print(f"batch_size: {args.batch_size}")
    print(f"num_batches: {results['num_batches']}")
    print(f"give_mask: {results['give_mask']}")
    print(f"mask_ratio: {results['mask_ratio']}")
    print(f"patch_size: {results['patch_size']}")
    print(f"masked_loss_weight: {results['masked_loss_weight']}")
    print(f"norm_type: {results['norm_type']}")
    print(f"reconstructions_saved_to: {args.save_path}")
    print(f"visible_mse: {results['metrics']['visible_mse'].item():.6f}")
    print(f"masked_mse: {results['metrics']['masked_mse'].item():.6f}")
    print(f"full_mse: {results['metrics']['full_mse'].item():.6f}")
    print(f"weighted_loss: {results['metrics']['weighted_loss'].item():.6f}")
