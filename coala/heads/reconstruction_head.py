import argparse
import os

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning.callbacks import ModelCheckpoint
from torchvision.utils import make_grid, save_image

from coala import DATADIR
from coala.datasets import get_dataloaders
from coala import Head_logs, LeJEPA_logs, dataset_lightning_logs_dir, dataset_log_dir
from coala.masking import add_masking_arguments, clear_mask_bank_caches, masking_kwargs_from_args, sample_keep_mask
from coala.architecture.sparse_cnn_unet import SparseCNNUNet


DEFAULT_CHECKPOINT_PATH = os.path.join(
    dataset_lightning_logs_dir(LeJEPA_logs, "mnist"),
    "version_11",
    "checkpoints",
    "epoch=50-step=21522.ckpt",
)
DEFAULT_LOG_DIR = os.path.join(Head_logs, "reconstruction")


def load_pretrained_unet(
    checkpoint_path: str,
    *,
    map_location: str | torch.device = "cpu",
) -> tuple[SparseCNNUNet, dict]:
    checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    hparams = dict(checkpoint.get("hyper_parameters", {}))

    model = SparseCNNUNet(
        num_input_channels=int(hparams.get("num_input_channels", 1)),
        num_output_channels=int(hparams.get("num_input_channels", 1)),
        num_filters=int(hparams.get("num_filters", 32)),
        decoder_densify_mode=str(hparams.get("decoder_densify_mode", "random")),
        use_skip=bool(hparams.get("use_skip", False)),
        upconv_method=str(hparams.get("upconv_method", "upsample+conv")),
        norm_type=str(hparams.get("norm_type", "layernorm")),
    )

    state_dict = checkpoint.get("state_dict", checkpoint)
    if any(key.startswith("model.") for key in state_dict):
        state_dict = {key[len("model."):]: value for key, value in state_dict.items() if key.startswith("model.")}
    model.load_state_dict(state_dict, strict=False)
    return model, hparams


class ReconstructionHead(pl.LightningModule):
    def __init__(
        self,
        checkpoint_path: str,
        lr: float = 1e-3,
        mask_ratio: float = 0.6,
        patch_size: int = 4,
        masked_loss_weight: float = 4.0,
        denoise: bool = False,
        denoise_sigma: float = 1.0,
        masking_strategy: str = "random",
        multi_block_scale_min: float = 0.15,
        multi_block_scale_max: float = 0.2,
        multi_block_aspect_ratio_min: float = 0.75,
        multi_block_aspect_ratio_max: float = 1.5,
        multi_block_square_aspect_ratio: float = 1.0,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model, self.pretrained_hparams = load_pretrained_unet(checkpoint_path)

        self.model.requires_grad_(False)
        self.model.decoder.up28_to_out.requires_grad_(True)

    def _mask(self, imgs: torch.Tensor) -> torch.BoolTensor:
        return sample_keep_mask(
            imgs,
            patch_size=self.hparams.patch_size,
            mask_ratio=self.hparams.mask_ratio,
            masking_strategy=self.hparams.masking_strategy,
            multi_block_scale_min=self.hparams.multi_block_scale_min,
            multi_block_scale_max=self.hparams.multi_block_scale_max,
            multi_block_aspect_ratio_min=self.hparams.multi_block_aspect_ratio_min,
            multi_block_aspect_ratio_max=self.hparams.multi_block_aspect_ratio_max,
            multi_block_square_aspect_ratio=self.hparams.multi_block_square_aspect_ratio,
        )

    def on_train_epoch_start(self) -> None:
        if self.hparams.masking_strategy in {"multi-block", "mixed"}:
            clear_mask_bank_caches()

    def reconstruct(
        self,
        imgs: torch.Tensor,
        keep_mask: torch.BoolTensor | None = None,
    ) -> tuple[torch.Tensor, torch.BoolTensor]:
        if keep_mask is None:
            keep_mask = self._mask(imgs)
        model_input = imgs
        if self.hparams.denoise:
            noise = self.hparams.denoise_sigma * torch.randn_like(imgs)
            model_input = torch.where(keep_mask, imgs + noise, noise).clamp_(-1.0, 1.0)
        recon = self.model(model_input, keep_mask=keep_mask)
        return recon, keep_mask

    def _reconstruction_loss(self, recon: torch.Tensor, imgs: torch.Tensor, keep_mask: torch.BoolTensor) -> torch.Tensor:
        per_pixel_mse = (recon - imgs).pow(2).mean(dim=1)
        masked_pixels = (~keep_mask.squeeze(1)).to(dtype=per_pixel_mse.dtype)
        weights = torch.ones_like(per_pixel_mse) + masked_pixels * (self.hparams.masked_loss_weight - 1.0)
        weighted = per_pixel_mse * weights
        per_image = weighted.sum(dim=(1, 2)) / weights.sum(dim=(1, 2)).clamp_min(1e-8)
        return per_image.mean()

    def forward(
        self,
        imgs: torch.Tensor,
        keep_mask: torch.BoolTensor | None = None,
    ) -> tuple[torch.Tensor, torch.BoolTensor]:
        return self.reconstruct(imgs, keep_mask=keep_mask)

    def _shared_step(self, batch, stage: str) -> torch.Tensor:
        imgs = batch[0]
        recon, keep_mask = self.reconstruct(imgs)
        loss = self._reconstruction_loss(recon, imgs, keep_mask)
        self.log(f"{stage}_reconstruction_loss", loss, on_step=False, on_epoch=True, prog_bar=(stage != "train"))
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        self._shared_step(batch, "test")

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.decoder.up28_to_out.parameters(), lr=self.hparams.lr)


class ReconstructionPlotCallback(pl.Callback):
    def __init__(self, every_n_epochs: int = 1, num_images: int = 16, save_to_disk: bool = True):
        super().__init__()
        self.every_n_epochs = every_n_epochs
        self.num_images = num_images
        self.save_to_disk = save_to_disk
        self._example_batch = None

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        if self._example_batch is None and batch_idx == 0:
            self._example_batch = batch[0][:self.num_images].detach().cpu()

    @torch.no_grad()
    def on_validation_epoch_end(self, trainer, pl_module):
        if self._example_batch is None:
            return
        if trainer.current_epoch % self.every_n_epochs != 0:
            return

        imgs = self._example_batch.to(pl_module.device)
        recon, keep_mask = pl_module.reconstruct(imgs)
        grey = 0.0
        visible_noise = 0.0 if not pl_module.hparams.denoise else (pl_module.hparams.denoise_sigma * torch.randn_like(imgs)).clamp_(-1.0, 1.0)
        masked = torch.where(keep_mask, imgs + visible_noise, grey).float()
        panel = torch.cat([imgs.float(), masked, recon.float()], dim=0).detach().cpu()
        grid = make_grid(
            panel,
            nrow=min(self.num_images, imgs.shape[0]),
            normalize=True,
            pad_value=grey,
            value_range=(-1.0, 1.0),
        )

        epoch = trainer.current_epoch
        is_sanity = trainer.sanity_checking
        log_step = epoch + (0 if is_sanity else 1)
        logger_experiment = None if trainer.logger is None else getattr(trainer.logger, "experiment", None)
        if logger_experiment is not None and hasattr(logger_experiment, "add_image"):
            trainer.logger.experiment.add_image("reconstruction_head/original_masked_recon", grid, global_step=log_step)

        if self.save_to_disk:
            save_dir = trainer.default_root_dir
            if trainer.logger is not None and hasattr(trainer.logger, "log_dir"):
                save_dir = trainer.logger.log_dir
            os.makedirs(save_dir, exist_ok=True)
            image_name = f"epoch_{epoch}_recon.png" if not is_sanity else "pre_epoch_0_recon.png"
            save_image(grid, os.path.join(save_dir, image_name))


def train_reconstruction_head(args) -> list[dict[str, torch.Tensor]]:
    data_dir = os.path.abspath(args.data_dir)
    log_dir = dataset_log_dir(args.log_dir, args.dataset)
    os.makedirs(log_dir, exist_ok=True)
    train_loader, val_loader, test_loader = get_dataloaders(
        args.dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        root=data_dir,
    )

    model = ReconstructionHead(
        checkpoint_path=args.checkpoint_path,
        lr=args.lr,
        mask_ratio=args.mask_ratio,
        patch_size=args.patch_size,
        masked_loss_weight=args.masked_loss_weight,
        denoise=args.denoise,
        **masking_kwargs_from_args(args),
    )

    checkpoint_callback = ModelCheckpoint(
        save_weights_only=True,
        mode="min",
        monitor="val_reconstruction_loss",
    )
    recon_plot_callback = ReconstructionPlotCallback(
        every_n_epochs=args.plot_every_n_epochs,
        num_images=args.num_plot_images,
        save_to_disk=args.save_recon_plots,
    )
    trainer = pl.Trainer(
        default_root_dir=log_dir,
        accelerator="auto",
        max_epochs=args.epochs,
        callbacks=[checkpoint_callback, recon_plot_callback],
        enable_progress_bar=args.progress_bar,
    )
    if trainer.logger is not None and hasattr(trainer.logger, "_default_hp_metric"):
        trainer.logger._default_hp_metric = None

    pl.seed_everything(args.seed)
    trainer.fit(model, train_loader, val_loader)

    best_model = ReconstructionHead.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)
    return trainer.test(best_model, dataloaders=test_loader, verbose=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--checkpoint_path", default=DEFAULT_CHECKPOINT_PATH, type=str, help="Pretrained UNet/JEPA checkpoint.")
    parser.add_argument("--epochs", default=10, type=int, help="Max number of epochs.")
    parser.add_argument("--lr", default=1e-3, type=float, help="Learning rate.")
    parser.add_argument("--batch_size", default=128, type=int, help="Minibatch size.")
    parser.add_argument("--seed", default=42, type=int, help="Random seed.")
    parser.add_argument("--masked_loss_weight", default=4.0, type=float, help="Extra weight for masked pixels.")
    parser.add_argument("--denoise", action="store_true", help="Corrupt visible pixels with noise before reconstruction.")
    add_masking_arguments(parser)
    parser.add_argument("--dataset", default="mnist", type=str, help="Dataset name from coala.datasets registry.")
    parser.add_argument("--data_dir", default=DATADIR, type=str, help="Dataset directory.")
    parser.add_argument("--num_workers", default=4, type=int, help="Dataloader workers.")
    parser.add_argument("--log_dir", default=DEFAULT_LOG_DIR, type=str, help="Lightning log directory.")
    parser.add_argument("--num_plot_images", default=16, type=int, help="How many reconstructions to show per plot.")
    parser.add_argument("--plot_every_n_epochs", default=1, type=int, help="How often to write reconstruction plots.")
    parser.add_argument(
        "--save_recon_plots",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save reconstruction plots to the Lightning log dir.",
    )
    parser.add_argument("--progress_bar", action="store_true", help="Enable Lightning progress bar.")

    train_reconstruction_head(parser.parse_args())
