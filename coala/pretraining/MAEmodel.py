import argparse
import os

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from torchvision.utils import make_grid, save_image

from coala import DATADIR
from coala.datasets import get_dataloaders
from coala import MAE_logs, dataset_log_dir
from coala.masking import add_masking_arguments, clear_mask_bank_caches, masking_kwargs_from_args, sample_keep_mask
from coala.pretraining.common import (
    configure_adamw_with_warmup_and_cosine_decay,
    GenerativeHead,
    default_model_config,
    default_reconstruction_head_config,
    feature_names_from_latents,
    instantiate_autoencoder,
    normalize_model_config,
    normalize_reconstruction_head_config,
)


class MAE(pl.LightningModule):
    def __init__(
        self,
        num_filters: int,
        lr: float,
        mask_ratio: float,
        patch_size: int,
        masked_loss_weight: float,
        batch_size: int = 128,
        num_input_channels: int = 1,
        image_size: int = 28,
        decoder_densify_mode: str = "random",
        use_skip: bool = True,
        upconv_method: str = "upsample+conv",
        norm_type: str = "rmsnorm",
        denoise: bool = False,
        denoise_sigma: float = 1.0,
        warmup_epochs: int = 0,
        weight_decay: float = 0.0,
        reconstruction_head_family: str = "ViT",
        model_config: dict | None = None,
        reconstruction_head_config: dict | None = None,
        masking_strategy: str = "random",
        multi_block_scale_min: float = 0.15,
        multi_block_scale_max: float = 0.2,
        multi_block_aspect_ratio_min: float = 0.75,
        multi_block_aspect_ratio_max: float = 1.5,
        multi_block_square_aspect_ratio: float = 1.0,
    ):
        super().__init__()
        if model_config is None:
            model_config = default_model_config(
                image_size=image_size,
                num_input_channels=num_input_channels,
                num_filters=num_filters,
                norm_type=norm_type,
                decoder_densify_mode=decoder_densify_mode,
                use_skip=use_skip,
                upconv_method=upconv_method,
            )
        model_config = normalize_model_config(model_config)
        assert model_config["input_shape"][0] == num_input_channels

        self.model = instantiate_autoencoder(model_config, predictive=False)
        self.feature_names = list(self.model.encoder.feature_names)
        first_feature = self.feature_names[0]
        if reconstruction_head_config is None:
            reconstruction_head_config = default_reconstruction_head_config(
                family=reconstruction_head_family,
                input_shape=self.model.encoder.spatial_shapes[0],
                output_shape=self.model.input_shape[1:],
                feature_dim=self.model.encoder.feature_dims[0],
                num_output_channels=self.model.input_shape[0],
            )
        reconstruction_head_config = normalize_reconstruction_head_config(reconstruction_head_config)
        self.reconstruction_head = GenerativeHead(
            family=reconstruction_head_config["family"],
            in_channels=self.model.encoder.feature_dims[0],
            input_spatial_shape=self.model.encoder.spatial_shapes[0],
            output_spatial_shape=self.model.input_shape[1:],
            num_output_channels=reconstruction_head_config["num_output_channels"],
            kwargs=reconstruction_head_config["kwargs"],
        )
        self.reconstruction_feature = first_feature
        self.save_hyperparameters()

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
        decoder_latents, _ = self.model(model_input, keep_mask=keep_mask)
        recon = self.reconstruction_head(decoder_latents[self.reconstruction_feature])
        return recon, keep_mask

    def _reconstruction_loss(self, recon: torch.Tensor, imgs: torch.Tensor, keep_mask: torch.BoolTensor) -> torch.Tensor:
        per_pixel_mse = (recon - imgs).pow(2).mean(dim=1)
        masked_pixels = (~keep_mask.squeeze(1)).to(dtype=per_pixel_mse.dtype)
        weights = torch.ones_like(per_pixel_mse) + masked_pixels * (self.hparams.masked_loss_weight - 1.0)
        weighted = per_pixel_mse * weights
        per_image = weighted.sum(dim=(1, 2)) / weights.sum(dim=(1, 2)).clamp_min(1e-8)
        return per_image.mean()

    def forward(self, imgs: torch.Tensor) -> torch.Tensor:
        recon, keep_mask = self.reconstruct(imgs)
        return self._reconstruction_loss(recon, imgs, keep_mask)

    def configure_optimizers(self):
        return configure_adamw_with_warmup_and_cosine_decay(self)

    def training_step(self, batch, batch_idx):
        loss = self.forward(batch[0])
        self.log("train_reconstruction_loss", loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.forward(batch[0])
        self.log("val_reconstruction_loss", loss, on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx):
        loss = self.forward(batch[0])
        self.log("test_reconstruction_loss", loss, on_step=False, on_epoch=True)


class ReconstructionCallback(pl.Callback):
    def __init__(self, every_n_epochs: int = 5, num_images: int = 20, save_to_disk: bool = False):
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
        if self._example_batch is None or trainer.current_epoch % self.every_n_epochs != 0:
            return

        imgs = self._example_batch.to(pl_module.device)
        recon, keep_mask = pl_module.reconstruct(imgs)
        grey = 0.0
        noise = 0.0 if not pl_module.hparams.denoise else (pl_module.hparams.denoise_sigma * torch.randn_like(imgs)).clamp_(-1.0, 1.0)
        masked = torch.where(keep_mask, imgs + noise, grey).float()
        panel = torch.cat([imgs.float(), masked, recon.float()], dim=0).detach().cpu()
        grid = make_grid(panel, nrow=self.num_images, normalize=True, pad_value=grey, value_range=(-1.0, 1.0))
        epoch = trainer.current_epoch
        is_sanity = trainer.sanity_checking
        log_step = epoch + (0 if is_sanity else 1)
        trainer.logger.experiment.add_image("MAE/original_masked_recon", grid, global_step=log_step)

        if self.save_to_disk:
            image_name = f"epoch_{epoch}_recon.png" if not is_sanity else "pre_epoch_0_recon.png"
            save_image(grid, os.path.join(trainer.logger.log_dir, image_name))


def train_mae(args):
    data_dir = os.path.abspath(args.data_dir)
    log_dir = dataset_log_dir(args.log_dir, args.dataset)
    os.makedirs(log_dir, exist_ok=True)
    train_loader, val_loader, test_loader = get_dataloaders(
        args.dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        root=data_dir,
    )

    recon_callback = ReconstructionCallback(save_to_disk=True)
    save_callback = ModelCheckpoint(save_weights_only=True, mode="min", monitor="val_reconstruction_loss")
    trainer = pl.Trainer(
        default_root_dir=log_dir,
        accelerator="auto",
        max_epochs=args.epochs,
        callbacks=[save_callback, recon_callback],
        enable_progress_bar=args.progress_bar,
    )
    trainer.logger._default_hp_metric = None

    pl.seed_everything(args.seed)
    model = MAE(
        num_filters=args.num_filters,
        lr=args.lr,
        batch_size=args.batch_size,
        mask_ratio=args.mask_ratio,
        patch_size=args.patch_size,
        masked_loss_weight=args.masked_loss_weight,
        num_input_channels=args.num_input_channels,
        image_size=args.image_size,
        decoder_densify_mode=args.decoder_densify_mode,
        use_skip=args.use_skip,
        upconv_method=args.upconv_method,
        norm_type=args.norm_type,
        denoise=args.denoise,
        warmup_epochs=args.warmup_epochs,
        weight_decay=args.weight_decay,
        reconstruction_head_family=args.reconstruction_head_family,
        **masking_kwargs_from_args(args),
    )

    trainer.fit(model, train_loader, val_loader)
    model = MAE.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)
    return trainer.test(model, dataloaders=test_loader, verbose=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--epochs", default=21, type=int)
    parser.add_argument("--lr", default=1.5e-3, type=float)
    parser.add_argument("--warmup_epochs", default=0, type=int)
    parser.add_argument("--weight_decay", default=0.0, type=float)
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--denoise", action="store_true")
    add_masking_arguments(parser)
    parser.add_argument("--masked_loss_weight", default=1.0, type=float)
    parser.add_argument("--num_filters", default=32, type=int)
    parser.add_argument("--num_input_channels", default=1, type=int)
    parser.add_argument("--image_size", default=28, type=int)
    parser.add_argument("--dataset", default="mnist", type=str)
    parser.add_argument("--decoder_densify_mode", default="random", choices=("random", "token", "zero"), type=str)
    parser.add_argument("--upconv_method", default="upsample+conv", choices=("transposed_conv", "upsample+conv"), type=str)
    parser.add_argument("--norm_type", default="rmsnorm", choices=("layernorm", "rmsnorm"), type=str)
    parser.add_argument("--use_skip", action="store_true")
    parser.add_argument("--reconstruction_head_family", default="ConvNet", choices=("ViT", "ConvNet", "ConvNeXt"), type=str)
    parser.add_argument("--data_dir", default=DATADIR, type=str)
    parser.add_argument("--num_workers", default=0, type=int)
    parser.add_argument("--log_dir", default=MAE_logs, type=str)
    parser.add_argument("--progress_bar", action="store_true")
    train_mae(parser.parse_args())
