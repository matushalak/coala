import argparse
import os
import re

import torch
import torch.nn.functional as F
from torchvision.utils import make_grid, save_image
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from cc.datasets.mnist import mnist
from cc.ml.masking import add_masking_arguments, clear_mask_bank_caches, masking_kwargs_from_args, sample_keep_mask
from cc.ml.architecture.sparse_cnn_unet import SparseCNNUNet
from cc.utils import ExponentialMovingAverage

class JEPA(pl.LightningModule):
    """
    Bare-bones CNN masked autoencoder for [-1, 1]-normalized images.
    """

    def __init__(
        self,
        num_filters: int,
        lr: float,
        mask_ratio: float,
        patch_size: int,
        masked_loss_weight: float,
        num_input_channels: int = 1,
        decoder_densify_mode: str = "random",
        use_skip: bool = True,
        upconv_method: str = "upsample+conv",
        norm_type: str = "rmsnorm",
        denoise: bool = False,
        reconstruction_loss: bool = False,
        teacher_ema_decay: float = 0.999,
        masking_strategy: str = "random",
        multi_block_scale_min: float = 0.15,
        multi_block_scale_max: float = 0.2,
        multi_block_aspect_ratio_min: float = 0.75,
        multi_block_aspect_ratio_max: float = 1.5,
        multi_block_square_aspect_ratio: float = 1.0,
    ):
        super().__init__()
        self.save_hyperparameters()
        # Student
        self.model = SparseCNNUNet(
            num_input_channels=num_input_channels,
            num_output_channels=num_input_channels,
            num_filters=num_filters,
            decoder_densify_mode=decoder_densify_mode,
            use_skip=use_skip,
            upconv_method=upconv_method,
            norm_type=norm_type,
        )
        
        # Teacher: Not part of student computation graph
        self.teacher = ExponentialMovingAverage(
            self.model.encoder,
            decay = teacher_ema_decay,
        ).eval()

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

    def _reconstruction_loss(self, recon: torch.Tensor, imgs: torch.Tensor, keep_mask: torch.BoolTensor) -> torch.Tensor:
        """
        MSE reconstruction loss for [-1, 1] targets.
        """
        per_pixel_mse = (recon - imgs).pow(2).mean(dim=1)
        masked_pixels = (~keep_mask.squeeze(1)).to(dtype=per_pixel_mse.dtype)
        weights = torch.ones_like(per_pixel_mse) + masked_pixels * (self.hparams.masked_loss_weight - 1.0)
        # unlike SparK and mainstream ViT MAEs implementations, 
        # we still include unmasked pixels in the loss (instead of ignoring them completely)
        # because we want to achieve good reconstructions across the whole image 
        # (at test time we don't know what is a mask)
        weighted = per_pixel_mse * weights
        per_image = weighted.sum(dim=(1, 2)) / weights.sum(dim=(1, 2)).clamp_min(1e-8)
        return per_image.mean()

    @staticmethod
    def _weighted_spatial_mean(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        num = (values * weights).flatten(1).sum(dim=1)
        den = weights.flatten(1).sum(dim=1).clamp_min(1e-8)
        return num / den

    @staticmethod
    def _masked_spatial_mean(values: torch.Tensor, mask: torch.BoolTensor) -> torch.Tensor:
        weights = mask.to(dtype=values.dtype)
        num = (values * weights).flatten(1).sum(dim=1)
        den = weights.flatten(1).sum(dim=1).clamp_min(1e-8)
        return num / den

    def _student_input(self, imgs: torch.Tensor, keep_mask: torch.BoolTensor) -> torch.Tensor:
        if not self.hparams.denoise:
            return imgs

        noise = torch.randn_like(imgs)
        # dMAE-style noise to corrupt images
        return torch.where(keep_mask, imgs + torch.randn_like(imgs), noise).clamp_(-1.0, 1.0)

    def _jepa_outputs(
        self,
        imgs: torch.Tensor,
        keep_mask: torch.BoolTensor | None = None,
    ) -> tuple[torch.Tensor, torch.BoolTensor, dict[str, torch.Tensor], torch.Tensor, dict[str, torch.Tensor], dict[str, torch.BoolTensor]]:
        if keep_mask is None:
            keep_mask = self._mask(imgs)

        teacher_latents = self.teacher(imgs, keep_mask=torch.ones_like(keep_mask, dtype=torch.bool))
        student_imgs = self._student_input(imgs, keep_mask)
        student_encoder_latents = self.model.encoder(student_imgs, keep_mask=keep_mask)
        recon, student_predictor_latents = self.model.decoder(student_encoder_latents)
        keep_masks = {k: v for k, v in student_encoder_latents.items() if "mask" in k}
        return student_imgs, keep_mask, teacher_latents, recon, student_predictor_latents, keep_masks

    def self_distillation_loss(
        self,
        predictor_latents: dict[str, torch.Tensor],
        teacher_latents: dict[str, torch.Tensor],
        keep_masks: dict[str, torch.BoolTensor],
        return_metrics: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        # (B, H, W) - average over channels
        diffs = {k: (predictor_latents[k] - teacher_latents[k]).pow(2).mean(dim=1)
                 for k in predictor_latents}

        per_scale = []
        metrics = {}
        # (B,) average over spatial dimensions
        for feat_name, diff in diffs.items():
            keep_mask = keep_masks[feat_name.replace("feat", "mask")].squeeze(1)
            weights = (
                torch.ones_like(diff)
                + (~keep_mask).to(dtype=diff.dtype) * (self.hparams.masked_loss_weight - 1.0)
            )
            scale_loss = self._weighted_spatial_mean(diff, weights)
            per_scale.append(scale_loss)

            if return_metrics:
                masked_loss = self._masked_spatial_mean(diff, ~keep_mask)
                visible_loss = self._masked_spatial_mean(diff, keep_mask)
                student_norm = predictor_latents[feat_name].float().pow(2).mean(dim=1).sqrt().mean()
                teacher_norm = teacher_latents[feat_name].float().pow(2).mean(dim=1).sqrt().mean()
                metrics[f"{feat_name}_loss"] = scale_loss.mean()
                metrics[f"{feat_name}_masked_loss"] = masked_loss.mean()
                metrics[f"{feat_name}_visible_loss"] = visible_loss.mean()
                metrics[f"{feat_name}_student_norm"] = student_norm
                metrics[f"{feat_name}_teacher_norm"] = teacher_norm

        # average equally over scales and batches
        loss = torch.stack(per_scale, dim=1).mean()
        if return_metrics:
            return loss, metrics
        return loss

    def forward(
        self,
        imgs: torch.Tensor,
        keep_mask: torch.BoolTensor | None = None,
        return_metrics: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        student_imgs, keep_mask, teacher_latents, recon, student_predictor_latents, keep_masks = self._jepa_outputs(
            imgs, keep_mask=keep_mask
        )

        if return_metrics:
            L_distill, metrics = self.self_distillation_loss(
                student_predictor_latents,
                teacher_latents,
                keep_masks,
                return_metrics=True,
            )
        else:
            L_distill = self.self_distillation_loss(student_predictor_latents, teacher_latents, keep_masks)
        L_recon = self._reconstruction_loss(recon, student_imgs, keep_mask)

        if return_metrics:
            return L_distill, L_recon, metrics
        return L_distill, L_recon
    
    @torch.no_grad()
    def _project_pair_to_rgb(
        self,
        teacher_latents: torch.Tensor,
        student_latents: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        b, c, h, w = teacher_latents.shape
        teacher_flat = teacher_latents.float().movedim(1, -1).reshape(-1, c)
        student_flat = student_latents.float().movedim(1, -1).reshape(-1, c)

        mean = teacher_flat.mean(dim=0, keepdim=True)
        teacher_centered = teacher_flat - mean
        student_centered = student_flat - mean
        cov = teacher_centered.T @ teacher_centered
        if cov.device.type == "mps":
            _, eigvecs = torch.linalg.eigh(cov.cpu())
            eigvecs = eigvecs.to(device=cov.device)
        else:
            _, eigvecs = torch.linalg.eigh(cov)

        num_components = min(3, c)
        principal_axes = eigvecs[:, -num_components:]
        teacher_rgb = teacher_centered @ principal_axes
        student_rgb = student_centered @ principal_axes
        if num_components < 3:
            pad = (0, 3 - num_components)
            teacher_rgb = F.pad(teacher_rgb, pad)
            student_rgb = F.pad(student_rgb, pad)

        teacher_rgb = teacher_rgb.view(b, h, w, 3).movedim(-1, 1)
        student_rgb = student_rgb.view(b, h, w, 3).movedim(-1, 1)
        rgb_flat = torch.cat([teacher_rgb.flatten(2), student_rgb.flatten(2)], dim=2)
        rgb_min = rgb_flat.amin(dim=2, keepdim=True)
        rgb_max = rgb_flat.amax(dim=2, keepdim=True)
        denom = (rgb_max - rgb_min).clamp_min(1e-6)
        teacher_rgb = teacher_rgb.flatten(2).sub(rgb_min).div(denom).view(b, 3, h, w).mul(2.0).sub(1.0)
        student_rgb = student_rgb.flatten(2).sub(rgb_min).div(denom).view(b, 3, h, w).mul(2.0).sub(1.0)
        return teacher_rgb, student_rgb

    @staticmethod
    def _expand_to_rgb(imgs: torch.Tensor) -> torch.Tensor:
        if imgs.shape[1] == 1:
            return imgs.expand(-1, 3, -1, -1)
        if imgs.shape[1] >= 3:
            return imgs[:, :3]
        raise ValueError(f"Unsupported image shape for RGB visualization: {tuple(imgs.shape)}")

    @staticmethod
    def _error_to_rgb(error_map: torch.Tensor) -> torch.Tensor:
        b, _, h, w = error_map.shape
        error_flat = error_map.float().flatten(2)
        error_min = error_flat.amin(dim=2, keepdim=True)
        error_max = error_flat.amax(dim=2, keepdim=True)
        error = error_flat.sub(error_min).div((error_max - error_min).clamp_min(1e-6)).view(b, 1, h, w)
        return torch.cat(
            [
                error.mul(2.0).sub(1.0),
                error.new_full((b, 1, h, w), -1.0),
                error.new_full((b, 1, h, w), -1.0),
            ],
            dim=1,
        )

    @staticmethod
    def _upsample_for_display(x: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
        if x.shape[-2:] == size:
            return x
        return F.interpolate(x, size=size, mode="nearest")

    @torch.no_grad()
    def feature_visualizations(
        self,
        imgs: torch.Tensor,
        keep_mask: torch.BoolTensor | None = None,
    ) -> dict[str, torch.Tensor]:
        student_imgs, keep_mask, teacher_latents, _, student_predictor_latents, _ = self._jepa_outputs(
            imgs, keep_mask=keep_mask
        )

        visualizations = {
            "context": torch.cat(
                [
                    self._expand_to_rgb(imgs.float()),
                    self._expand_to_rgb(torch.where(keep_mask, student_imgs, student_imgs.new_zeros((), dtype=student_imgs.dtype)).float()),
                ],
                dim=0,
            )
        }

        target_size = imgs.shape[-2:]
        for feat_name in ("feat28", "feat14", "feat7", "feat4"):
            teacher_rgb, student_rgb = self._project_pair_to_rgb(
                teacher_latents[feat_name],
                student_predictor_latents[feat_name],
            )
            error_rgb = self._error_to_rgb(
                (student_predictor_latents[feat_name] - teacher_latents[feat_name]).pow(2).mean(dim=1, keepdim=True)
            )
            visualizations[feat_name] = torch.cat(
                [
                    self._upsample_for_display(teacher_rgb, target_size),
                    self._upsample_for_display(student_rgb, target_size),
                    self._upsample_for_display(error_rgb, target_size),
                ],
                dim=0,
            )

        return visualizations
    
    def configure_optimizers(self):
        # only optimize student parameters
        return torch.optim.Adam(self.model.parameters(), lr=self.hparams.lr)

    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        self.teacher.update(self.model.encoder)

    def _log_distillation_metrics(self, prefix: str, metrics: dict[str, torch.Tensor], *, on_step: bool, on_epoch: bool):
        for name, value in metrics.items():
            self.log(f"{prefix}_{name}", value, on_step=on_step, on_epoch=on_epoch)

    def training_step(self, batch, batch_idx):
        loss_distil, loss_recon, metrics = self.forward(batch[0], return_metrics=True)
        self.log("train_reconstruction_loss", loss_recon, on_step=False, on_epoch=True)
        self.log("train_distillation_loss", loss_distil, on_step=False, on_epoch=True)
        self._log_distillation_metrics("train", metrics, on_step=False, on_epoch=True)
        return loss_distil + self.hparams.reconstruction_loss*loss_recon

    def validation_step(self, batch, batch_idx):
        loss_distil, loss_recon, metrics = self.forward(batch[0], return_metrics=True)
        self.log("val_reconstruction_loss", loss_recon, on_step=False, on_epoch=True)
        self.log("val_distillation_loss", loss_distil, on_step=False, on_epoch=True)
        self._log_distillation_metrics("val", metrics, on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx):
        loss_distil, loss_recon, metrics = self.forward(batch[0], return_metrics=True)
        self.log("test_reconstruction_loss", loss_recon, on_step=False, on_epoch=True)
        self.log("test_distillation_loss", loss_distil, on_step=False, on_epoch=True)
        self._log_distillation_metrics("test", metrics, on_step=False, on_epoch=True)


class ReconstructionCallback(pl.Callback):
    """
    Logs JEPA context plus multi-scale teacher / student / error visualizations.
    """

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
        if self._example_batch is None:
            return
        if trainer.current_epoch % self.every_n_epochs != 0:
            return

        imgs = self._example_batch.to(pl_module.device)
        grey = 0.0
        epoch = trainer.current_epoch
        is_sanity = trainer.sanity_checking
        log_step = epoch + (0 if is_sanity else 1)
        visualizations = pl_module.feature_visualizations(imgs)

        for name, panel in visualizations.items():
            grid = make_grid(
                panel.detach().cpu(),
                nrow=self.num_images,
                normalize=True,
                pad_value=grey,
                value_range=(-1.0, 1.0),
            )
            image_tag = "JEPA/context" if name == "context" else f"JEPA/{name}_teacher_student_error"
            trainer.logger.experiment.add_image(image_tag, grid, global_step=log_step)

            if self.save_to_disk:
                suffix = "context" if name == "context" else f"{name}_teacher_student_error"
                image_name = f"epoch_{epoch}_{suffix}.png" if not is_sanity else f"pre_epoch_0_{suffix}.png"
                save_image(grid, os.path.join(trainer.logger.log_dir, image_name))


def train_mae(args):
    os.makedirs(args.log_dir, exist_ok=True)
    train_loader, val_loader, test_loader = mnist(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        root=args.data_dir,
    )

    recon_callback = ReconstructionCallback(save_to_disk=True)
    save_callback = ModelCheckpoint(save_weights_only=True, mode="min", monitor="val_distillation_loss")
    trainer = pl.Trainer(
        default_root_dir=args.log_dir,
        accelerator="auto",
        max_epochs=args.epochs,
        callbacks=[save_callback, recon_callback],
        enable_progress_bar=args.progress_bar,
    )
    trainer.logger._default_hp_metric = None
    if not args.progress_bar:
        print(
            "[INFO] The progress bar has been suppressed. For updates on the training "
            f"progress, check the TensorBoard file at {trainer.logger.log_dir}. If you "
            "want to see the progress bar, use the argparse option \"progress_bar\".\n"
        )

    pl.seed_everything(args.seed)
    print('Using skip connections in decoder:', args.no_skip)
    print('Decoder densify mode:', args.decoder_densify_mode)
    print('Decoder upconv method:', args.upconv_method)
    print('Encoder & Decoder Norm type:', args.norm_type)
    print('Using reconstruction loss:', args.reconstruction_loss)
    print('Using denoising:', args.denoise)
    print('Teacher EMA decay:', args.teacher_ema_decay)
    
    # Define model with the specified hyperparameters
    model = JEPA(
        num_filters=args.num_filters,
        lr=args.lr,
        mask_ratio=args.mask_ratio,
        patch_size=args.patch_size,
        masked_loss_weight=args.masked_loss_weight,
        num_input_channels=args.num_input_channels,
        decoder_densify_mode=args.decoder_densify_mode,
        use_skip=args.no_skip,
        upconv_method=args.upconv_method,
        norm_type=args.norm_type,
        denoise=args.denoise,
        reconstruction_loss=args.reconstruction_loss,
        teacher_ema_decay=args.teacher_ema_decay,
        **masking_kwargs_from_args(args),
    )

    trainer.fit(model, train_loader, val_loader)

    model = JEPA.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)
    return trainer.test(model, dataloaders=test_loader, verbose=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Training params
    parser.add_argument("--epochs", default=21, type=int, help="Max number of epochs.") # probably less is enough
    parser.add_argument("--lr", default=1e-3, type=float, help="Learning rate to use.")
    parser.add_argument("--batch_size", default=128, type=int, help="Minibatch size.")
    parser.add_argument("--seed", default=42, type=int, help="Seed to use for reproducing results.")
    
    # Masking / data params
    parser.add_argument("--denoise", action="store_true", 
                        help="Whether to add noise to the visible pixels (denoising MAE).")
    parser.set_defaults(denoise=True)
    add_masking_arguments(parser)
    
    # JEPA params
    parser.add_argument("--masked_loss_weight", default=1.0, type=float, help="Extra weight for masked pixels in MSE.")
    parser.add_argument("--teacher_ema_decay", default=0.99, type=float, help="Decay rate for the teacher EMA model.")
    parser.add_argument("--reconstruction_loss", action="store_true", 
                        help="Whether to include reconstruction loss (probably not good idea).")


    # Architecture params
    parser.add_argument("--num_filters", default=32, type=int, 
                        help="Number of channels/filters to use.")
    parser.add_argument("--num_input_channels", default=1, type=int,
                        help="Number of image channels (1 for MNIST/FashionMNIST, 3 for CIFAR/SVHN).")
    parser.add_argument("--decoder_densify_mode",default="random",choices=("random", "token", "zero"),type=str,
                        help="How sparse encoder features are filled before decoder local processing.",)
    parser.add_argument("--upconv_method", default="upsample+conv", choices=("transposed_conv", "upsample+conv"), type=str,
                        help="Whether to use transposed convolutions or upsample+conv in the decoder.")
    parser.add_argument("--norm_type", default="rmsnorm", choices=("layernorm", "rmsnorm"), type=str, 
                        help="Type of normalization to use in the model.")
    # For JEPA we do not want skip connections
    parser.add_argument("--no_skip", action="store_false", 
                        help="Whether to use skip connections in the decoder.")
    parser.set_defaults(no_skip=False)
    
    # Logging / other params
    parser.add_argument("--data_dir", default="../data/", type=str, help="Directory where to look for the data.")
    parser.add_argument("--num_workers",default=10,type=int,
                        help=("Number of workers to use in data loaders. For strict determinism set this to 0."),)
    parser.add_argument("--log_dir", default="JEPA_logs", type=str, help="Directory for PyTorch Lightning logs.")
    parser.add_argument("--progress_bar",action="store_true",
        help=("Use a progress bar indicator for interactive experimentation. Not to be used with SLURM jobs."),)

    args = parser.parse_args()
    train_mae(args)
