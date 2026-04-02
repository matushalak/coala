import argparse
import os

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning.callbacks import ModelCheckpoint
from torchvision.utils import make_grid, save_image

from cc import DATADIR
from cc.datasets import get_dataloaders
from cc.ml import JEPA_logs, dataset_log_dir
from cc.ml.masking import add_masking_arguments, clear_mask_bank_caches, masking_kwargs_from_args, sample_keep_mask
from cc.ml.pretraining.common import (
    GenerativeHead,
    default_model_config,
    default_reconstruction_head_config,
    instantiate_autoencoder,
    normalize_model_config,
    normalize_predictor_config,
    normalize_reconstruction_head_config,
)
from cc.utils import ExponentialMovingAverage


class JEPA(pl.LightningModule):
    def __init__(
        self,
        num_filters: int,
        lr: float,
        mask_ratio: float,
        patch_size: int,
        masked_loss_weight: float,
        num_input_channels: int = 1,
        image_size: int = 28,
        decoder_densify_mode: str = "random",
        use_skip: bool = True,
        upconv_method: str = "upsample+conv",
        norm_type: str = "rmsnorm",
        denoise: bool = False,
        denoise_sigma: float = 1.0,
        reconstruction_loss: bool = False,
        teacher_ema_decay: float = 0.999,
        reconstruction_head_family: str = "ViT",
        model_config: dict | None = None,
        predictor_config: dict | None = None,
        reconstruction_head_config: dict | None = None,
        masking_strategy: str = "random",
        multi_block_scale_min: float = 0.15,
        multi_block_scale_max: float = 0.2,
        multi_block_aspect_ratio_min: float = 0.75,
        multi_block_aspect_ratio_max: float = 1.5,
        multi_block_square_aspect_ratio: float = 1.0,
    ):
        super().__init__()
        assert use_skip
        if model_config is None:
            model_config = default_model_config(
                image_size=image_size,
                num_input_channels=num_input_channels,
                num_filters=num_filters,
                norm_type=norm_type,
                decoder_densify_mode=decoder_densify_mode,
                use_skip=True,
                upconv_method=upconv_method,
            )
        model_config = normalize_model_config(model_config)
        model_config["D_kwargs"]["use_skip"] = True
        predictor_config = normalize_predictor_config(predictor_config)

        self.model = instantiate_autoencoder(model_config, predictive=True, predictor_config=predictor_config)
        self.feature_names = list(self.model.encoder.feature_names)
        self.teacher = ExponentialMovingAverage(self.model.encoder, decay=teacher_ema_decay).eval()
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
        self.reconstruction_feature = self.feature_names[0]
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

    def _student_input(self, imgs: torch.Tensor, keep_mask: torch.BoolTensor) -> torch.Tensor:
        if not self.hparams.denoise:
            return imgs
        noise = self.hparams.denoise_sigma * torch.randn_like(imgs)
        return torch.where(keep_mask, imgs + noise, noise).clamp_(-1.0, 1.0)

    def _reconstruction_loss(self, recon: torch.Tensor, imgs: torch.Tensor, keep_mask: torch.BoolTensor) -> torch.Tensor:
        per_pixel_mse = (recon - imgs).pow(2).mean(dim=1)
        masked_pixels = (~keep_mask.squeeze(1)).to(dtype=per_pixel_mse.dtype)
        weights = torch.ones_like(per_pixel_mse) + masked_pixels * (self.hparams.masked_loss_weight - 1.0)
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

    def _jepa_outputs(self, imgs: torch.Tensor, keep_mask: torch.BoolTensor | None = None):
        if keep_mask is None:
            keep_mask = self._mask(imgs)
        full_mask = torch.ones_like(keep_mask, dtype=torch.bool)
        teacher_latents = self.teacher(imgs, keep_mask=full_mask)
        student_imgs = self._student_input(imgs, keep_mask)
        decoder_latents, student_encoder_latents, predicted_latents = self.model(student_imgs, keep_mask=keep_mask)
        recon = self.reconstruction_head(decoder_latents[self.reconstruction_feature])
        keep_masks = {name.replace("feat", "mask"): student_encoder_latents[name.replace("feat", "mask")] for name in self.feature_names}
        return student_imgs, keep_mask, teacher_latents, recon, predicted_latents, keep_masks

    def self_distillation_loss(self, predictor_latents, teacher_latents, keep_masks, return_metrics: bool = False):
        per_scale = []
        metrics = {}
        for feat_name in self.feature_names:
            diff = (predictor_latents[feat_name] - teacher_latents[feat_name]).pow(2).mean(dim=1)
            keep_mask = keep_masks[feat_name.replace("feat", "mask")].squeeze(1)
            weights = torch.ones_like(diff) + (~keep_mask).to(dtype=diff.dtype) * (self.hparams.masked_loss_weight - 1.0)
            scale_loss = self._weighted_spatial_mean(diff, weights)
            per_scale.append(scale_loss)
            if return_metrics:
                metrics[f"{feat_name}_loss"] = scale_loss.mean()
                metrics[f"{feat_name}_masked_loss"] = self._masked_spatial_mean(diff, ~keep_mask).mean()
                metrics[f"{feat_name}_visible_loss"] = self._masked_spatial_mean(diff, keep_mask).mean()
                metrics[f"{feat_name}_student_norm"] = predictor_latents[feat_name].float().pow(2).mean(dim=1).sqrt().mean()
                metrics[f"{feat_name}_teacher_norm"] = teacher_latents[feat_name].float().pow(2).mean(dim=1).sqrt().mean()
        loss = torch.stack(per_scale, dim=1).mean()
        if return_metrics:
            return loss, metrics
        return loss

    def forward(self, imgs: torch.Tensor, keep_mask: torch.BoolTensor | None = None, return_metrics: bool = False):
        student_imgs, keep_mask, teacher_latents, recon, predicted_latents, keep_masks = self._jepa_outputs(
            imgs, keep_mask=keep_mask
        )
        if return_metrics:
            distill_loss, metrics = self.self_distillation_loss(predicted_latents, teacher_latents, keep_masks, return_metrics=True)
        else:
            distill_loss = self.self_distillation_loss(predicted_latents, teacher_latents, keep_masks)
        recon_loss = self._reconstruction_loss(recon, student_imgs, keep_mask)
        if return_metrics:
            return distill_loss, recon_loss, metrics
        return distill_loss, recon_loss

    @torch.no_grad()
    def _project_pair_to_rgb(self, teacher_latents: torch.Tensor, student_latents: torch.Tensor):
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
            teacher_rgb = F.pad(teacher_rgb, (0, 3 - num_components))
            student_rgb = F.pad(student_rgb, (0, 3 - num_components))
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
        assert False

    @staticmethod
    def _error_to_rgb(error_map: torch.Tensor) -> torch.Tensor:
        b, _, h, w = error_map.shape
        error_flat = error_map.float().flatten(2)
        error_min = error_flat.amin(dim=2, keepdim=True)
        error_max = error_flat.amax(dim=2, keepdim=True)
        error = error_flat.sub(error_min).div((error_max - error_min).clamp_min(1e-6)).view(b, 1, h, w)
        return torch.cat([error.mul(2.0).sub(1.0), error.new_full((b, 1, h, w), -1.0), error.new_full((b, 1, h, w), -1.0)], dim=1)

    @staticmethod
    def _upsample_for_display(x: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
        if x.shape[-2:] == size:
            return x
        return F.interpolate(x, size=size, mode="nearest")

    @torch.no_grad()
    def feature_visualizations(self, imgs: torch.Tensor, keep_mask: torch.BoolTensor | None = None):
        student_imgs, keep_mask, teacher_latents, _, predicted_latents, _ = self._jepa_outputs(imgs, keep_mask=keep_mask)
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
        for feat_name in self.feature_names:
            teacher_rgb, student_rgb = self._project_pair_to_rgb(teacher_latents[feat_name], predicted_latents[feat_name])
            error_rgb = self._error_to_rgb((predicted_latents[feat_name] - teacher_latents[feat_name]).pow(2).mean(dim=1, keepdim=True))
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
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

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
        return loss_distil + self.hparams.reconstruction_loss * loss_recon

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
        grey = 0.0
        epoch = trainer.current_epoch
        is_sanity = trainer.sanity_checking
        log_step = epoch + (0 if is_sanity else 1)
        visualizations = pl_module.feature_visualizations(imgs)
        for name, panel in visualizations.items():
            grid = make_grid(panel.detach().cpu(), nrow=self.num_images, normalize=True, pad_value=grey, value_range=(-1.0, 1.0))
            image_tag = "JEPA/context" if name == "context" else f"JEPA/{name}_teacher_student_error"
            trainer.logger.experiment.add_image(image_tag, grid, global_step=log_step)
            if self.save_to_disk:
                suffix = "context" if name == "context" else f"{name}_teacher_student_error"
                image_name = f"epoch_{epoch}_{suffix}.png" if not is_sanity else f"pre_epoch_0_{suffix}.png"
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
    save_callback = ModelCheckpoint(save_weights_only=True, mode="min", monitor="val_distillation_loss")
    trainer = pl.Trainer(
        default_root_dir=log_dir,
        accelerator="auto",
        max_epochs=args.epochs,
        callbacks=[save_callback, recon_callback],
        enable_progress_bar=args.progress_bar,
    )
    trainer.logger._default_hp_metric = None
    pl.seed_everything(args.seed)
    model = JEPA(
        num_filters=args.num_filters,
        lr=args.lr,
        mask_ratio=args.mask_ratio,
        patch_size=args.patch_size,
        masked_loss_weight=args.masked_loss_weight,
        num_input_channels=args.num_input_channels,
        image_size=args.image_size,
        decoder_densify_mode=args.decoder_densify_mode,
        use_skip=True,
        upconv_method=args.upconv_method,
        norm_type=args.norm_type,
        denoise=args.denoise,
        reconstruction_loss=args.reconstruction_loss,
        teacher_ema_decay=args.teacher_ema_decay,
        reconstruction_head_family=args.reconstruction_head_family,
        **masking_kwargs_from_args(args),
    )
    trainer.fit(model, train_loader, val_loader)
    model = JEPA.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)
    return trainer.test(model, dataloaders=test_loader, verbose=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--epochs", default=21, type=int)
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--denoise", action="store_true")
    parser.set_defaults(denoise=True)
    add_masking_arguments(parser)
    parser.add_argument("--masked_loss_weight", default=1.0, type=float)
    parser.add_argument("--teacher_ema_decay", default=0.99, type=float)
    parser.add_argument("--reconstruction_loss", action="store_true")
    parser.add_argument("--num_filters", default=32, type=int)
    parser.add_argument("--num_input_channels", default=1, type=int)
    parser.add_argument("--image_size", default=28, type=int)
    parser.add_argument("--dataset", default="mnist", type=str)
    parser.add_argument("--decoder_densify_mode", default="random", choices=("random", "token", "zero"), type=str)
    parser.add_argument("--upconv_method", default="upsample+conv", choices=("transposed_conv", "upsample+conv"), type=str)
    parser.add_argument("--norm_type", default="rmsnorm", choices=("layernorm", "rmsnorm"), type=str)
    parser.add_argument("--reconstruction_head_family", default="ViT", choices=("ViT", "ConvNet", "ConvNeXt"), type=str)
    parser.add_argument("--data_dir", default=DATADIR, type=str)
    parser.add_argument("--num_workers", default=10, type=int)
    parser.add_argument("--log_dir", default=JEPA_logs, type=str)
    parser.add_argument("--progress_bar", action="store_true")
    train_mae(parser.parse_args())
