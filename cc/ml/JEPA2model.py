import argparse
import os

import torch
import torch.nn.functional as F
from torchvision.utils import make_grid, save_image
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from cc.datasets.mnist import mnist
from cc.ml.sparse_cnn_unet import (
    GlobalResponseNorm,
    SparseCNNUNet,
    SparseGlobalResponseNorm,
)
from cc.utils import ExponentialMovingAverage


class SIGReg(torch.nn.Module):
    """
    Lightweight SIGReg-inspired regularizer based on random slicing and
    an Epps-Pulley-style characteristic-function matching statistic.
    """

    def __init__(
        self,
        num_slices: int = 64,
        knots: int = 17,
        t_max: float = 3.0,
        max_samples: int = 4096,
        mean_weight: float = 1.0,
        var_weight: float = 1.0,
        eps: float = 1e-4,
    ):
        super().__init__()
        if knots < 3 or knots % 2 == 0:
            raise ValueError(f"knots must be an odd integer >= 3, got {knots}")
        self.num_slices = num_slices
        self.max_samples = max_samples
        self.mean_weight = mean_weight
        self.var_weight = var_weight
        self.eps = eps

        t = torch.linspace(0.0, t_max, knots, dtype=torch.float32)
        dt = t_max / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        phi = torch.exp(-0.5 * t.square())
        self.register_buffer("t", t)
        self.register_buffer("phi", phi)
        self.register_buffer("weights", weights * phi)

    def forward(self, samples: torch.Tensor) -> torch.Tensor:
        samples = samples.float()
        if samples.ndim != 2:
            raise ValueError(f"SIGReg expects (N, D) samples, got shape {tuple(samples.shape)}")
        if samples.shape[0] < 8 or samples.shape[1] < 2:
            return samples.new_zeros(())

        if samples.shape[0] > self.max_samples:
            idx = torch.randperm(samples.shape[0], device=samples.device)[: self.max_samples]
            samples = samples[idx]

        mean = samples.mean(dim=0, keepdim=True)
        centered = samples - mean
        var = centered.var(dim=0, unbiased=False, keepdim=True)
        std = var.clamp_min(self.eps).sqrt()
        standardized = centered / std

        mean_loss = mean.square().mean()
        var_loss = (var.squeeze(0) - 1.0).square().mean()

        directions = torch.randn(
            standardized.shape[1],
            self.num_slices,
            device=standardized.device,
            dtype=standardized.dtype,
        )
        directions = directions / directions.norm(p=2, dim=0, keepdim=True).clamp_min(self.eps)

        t = self.t.to(device=standardized.device, dtype=standardized.dtype)
        phi = self.phi.to(device=standardized.device, dtype=standardized.dtype)
        weights = self.weights.to(device=standardized.device, dtype=standardized.dtype)
        x_t = (standardized @ directions).unsqueeze(-1) * t
        err = (x_t.cos().mean(dim=0) - phi).square() + x_t.sin().mean(dim=0).square()
        statistic = (err @ weights) * standardized.shape[0]
        return statistic.mean() + self.mean_weight * mean_loss + self.var_weight * var_loss


class StableGlobalResponseNorm(GlobalResponseNorm):
    def forward(self, x: torch.Tensor, keep_mask: torch.BoolTensor | None = None) -> torch.Tensor:
        if keep_mask is None:
            mask = 1.0
        else:
            mask = keep_mask.to(dtype=x.dtype)

        gx = torch.sqrt((x.pow(2) * mask).sum(dim=[2, 3], keepdim=True) + self.eps)
        nx = gx / (gx.mean(dim=1, keepdim=True) + self.eps)
        return self.gamma * (x * nx) + self.beta + x


class StableSparseGlobalResponseNorm(StableGlobalResponseNorm):
    def forward(self, x: torch.Tensor, keep_mask: torch.BoolTensor) -> torch.Tensor:
        return super().forward(x, keep_mask=keep_mask) * keep_mask.to(dtype=x.dtype)


def _stabilize_grn_modules(module: torch.nn.Module) -> None:
    for name, child in module.named_children():
        if isinstance(child, SparseGlobalResponseNorm):
            replacement = StableSparseGlobalResponseNorm(child.gamma.shape[1], eps=child.eps)
            replacement.load_state_dict(child.state_dict())
            setattr(module, name, replacement)
            continue
        if isinstance(child, GlobalResponseNorm):
            replacement = StableGlobalResponseNorm(child.gamma.shape[1], eps=child.eps)
            replacement.load_state_dict(child.state_dict())
            setattr(module, name, replacement)
            continue
        _stabilize_grn_modules(child)


class JEPA2(pl.LightningModule):
    SCALE_NAMES = ("feat4", "feat7", "feat14", "feat28")

    def __init__(
        self,
        num_filters: int,
        lr: float,
        mask_ratio: float,
        patch_size: int,
        num_input_channels: int = 1,
        decoder_densify_mode: str = "random",
        use_skip: bool = False,
        upconv_method: str = "upsample+conv",
        norm_type: str = "rmsnorm",
        denoise: bool = False,
        teacher_ema_decay: float = 0.999,
        loss_exp: float = 1.0,
        masked_distill_weight: float = 1.0,
        visible_distill_weight: float = 0.5,
        feat4_loss_weight: float = 0.25,
        feat7_loss_weight: float = 0.5,
        feat14_loss_weight: float = 1.0,
        feat28_loss_weight: float = 1.0,
        normalize_distill: bool = True,
        sigreg_weight: float = 0.02,
        sigreg_num_slices: int = 64,
        sigreg_knots: int = 17,
        sigreg_t_max: float = 3.0,
        sigreg_max_samples: int = 4096,
        sigreg_mean_weight: float = 1.0,
        sigreg_var_weight: float = 1.0,
        sigreg_on: str = "encoder",
        reconstruction_loss_weight: float = 0.0,
        stabilize_grn: bool = True,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = SparseCNNUNet(
            num_input_channels=num_input_channels,
            num_output_channels=num_input_channels,
            num_filters=num_filters,
            decoder_densify_mode=decoder_densify_mode,
            use_skip=use_skip,
            upconv_method=upconv_method,
            norm_type=norm_type,
        )
        if stabilize_grn:
            _stabilize_grn_modules(self.model)
        self.teacher = ExponentialMovingAverage(
            self.model.encoder,
            decay=teacher_ema_decay,
        ).eval()
        self.sigreg = SIGReg(
            num_slices=sigreg_num_slices,
            knots=sigreg_knots,
            t_max=sigreg_t_max,
            max_samples=sigreg_max_samples,
            mean_weight=sigreg_mean_weight,
            var_weight=sigreg_var_weight,
        )
        self.scale_weights = {
            "feat4": feat4_loss_weight,
            "feat7": feat7_loss_weight,
            "feat14": feat14_loss_weight,
            "feat28": feat28_loss_weight,
        }

    def _mask(self, imgs: torch.Tensor) -> torch.BoolTensor:
        b, _, h, w = imgs.shape
        patch_size = self.hparams.patch_size
        if h % patch_size != 0 or w % patch_size != 0:
            raise ValueError(f"Image size ({h}, {w}) must be divisible by patch_size={patch_size}.")

        ph, pw = h // patch_size, w // patch_size
        num_patches = ph * pw
        num_keep = max(1, int(round((1.0 - self.hparams.mask_ratio) * num_patches)))

        noise = torch.rand(b, num_patches, device=imgs.device)
        keep_idx = noise.argsort(dim=1)[:, :num_keep]
        keep_patch = torch.zeros(b, num_patches, device=imgs.device, dtype=torch.bool)
        keep_patch.scatter_(1, keep_idx, True)
        keep_patch = keep_patch.view(b, 1, ph, pw)
        return keep_patch.repeat_interleave(patch_size, dim=2).repeat_interleave(patch_size, dim=3)

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

    def _student_input(self, imgs: torch.Tensor, keep_mask: torch.BoolTensor) -> torch.Tensor:
        if not self.hparams.denoise:
            return imgs
        noise = torch.randn_like(imgs)
        return torch.where(keep_mask, imgs + torch.randn_like(imgs), noise).clamp_(-1.0, 1.0)

    def _model_outputs(
        self,
        imgs: torch.Tensor,
        keep_mask: torch.BoolTensor | None = None,
    ) -> tuple[
        torch.Tensor,
        torch.BoolTensor,
        dict[str, torch.Tensor],
        dict[str, torch.Tensor],
        torch.Tensor,
        dict[str, torch.Tensor],
        dict[str, torch.BoolTensor],
    ]:
        if keep_mask is None:
            keep_mask = self._mask(imgs)

        teacher_latents = self.teacher(imgs, keep_mask=torch.ones_like(keep_mask, dtype=torch.bool))
        student_imgs = self._student_input(imgs, keep_mask)
        student_encoder_latents = self.model.encoder(student_imgs, keep_mask=keep_mask)
        recon, student_predictor_latents = self.model.decoder(student_encoder_latents)
        keep_masks = {k: v for k, v in student_encoder_latents.items() if "mask" in k}
        return (
            student_imgs,
            keep_mask,
            teacher_latents,
            student_encoder_latents,
            recon,
            student_predictor_latents,
            keep_masks,
        )

    def _regression_map(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = pred.float()
        target = target.float()
        if self.hparams.normalize_distill:
            pred = F.normalize(pred, dim=1)
            target = F.normalize(target, dim=1)
        diff = (pred - target).abs().pow(self.hparams.loss_exp)
        return diff.mean(dim=1) / self.hparams.loss_exp

    def _dense_predictive_loss(
        self,
        predictor_latents: dict[str, torch.Tensor],
        teacher_latents: dict[str, torch.Tensor],
        keep_masks: dict[str, torch.BoolTensor],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        total = predictor_latents["feat4"].new_zeros(())
        total_weight = predictor_latents["feat4"].new_zeros(())
        metrics = {}

        for feat_name in self.SCALE_NAMES:
            scale_weight = self.scale_weights[feat_name]
            if scale_weight <= 0:
                continue

            keep_mask = keep_masks[feat_name.replace("feat", "mask")].squeeze(1)
            reg_map = self._regression_map(predictor_latents[feat_name], teacher_latents[feat_name])
            masked_loss = self._masked_spatial_mean(reg_map, ~keep_mask)
            visible_loss = self._masked_spatial_mean(reg_map, keep_mask)
            scale_loss = (
                self.hparams.masked_distill_weight * masked_loss
                + self.hparams.visible_distill_weight * visible_loss
            )
            total = total + scale_weight * scale_loss.mean()
            total_weight = total_weight + scale_weight

            student_norm = predictor_latents[feat_name].float().pow(2).mean(dim=1).sqrt().mean()
            teacher_norm = teacher_latents[feat_name].float().pow(2).mean(dim=1).sqrt().mean()
            metrics[f"{feat_name}_loss"] = scale_loss.mean()
            metrics[f"{feat_name}_masked_loss"] = masked_loss.mean()
            metrics[f"{feat_name}_visible_loss"] = visible_loss.mean()
            metrics[f"{feat_name}_student_norm"] = student_norm
            metrics[f"{feat_name}_teacher_norm"] = teacher_norm

        return total / total_weight.clamp_min(1e-8), metrics

    def _sigreg_samples(
        self,
        feat: torch.Tensor,
        keep_mask: torch.BoolTensor | None = None,
    ) -> torch.Tensor | None:
        samples = feat.float().movedim(1, -1).reshape(-1, feat.shape[1])
        if keep_mask is not None:
            valid = keep_mask.squeeze(1).reshape(-1)
            if valid.any():
                samples = samples[valid]
            else:
                return None
        if samples.shape[0] < 8:
            return None
        return samples

    def _sigreg_loss(
        self,
        student_encoder_latents: dict[str, torch.Tensor],
        student_predictor_latents: dict[str, torch.Tensor],
        keep_masks: dict[str, torch.BoolTensor],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        total = student_predictor_latents["feat4"].new_zeros(())
        total_weight = student_predictor_latents["feat4"].new_zeros(())
        metrics = {}

        for feat_name in self.SCALE_NAMES:
            scale_weight = self.scale_weights[feat_name]
            if scale_weight <= 0:
                continue

            scale_terms = []
            if self.hparams.sigreg_on in ("encoder", "both"):
                enc_samples = self._sigreg_samples(
                    student_encoder_latents[feat_name],
                    keep_masks[feat_name.replace("feat", "mask")],
                )
                if enc_samples is not None:
                    scale_terms.append(self.sigreg(enc_samples))

            if self.hparams.sigreg_on in ("predictor", "both"):
                pred_samples = self._sigreg_samples(student_predictor_latents[feat_name])
                if pred_samples is not None:
                    scale_terms.append(self.sigreg(pred_samples))

            if not scale_terms:
                metrics[f"{feat_name}_sigreg"] = total.new_zeros(())
                continue

            scale_reg = torch.stack(scale_terms).mean()
            total = total + scale_weight * scale_reg
            total_weight = total_weight + scale_weight
            metrics[f"{feat_name}_sigreg"] = scale_reg

        return total / total_weight.clamp_min(1e-8), metrics

    def _reconstruction_loss(
        self,
        recon: torch.Tensor,
        imgs: torch.Tensor,
        keep_mask: torch.BoolTensor,
    ) -> torch.Tensor:
        per_pixel_mae = (recon - imgs).abs().mean(dim=1)
        masked = (~keep_mask.squeeze(1)).to(dtype=per_pixel_mae.dtype)
        weights = torch.ones_like(per_pixel_mae) + masked * (self.hparams.masked_distill_weight - 1.0)
        per_image = (per_pixel_mae * weights).sum(dim=(1, 2)) / weights.sum(dim=(1, 2)).clamp_min(1e-8)
        return per_image.mean()

    def compute_losses(
        self,
        imgs: torch.Tensor,
        keep_mask: torch.BoolTensor | None = None,
        return_metrics: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]] | tuple[torch.Tensor]:
        (
            student_imgs,
            keep_mask,
            teacher_latents,
            student_encoder_latents,
            recon,
            student_predictor_latents,
            keep_masks,
        ) = self._model_outputs(imgs, keep_mask=keep_mask)

        distill_loss, distill_metrics = self._dense_predictive_loss(
            student_predictor_latents,
            teacher_latents,
            keep_masks,
        )
        sigreg_loss, sigreg_metrics = self._sigreg_loss(
            student_encoder_latents,
            student_predictor_latents,
            keep_masks,
        )
        recon_loss = self._reconstruction_loss(recon, student_imgs, keep_mask)
        total_loss = (
            distill_loss
            + self.hparams.sigreg_weight * sigreg_loss
            + self.hparams.reconstruction_loss_weight * recon_loss
        )

        if not return_metrics:
            return (total_loss,)

        metrics = {}
        metrics.update(distill_metrics)
        metrics.update(sigreg_metrics)
        return total_loss, distill_loss, sigreg_loss, recon_loss, metrics

    def forward(self, imgs: torch.Tensor, keep_mask: torch.BoolTensor | None = None) -> torch.Tensor:
        return self.compute_losses(imgs, keep_mask=keep_mask, return_metrics=False)[0]

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
        cov = cov + 1e-6 * torch.eye(cov.shape[0], device=cov.device, dtype=cov.dtype)
        try:
            if cov.device.type == "mps":
                _, eigvecs = torch.linalg.eigh(cov.cpu())
                eigvecs = eigvecs.to(device=cov.device)
            else:
                _, eigvecs = torch.linalg.eigh(cov)
        except RuntimeError:
            _, _, vh = torch.linalg.svd(teacher_centered, full_matrices=False)
            eigvecs = vh.transpose(-2, -1)

        num_components = min(3, c)
        axes = eigvecs[:, -num_components:]
        teacher_rgb = teacher_centered @ axes
        student_rgb = student_centered @ axes
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

    @torch.no_grad()
    def feature_visualizations(
        self,
        imgs: torch.Tensor,
        keep_mask: torch.BoolTensor | None = None,
    ) -> dict[str, torch.Tensor]:
        student_imgs, keep_mask, teacher_latents, _, _, student_predictor_latents, _ = self._model_outputs(
            imgs, keep_mask=keep_mask
        )
        visualizations = {
            "context": torch.cat(
                [
                    self._expand_to_rgb(imgs.float()),
                    self._expand_to_rgb(
                        torch.where(
                            keep_mask,
                            student_imgs,
                            student_imgs.new_zeros((), dtype=student_imgs.dtype),
                        ).float()
                    ),
                ],
                dim=0,
            )
        }

        target_size = imgs.shape[-2:]
        for feat_name in self.SCALE_NAMES[::-1]:
            teacher_rgb, student_rgb = self._project_pair_to_rgb(
                teacher_latents[feat_name],
                student_predictor_latents[feat_name],
            )
            error_rgb = self._error_to_rgb(
                self._regression_map(
                    student_predictor_latents[feat_name],
                    teacher_latents[feat_name],
                ).unsqueeze(1)
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
        return torch.optim.Adam(self.model.parameters(), lr=self.hparams.lr)

    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        self.teacher.update(self.model.encoder)

    def _log_metrics(self, prefix: str, metrics: dict[str, torch.Tensor], *, on_step: bool, on_epoch: bool):
        for name, value in metrics.items():
            self.log(f"{prefix}_{name}", value, on_step=on_step, on_epoch=on_epoch)

    def training_step(self, batch, batch_idx):
        total_loss, distill_loss, sigreg_loss, recon_loss, metrics = self.compute_losses(
            batch[0],
            return_metrics=True,
        )
        self.log("train_total_loss", total_loss, on_step=False, on_epoch=True)
        self.log("train_distillation_loss", distill_loss, on_step=False, on_epoch=True)
        self.log("train_sigreg_loss", sigreg_loss, on_step=False, on_epoch=True)
        self.log("train_reconstruction_loss", recon_loss, on_step=False, on_epoch=True)
        self._log_metrics("train", metrics, on_step=False, on_epoch=True)
        return total_loss

    def validation_step(self, batch, batch_idx):
        total_loss, distill_loss, sigreg_loss, recon_loss, metrics = self.compute_losses(
            batch[0],
            return_metrics=True,
        )
        self.log("val_total_loss", total_loss, on_step=False, on_epoch=True)
        self.log("val_distillation_loss", distill_loss, on_step=False, on_epoch=True)
        self.log("val_sigreg_loss", sigreg_loss, on_step=False, on_epoch=True)
        self.log("val_reconstruction_loss", recon_loss, on_step=False, on_epoch=True)
        self._log_metrics("val", metrics, on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx):
        total_loss, distill_loss, sigreg_loss, recon_loss, metrics = self.compute_losses(
            batch[0],
            return_metrics=True,
        )
        self.log("test_total_loss", total_loss, on_step=False, on_epoch=True)
        self.log("test_distillation_loss", distill_loss, on_step=False, on_epoch=True)
        self.log("test_sigreg_loss", sigreg_loss, on_step=False, on_epoch=True)
        self.log("test_reconstruction_loss", recon_loss, on_step=False, on_epoch=True)
        self._log_metrics("test", metrics, on_step=False, on_epoch=True)


class DenseFeatureCallback(pl.Callback):
    """
    Logs JEPA2 context plus multi-scale teacher / student / error visualizations.
    """

    def __init__(self, every_n_epochs: int = 5, num_images: int = 20, save_to_disk: bool = False):
        super().__init__()
        self.every_n_epochs = every_n_epochs
        self.num_images = num_images
        self.save_to_disk = save_to_disk
        self._example_batch = None

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        if self._example_batch is None and batch_idx == 0:
            self._example_batch = batch[0][: self.num_images].detach().cpu()

    @torch.no_grad()
    def on_validation_epoch_end(self, trainer, pl_module):
        if self._example_batch is None:
            return
        if trainer.current_epoch % self.every_n_epochs != 0:
            return

        imgs = self._example_batch.to(pl_module.device)
        epoch = trainer.current_epoch
        is_sanity = trainer.sanity_checking
        log_step = epoch + (0 if is_sanity else 1)
        visualizations = pl_module.feature_visualizations(imgs)

        for name, panel in visualizations.items():
            grid = make_grid(
                panel.detach().cpu(),
                nrow=self.num_images,
                normalize=True,
                pad_value=0.0,
                value_range=(-1.0, 1.0),
            )
            tag = "JEPA2/context" if name == "context" else f"JEPA2/{name}_teacher_student_error"
            trainer.logger.experiment.add_image(tag, grid, global_step=log_step)

            if self.save_to_disk:
                suffix = "context" if name == "context" else f"{name}_teacher_student_error"
                image_name = f"epoch_{epoch}_{suffix}.png" if not is_sanity else f"pre_epoch_0_{suffix}.png"
                save_image(grid, os.path.join(trainer.logger.log_dir, image_name))


def train_jepa2(args):
    os.makedirs(args.log_dir, exist_ok=True)
    train_loader, val_loader, test_loader = mnist(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        root=args.data_dir,
    )

    vis_callback = DenseFeatureCallback(save_to_disk=True)
    save_callback = ModelCheckpoint(save_weights_only=True, mode="min", monitor="val_total_loss")
    trainer = pl.Trainer(
        default_root_dir=args.log_dir,
        accelerator="auto",
        max_epochs=args.epochs,
        callbacks=[save_callback, vis_callback],
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
    print("Using skip connections in decoder:", args.use_skip)
    print("Decoder densify mode:", args.decoder_densify_mode)
    print("Decoder upconv method:", args.upconv_method)
    print("Encoder & Decoder Norm type:", args.norm_type)
    print("Using denoising:", args.denoise)
    print("Teacher EMA decay:", args.teacher_ema_decay)
    print("Visible distill weight:", args.visible_distill_weight)
    print("Masked distill weight:", args.masked_distill_weight)
    print("SIGReg weight:", args.sigreg_weight)
    print("SIGReg target:", args.sigreg_on)

    model = JEPA2(
        num_filters=args.num_filters,
        lr=args.lr,
        mask_ratio=args.mask_ratio,
        patch_size=args.patch_size,
        num_input_channels=args.num_input_channels,
        decoder_densify_mode=args.decoder_densify_mode,
        use_skip=args.use_skip,
        upconv_method=args.upconv_method,
        norm_type=args.norm_type,
        denoise=args.denoise,
        teacher_ema_decay=args.teacher_ema_decay,
        loss_exp=args.loss_exp,
        masked_distill_weight=args.masked_distill_weight,
        visible_distill_weight=args.visible_distill_weight,
        feat4_loss_weight=args.feat4_loss_weight,
        feat7_loss_weight=args.feat7_loss_weight,
        feat14_loss_weight=args.feat14_loss_weight,
        feat28_loss_weight=args.feat28_loss_weight,
        normalize_distill=args.normalize_distill,
        sigreg_weight=args.sigreg_weight,
        sigreg_num_slices=args.sigreg_num_slices,
        sigreg_knots=args.sigreg_knots,
        sigreg_t_max=args.sigreg_t_max,
        sigreg_max_samples=args.sigreg_max_samples,
        sigreg_mean_weight=args.sigreg_mean_weight,
        sigreg_var_weight=args.sigreg_var_weight,
        sigreg_on=args.sigreg_on,
        reconstruction_loss_weight=args.reconstruction_loss_weight,
        stabilize_grn=args.stabilize_grn,
    )

    trainer.fit(model, train_loader, val_loader)
    model = JEPA2.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)
    return trainer.test(model, dataloaders=test_loader, verbose=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--epochs", default=21, type=int, help="Max number of epochs.")
    parser.add_argument("--lr", default=1e-3, type=float, help="Learning rate to use.")
    parser.add_argument("--batch_size", default=128, type=int, help="Minibatch size.")
    parser.add_argument("--seed", default=42, type=int, help="Seed to use for reproducing results.")

    parser.add_argument("--denoise", action="store_true", help="Whether to add noise to visible pixels.")
    parser.add_argument("--mask_ratio", default=0.6, type=float, help="Fraction of patches to hide.")
    parser.add_argument("--patch_size", default=4, type=int, help="Patch size used for random masking.")

    parser.add_argument("--teacher_ema_decay", default=0.999, type=float, help="Decay rate for the EMA teacher.")
    parser.add_argument("--loss_exp", default=1.0, type=float, help="Lp exponent for the dense predictive loss.")
    parser.add_argument("--masked_distill_weight", default=1.0, type=float, help="Weight for masked predictive loss.")
    parser.add_argument("--visible_distill_weight", default=0.5, type=float, help="Weight for visible/context predictive loss.")
    parser.add_argument("--feat4_loss_weight", default=0.25, type=float, help="Deep supervision weight for feat4.")
    parser.add_argument("--feat7_loss_weight", default=0.5, type=float, help="Deep supervision weight for feat7.")
    parser.add_argument("--feat14_loss_weight", default=1.0, type=float, help="Deep supervision weight for feat14.")
    parser.add_argument("--feat28_loss_weight", default=1.0, type=float, help="Deep supervision weight for feat28.")
    parser.add_argument("--normalize_distill", action=argparse.BooleanOptionalAction, default=True, help="Normalize features before dense regression.")
    parser.add_argument("--sigreg_weight", default=0.02, type=float, help="Weight for SIGReg regularization.")
    parser.add_argument("--sigreg_num_slices", default=64, type=int, help="Number of random SIGReg projections.")
    parser.add_argument("--sigreg_knots", default=17, type=int, help="Number of quadrature knots for SIGReg.")
    parser.add_argument("--sigreg_t_max", default=3.0, type=float, help="Maximum integration value for SIGReg.")
    parser.add_argument("--sigreg_max_samples", default=4096, type=int, help="Maximum number of tokens sampled per scale for SIGReg.")
    parser.add_argument("--sigreg_mean_weight", default=1.0, type=float, help="Penalty weight on latent means in SIGReg.")
    parser.add_argument("--sigreg_var_weight", default=1.0, type=float, help="Penalty weight on latent variances in SIGReg.")
    parser.add_argument("--sigreg_on", default="encoder", choices=("encoder", "predictor", "both"), help="Where to apply SIGReg.")
    parser.add_argument("--reconstruction_loss_weight", default=0.0, type=float, help="Optional reconstruction regularizer weight.")
    parser.add_argument("--stabilize_grn", action=argparse.BooleanOptionalAction, default=True, help="Replace GRN layers with numerically stable variants inside JEPA2.")

    parser.add_argument("--num_filters", default=32, type=int, help="Number of channels/filters to use.")
    parser.add_argument("--num_input_channels", default=1, type=int, help="Number of image channels.")
    parser.add_argument(
        "--decoder_densify_mode",
        default="random",
        choices=("random", "token", "zero"),
        type=str,
        help="How sparse encoder features are filled before decoder local processing.",
    )
    parser.add_argument(
        "--upconv_method",
        default="upsample+conv",
        choices=("transposed_conv", "upsample+conv"),
        type=str,
        help="Whether to use transposed convolutions or upsample+conv in the decoder.",
    )
    parser.add_argument(
        "--norm_type",
        default="rmsnorm",
        choices=("layernorm", "rmsnorm"),
        type=str,
        help="Type of normalization to use in the model.",
    )
    parser.add_argument("--use_skip", action="store_true", help="Whether to use decoder skip connections.")

    parser.add_argument("--data_dir", default="../data/", type=str, help="Directory where to look for the data.")
    parser.add_argument("--num_workers", default=10, type=int, help="Number of workers to use in data loaders.")
    parser.add_argument("--log_dir", default="JEPA2_logs", type=str, help="Directory for PyTorch Lightning logs.")
    parser.add_argument("--progress_bar", action="store_true", help="Use a progress bar during training.")

    args = parser.parse_args()
    train_jepa2(args)
