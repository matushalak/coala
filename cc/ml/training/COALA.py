import argparse
import os

import torch
import torch.nn.functional as F
from torchvision.utils import make_grid, save_image
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from cc.datasets.mnist import mnist
from cc.ml.masking import add_masking_arguments, clear_mask_bank_caches, masking_kwargs_from_args, sample_keep_mask
from cc.ml.architecture.sparse_cnn_unet import SparseCNNUNet
from cc.utils import ExponentialMovingAverage

class SIGReg(torch.nn.Module):
    """
    Sketched Isotropic Gaussian Regularization (SIGReg) for latent space.
    
    Args:
        - knots: number of points along random projection axis to evaluate
        - random_projections: number of random projections to average over for stability
        - max_samples: maximum number of samples to use for SIGReg computation (for memory efficiency)
    """
    def __init__(self, knots=17, random_projections=64, max_samples: int | None = 1024):
        super().__init__()
        assert knots >= 2, f"knots must be >= 2, got {knots}"
        if max_samples is not None:
            assert max_samples > 0, f"max_samples must be positive, got {max_samples}"
        self.random_projections = random_projections
        self.max_samples = max_samples
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, proj):
        assert proj.ndim >= 2, f"SIGReg expects at least 2 dimensions, got shape {tuple(proj.shape)}"
        if self.max_samples is not None and proj.size(-2) > self.max_samples:
            idx = torch.randperm(proj.size(-2), device=proj.device)[: self.max_samples]
            proj = proj.index_select(-2, idx)

        t = self.t.to(device=proj.device, dtype=proj.dtype)
        phi = self.phi.to(device=proj.device, dtype=proj.dtype)
        weights = self.weights.to(device=proj.device, dtype=proj.dtype)
        A = torch.randn(proj.size(-1), self.random_projections, device=proj.device, dtype=proj.dtype)
        A = A.div_(A.norm(p=2, dim=0))
        x_t = (proj @ A).unsqueeze(-1) * t
        err = (x_t.cos().mean(-3) - phi).square() + x_t.sin().mean(-3).square()
        statistic = (err @ weights) * proj.size(-2)
        return statistic.mean()


class COALA(pl.LightningModule):
    SCALE_NAMES = ("feat28", "feat14", "feat7", "feat4")

    """
    LeJEPA-style pretraining with SigReg regularization on latent space.

    Here we include a predictor (similar to LeWorldModel and previous JEPA iterations)
        and regularize latent space of dirty predictor and clean student encoder.
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
        use_skip: bool = False,
        upconv_method: str = "upsample+conv",
        norm_type: str = "rmsnorm",
        denoise: bool = False,
        noise_level: float = 0.5,
        sigreg_loss_weight: float = 0.1,
        sigreg_max_samples: int = 1024,
        masking_strategy: str = "random",
        multi_block_scale_min: float = 0.15,
        multi_block_scale_max: float = 0.2,
        multi_block_aspect_ratio_min: float = 0.75,
        multi_block_aspect_ratio_max: float = 1.5,
        multi_block_square_aspect_ratio: float = 1.0,
    ):
        super().__init__()
        self.save_hyperparameters()
        # original LeJEPA has no more student-teacher distinction
        # but it also does not use masking / denoising anymore
        # for masking & denoising, student-ema_teacher is still useful
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
            decay = 0.99,
        ).eval()

        # Sketched Isotropic Gaussian Regularization (SIGReg) module for latent space regularization
        self.sigreg = SIGReg(max_samples=sigreg_max_samples).eval()
        
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

        noise = torch.randn_like(imgs) * self.hparams.noise_level
        # dMAE-style noise to corrupt images
        return torch.where(keep_mask, imgs + noise, noise).clamp_(-1.0, 1.0)

    def _jepa_outputs(
        self,
        imgs: torch.Tensor,
        keep_mask: torch.BoolTensor | None = None,
    ) -> tuple[torch.Tensor, torch.BoolTensor, dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, torch.BoolTensor]]:
        if keep_mask is None:
            keep_mask = self._mask(imgs)
        teacher_latents = self.teacher(imgs, keep_mask=torch.ones_like(keep_mask, dtype=torch.bool))
        clean_student_latents = self.model.encoder(imgs, keep_mask=torch.ones_like(keep_mask, dtype=torch.bool))
        student_imgs = self._student_input(imgs, keep_mask)
        dirty_encoder_latents = self.model.encoder(student_imgs, keep_mask=keep_mask)
        recon, dirty_predictor_latents = self.model.decoder(dirty_encoder_latents)
        
        coala_latents = {k: dirty_encoder_latents[k] + dirty_predictor_latents[k] 
                         for k in dirty_predictor_latents}
        
        keep_masks = {k: v for k, v in dirty_encoder_latents.items() if "mask" in k}
        return student_imgs, keep_mask, clean_student_latents, teacher_latents, coala_latents, recon, keep_masks

    def self_distillation_loss(
        self,
        predictor_latents: dict[str, torch.Tensor],
        clean_latents: dict[str, torch.Tensor],
        keep_masks: dict[str, torch.BoolTensor],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        # because we have masked denoising objective, must do stopgrad on decoder
        diffs = {
            k: (predictor_latents[k] - clean_latents[k].detach()).pow(2).mean(dim=1)
            for k in predictor_latents
        }

        per_scale = []
        metrics = {}
        for feat_name, diff in diffs.items():
            keep_mask = keep_masks[feat_name.replace("feat", "mask")].squeeze(1)
            weights = (
                torch.ones_like(diff)
                + (~keep_mask).to(dtype=diff.dtype) * (self.hparams.masked_loss_weight - 1.0)
            )
            scale_loss = self._weighted_spatial_mean(diff, weights)
            per_scale.append(scale_loss)

            masked_loss = self._masked_spatial_mean(diff, ~keep_mask)
            visible_loss = self._masked_spatial_mean(diff, keep_mask)
            student_norm = predictor_latents[feat_name].float().pow(2).mean(dim=1).sqrt().mean()
            clean_norm = clean_latents[feat_name].float().pow(2).mean(dim=1).sqrt().mean()
            metrics[f"{feat_name}_loss"] = scale_loss.mean()
            metrics[f"{feat_name}_masked_loss"] = masked_loss.mean()
            metrics[f"{feat_name}_visible_loss"] = visible_loss.mean()
            metrics[f"{feat_name}_student_norm"] = student_norm
            metrics[f"{feat_name}_clean_norm"] = clean_norm

        loss = torch.stack(per_scale, dim=1).mean()
        return loss, metrics

    @staticmethod
    def _sigreg_samples(feat: torch.Tensor) -> torch.Tensor:
        return feat.float().movedim(1, -1).reshape(-1, feat.shape[1])

    def sigreg_loss(
        self,
        clean_latents: dict[str, torch.Tensor],
        # dirty_predictor_latents: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        per_scale = []
        metrics = {}

        for feat_name in self.SCALE_NAMES:
            clean_sigreg = self.sigreg(self._sigreg_samples(clean_latents[feat_name]))
            # dirty_sigreg = self.sigreg(self._sigreg_samples(dirty_predictor_latents[feat_name]))
            scale_sigreg = clean_sigreg
            per_scale.append(scale_sigreg)
            # metrics[f"{feat_name}_clean_sigreg"] = clean_sigreg
            # metrics[f"{feat_name}_dirty_sigreg"] = dirty_sigreg
            metrics[f"{feat_name}_sigreg"] = scale_sigreg

        return torch.stack(per_scale).mean(), metrics

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
    
    def forward(
        self,
        imgs: torch.Tensor,
        keep_mask: torch.BoolTensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        student_imgs, keep_mask, clean_student_latents, teacher_latents, coala_latents, recon, keep_masks = self._jepa_outputs(
            imgs, keep_mask=keep_mask
        )

        distill_loss, distill_metrics = self.self_distillation_loss(
            coala_latents,
            teacher_latents,
            keep_masks,
        )
        sigreg_loss, sigreg_metrics = self.sigreg_loss(clean_student_latents)
        recon_loss = self._reconstruction_loss(recon, imgs, keep_mask)
        total_loss = distill_loss + (self.hparams.sigreg_loss_weight * sigreg_loss) + (0.5 * self.hparams.sigreg_loss_weight * recon_loss)

        metrics = {}
        metrics.update(distill_metrics)
        metrics.update(sigreg_metrics)
        return total_loss, distill_loss, sigreg_loss, recon_loss, metrics
    
    @torch.no_grad()
    def _project_pair_to_rgb(
        self,
        teacher_latents: torch.Tensor,
        student_latents: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        b, c, h, w = teacher_latents.shape
        teacher_flat = teacher_latents.float().movedim(1, -1).reshape(-1, c)
        student_flat = student_latents.float().movedim(1, -1).reshape(-1, c)
        both_flat = torch.cat([teacher_flat, student_flat], dim=0)
        mean = both_flat.mean(dim=0, keepdim=True)
        teacher_centered = teacher_flat - mean
        student_centered = student_flat - mean
        both_centered = both_flat - mean
        cov = both_centered.T @ both_centered
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
        return self._normalize_projected_rgb(teacher_rgb), self._normalize_projected_rgb(student_rgb)

    @staticmethod
    def _normalize_projected_rgb(rgb: torch.Tensor) -> torch.Tensor:
        b, c, h, w = rgb.shape
        flat = rgb.flatten(2)
        scale = flat.abs().amax(dim=2, keepdim=True).clamp_min(1e-6)
        return flat.div(scale).view(b, c, h, w).clamp_(-1.0, 1.0)

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
        student_imgs, keep_mask, clean_student_latents, teacher_latents, coala_latents, recon, keep_masks = self._jepa_outputs(
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
        for feat_name in self.SCALE_NAMES:
            teacher_rgb, student_rgb = self._project_pair_to_rgb(
                teacher_latents[feat_name],
                coala_latents[feat_name],
            )
            error_rgb = self._error_to_rgb(
                (coala_latents[feat_name] - teacher_latents[feat_name]).pow(2).mean(dim=1, keepdim=True)
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

    def _log_metrics(self, prefix: str, metrics: dict[str, torch.Tensor], *, on_step: bool, on_epoch: bool):
        for name, value in metrics.items():
            self.log(f"{prefix}_{name}", value, on_step=on_step, on_epoch=on_epoch)

    def training_step(self, batch, batch_idx):
        total_loss, distill_loss, sigreg_loss, recon_loss, metrics = self.forward(batch[0])
        self.log("train_total_loss", total_loss, on_step=False, on_epoch=True)
        self.log("train_distillation_loss", distill_loss, on_step=False, on_epoch=True)
        self.log("train_sigreg_loss", sigreg_loss, on_step=False, on_epoch=True)
        self.log("train_reconstruction_loss", recon_loss, on_step=False, on_epoch=True)
        self._log_metrics("train", metrics, on_step=False, on_epoch=True)
        return total_loss

    def validation_step(self, batch, batch_idx):
        total_loss, distill_loss, sigreg_loss, recon_loss, metrics = self.forward(batch[0])
        self.log("val_total_loss", total_loss, on_step=False, on_epoch=True)
        self.log("val_distillation_loss", distill_loss, on_step=False, on_epoch=True)
        self.log("val_sigreg_loss", sigreg_loss, on_step=False, on_epoch=True)
        self.log("val_reconstruction_loss", recon_loss, on_step=False, on_epoch=True)
        self._log_metrics("val", metrics, on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx):
        total_loss, distill_loss, sigreg_loss, recon_loss, metrics = self.forward(batch[0])
        self.log("test_total_loss", total_loss, on_step=False, on_epoch=True)
        self.log("test_distillation_loss", distill_loss, on_step=False, on_epoch=True)
        self.log("test_sigreg_loss", sigreg_loss, on_step=False, on_epoch=True)
        self.log("test_reconstruction_loss", recon_loss, on_step=False, on_epoch=True)
        self._log_metrics("test", metrics, on_step=False, on_epoch=True)


class FeatureVisualizationCallback(pl.Callback):
    """
    Logs LeJEPA context plus multi-scale clean / predictor / error visualizations.
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
            image_tag = "LeJEPA/context" if name == "context" else f"LeJEPA/{name}_teacher_student_error"
            trainer.logger.experiment.add_image(image_tag, grid, global_step=log_step)

            if self.save_to_disk:
                suffix = "context" if name == "context" else f"{name}_teacher_student_error"
                image_name = f"epoch_{epoch}_{suffix}.png" if not is_sanity else f"pre_epoch_0_{suffix}.png"
                save_image(grid, os.path.join(trainer.logger.log_dir, image_name))


def train_lejepa(args):
    os.makedirs(args.log_dir, exist_ok=True)
    train_loader, val_loader, test_loader = mnist(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        root=args.data_dir,
    )

    vis_callback = FeatureVisualizationCallback(save_to_disk=True)
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
    print('Decoder densify mode:', args.decoder_densify_mode)
    print('Decoder upconv method:', args.upconv_method)
    print('Encoder & Decoder Norm type:', args.norm_type)
    print('Using denoising:', args.denoise)
    print('Denoising noise level:', args.noise_level)
    print("SIGReg loss weight:", args.sigreg_loss_weight)
    print("SIGReg max samples per call:", args.sigreg_max_samples)
    
    # Define model with the specified hyperparameters
    model = COALA(
        num_filters=args.num_filters,
        lr=args.lr,
        mask_ratio=args.mask_ratio,
        patch_size=args.patch_size,
        noise_level=args.noise_level,
        masked_loss_weight=args.masked_loss_weight,
        num_input_channels=args.num_input_channels,
        decoder_densify_mode=args.decoder_densify_mode,
        use_skip=args.use_skip,
        upconv_method=args.upconv_method,
        norm_type=args.norm_type,
        denoise=args.denoise,
        sigreg_loss_weight=args.sigreg_loss_weight,
        sigreg_max_samples=args.sigreg_max_samples,
        **masking_kwargs_from_args(args),
    )

    trainer.fit(model, train_loader, val_loader)

    model = COALA.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)
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
    parser.add_argument("--noise_level", default=1.0, type=float, help="Level of noise to add for denoising.")
    add_masking_arguments(parser)
    parser.set_defaults(mask_ratio=0.6,
                        patch_size=4,
                        masking_strategy="multi-block", 
                        multi_block_scale_min=0.1, 
                        multi_block_scale_max=0.6, 
                        multi_block_aspect_ratio_min=0.5, 
                        multi_block_aspect_ratio_max=1.5, 
                        multi_block_square_aspect_ratio=1.0
                        )

    # JEPA params
    parser.add_argument("--masked_loss_weight", default=2.0, type=float, help="Extra weight for masked pixels in MSE.")
    parser.add_argument("--sigreg_loss_weight", default=0.1, type=float, help="Weight for latent SIGReg regularization.")
    parser.add_argument("--sigreg_max_samples", default=1024, type=int,
                        help="Maximum number of latent samples used per SIGReg call.")


    # Architecture params
    parser.add_argument("--num_filters", default=64, type=int, 
                        help="Number of channels/filters to use.")
    parser.add_argument("--num_input_channels", default=1, type=int,
                        help="Number of image channels (1 for MNIST/FashionMNIST, 3 for CIFAR/SVHN).")
    parser.add_argument("--decoder_densify_mode",default="random",choices=("random", "token", "zero"),type=str,
                        help="How sparse encoder features are filled before decoder local processing.",)
    parser.add_argument("--upconv_method", default="upsample+conv", choices=("transposed_conv", "upsample+conv"), type=str,
                        help="Whether to use transposed convolutions or upsample+conv in the decoder.")
    parser.add_argument("--norm_type", default="rmsnorm", choices=("layernorm", "rmsnorm"), type=str, 
                        help="Type of normalization to use in the model.")
    parser.add_argument("--use_skip", action="store_true", help="Whether to use skip connections in the decoder.")
    
    # Logging / other params
    parser.add_argument("--data_dir", default="../data/", type=str, help="Directory where to look for the data.")
    parser.add_argument("--num_workers",default=10,type=int,
                        help=("Number of workers to use in data loaders. For strict determinism set this to 0."),)
    parser.add_argument("--log_dir", default="LeJEPA_logs", type=str, help="Directory for PyTorch Lightning logs.")
    parser.add_argument("--progress_bar",action="store_true",
        help=("Use a progress bar indicator for interactive experimentation. Not to be used with SLURM jobs."),)

    args = parser.parse_args()
    train_lejepa(args)
