import argparse
import os

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning.callbacks import ModelCheckpoint
from torchvision.utils import make_grid, save_image

from cc import DATADIR
from cc.datasets import get_dataloaders
from cc.ml import COALA_logs, dataset_log_dir
from cc.ml.masking import add_masking_arguments, clear_mask_bank_caches, masking_kwargs_from_args, sample_keep_mask
from cc.ml.pretraining.common import (
    PREDICTOR_MODES,
    combine_feature_latents,
    configure_adamw_with_warmup_and_cosine_decay,
    default_model_config,
    instantiate_autoencoder,
    normalize_model_config,
    normalize_predictor_config,
    select_prediction_latents,
)
from cc.utils import ExponentialMovingAverage


class SIGReg(torch.nn.Module):
    def __init__(self, knots=17, random_projections=64, max_samples: int | None = 1024):
        super().__init__()
        assert knots >= 2
        if max_samples is not None:
            assert max_samples > 0
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
        assert proj.ndim >= 2
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
        noise_level: float | None = None,
        sigreg_loss_weight: float = 0.1,
        sigreg_max_samples: int = 1024,
        model_config: dict | None = None,
        predictor_config: dict | None = None,
        predictor_mode: str = "predictor+decoder",
        masking_strategy: str = "random",
        multi_block_scale_min: float = 0.15,
        multi_block_scale_max: float = 0.2,
        multi_block_aspect_ratio_min: float = 0.75,
        multi_block_aspect_ratio_max: float = 1.5,
        multi_block_square_aspect_ratio: float = 1.0,
    ):
        super().__init__()
        if noise_level is not None:
            denoise_sigma = noise_level
        assert predictor_mode in PREDICTOR_MODES
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
        predictor_config = normalize_predictor_config(predictor_config)

        self.model = instantiate_autoencoder(model_config, predictive=True, predictor_config=predictor_config)
        self.feature_names = list(self.model.encoder.feature_names)
        self.teacher = ExponentialMovingAverage(self.model.encoder, decay=0.99).eval()
        self.sigreg = SIGReg(max_samples=sigreg_max_samples).eval()
        self.save_hyperparameters(ignore=["noise_level"])

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
        noise = self.hparams.denoise_sigma * torch.randn_like(imgs)
        return torch.where(keep_mask, imgs + noise, noise).clamp_(-1.0, 1.0)

    def _jepa_outputs(self, imgs: torch.Tensor, keep_mask: torch.BoolTensor | None = None):
        if keep_mask is None:
            keep_mask = self._mask(imgs)
        full_mask = torch.ones_like(keep_mask, dtype=torch.bool)
        teacher_latents = self.teacher(imgs, keep_mask=full_mask)
        clean_student_latents = self.model.encoder(imgs, keep_mask=full_mask)
        student_imgs = self._student_input(imgs, keep_mask)
        decoder_latents, dirty_encoder_latents, predictor_latents = self.model(student_imgs, keep_mask=keep_mask)
        predicted_latents = select_prediction_latents(
            self.hparams.predictor_mode,
            decoder_latents=decoder_latents,
            predictor_latents=predictor_latents,
            feature_names=self.feature_names,
        )
        coala_latents = combine_feature_latents(dirty_encoder_latents, predicted_latents, self.feature_names)
        keep_masks = {
            feat_name.replace("feat", "mask"): dirty_encoder_latents[feat_name.replace("feat", "mask")]
            for feat_name in self.feature_names
        }
        return student_imgs, keep_mask, clean_student_latents, teacher_latents, coala_latents, keep_masks

    def self_distillation_loss(self, predictor_latents, clean_latents, keep_masks):
        per_scale = []
        metrics = {}
        for feat_name in self.feature_names:
            diff = (predictor_latents[feat_name] - clean_latents[feat_name].detach()).pow(2).mean(dim=1)
            keep_mask = keep_masks[feat_name.replace("feat", "mask")].squeeze(1)
            weights = torch.ones_like(diff) + (~keep_mask).to(dtype=diff.dtype) * (self.hparams.masked_loss_weight - 1.0)
            scale_loss = self._weighted_spatial_mean(diff, weights)
            per_scale.append(scale_loss)
            metrics[f"{feat_name}_loss"] = scale_loss.mean()
            metrics[f"{feat_name}_masked_loss"] = self._masked_spatial_mean(diff, ~keep_mask).mean()
            metrics[f"{feat_name}_visible_loss"] = self._masked_spatial_mean(diff, keep_mask).mean()
            metrics[f"{feat_name}_student_norm"] = predictor_latents[feat_name].float().pow(2).mean(dim=1).sqrt().mean()
            metrics[f"{feat_name}_clean_norm"] = clean_latents[feat_name].float().pow(2).mean(dim=1).sqrt().mean()
        return torch.stack(per_scale, dim=1).mean(), metrics

    @staticmethod
    def _sigreg_samples(feat: torch.Tensor) -> torch.Tensor:
        return feat.float().movedim(1, -1).reshape(-1, feat.shape[1])

    def sigreg_loss(self, clean_latents):
        per_scale = []
        metrics = {}
        for feat_name in self.feature_names:
            scale_sigreg = self.sigreg(self._sigreg_samples(clean_latents[feat_name]))
            per_scale.append(scale_sigreg)
            metrics[f"{feat_name}_sigreg"] = scale_sigreg
        return torch.stack(per_scale).mean(), metrics

    def forward(self, imgs: torch.Tensor, keep_mask: torch.BoolTensor | None = None):
        _, _, clean_student_latents, teacher_latents, coala_latents, keep_masks = self._jepa_outputs(
            imgs, keep_mask=keep_mask
        )
        distill_loss, distill_metrics = self.self_distillation_loss(coala_latents, teacher_latents, keep_masks)
        sigreg_loss, sigreg_metrics = self.sigreg_loss(clean_student_latents)
        total_loss = distill_loss + self.hparams.sigreg_loss_weight * sigreg_loss
        metrics = {}
        metrics.update(distill_metrics)
        metrics.update(sigreg_metrics)
        return total_loss, distill_loss, sigreg_loss, metrics

    @torch.no_grad()
    def _project_pair_to_rgb(self, teacher_latents: torch.Tensor, student_latents: torch.Tensor):
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
            teacher_rgb = F.pad(teacher_rgb, (0, 3 - num_components))
            student_rgb = F.pad(student_rgb, (0, 3 - num_components))
        teacher_rgb = teacher_rgb.view(b, h, w, 3).movedim(-1, 1)
        student_rgb = student_rgb.view(b, h, w, 3).movedim(-1, 1)
        flat = torch.cat([teacher_rgb.flatten(2), student_rgb.flatten(2)], dim=2)
        rgb_min = flat.amin(dim=2, keepdim=True)
        rgb_max = flat.amax(dim=2, keepdim=True)
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
        student_imgs, keep_mask, _, teacher_latents, coala_latents, _ = self._jepa_outputs(imgs, keep_mask=keep_mask)
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
            teacher_rgb, student_rgb = self._project_pair_to_rgb(teacher_latents[feat_name], coala_latents[feat_name])
            error_rgb = self._error_to_rgb((coala_latents[feat_name] - teacher_latents[feat_name]).pow(2).mean(dim=1, keepdim=True))
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
        return configure_adamw_with_warmup_and_cosine_decay(self)

    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        self.teacher.update(self.model.encoder)

    def _log_metrics(self, prefix: str, metrics: dict[str, torch.Tensor], *, on_step: bool, on_epoch: bool):
        for name, value in metrics.items():
            self.log(f"{prefix}_{name}", value, on_step=on_step, on_epoch=on_epoch)

    def training_step(self, batch, batch_idx):
        total_loss, distill_loss, sigreg_loss, metrics = self.forward(batch[0])
        self.log("train_total_loss", total_loss, on_step=False, on_epoch=True)
        self.log("train_distillation_loss", distill_loss, on_step=False, on_epoch=True)
        self.log("train_sigreg_loss", sigreg_loss, on_step=False, on_epoch=True)
        self._log_metrics("train", metrics, on_step=False, on_epoch=True)
        return total_loss

    def validation_step(self, batch, batch_idx):
        total_loss, distill_loss, sigreg_loss, metrics = self.forward(batch[0])
        self.log("val_total_loss", total_loss, on_step=False, on_epoch=True)
        self.log("val_distillation_loss", distill_loss, on_step=False, on_epoch=True)
        self.log("val_sigreg_loss", sigreg_loss, on_step=False, on_epoch=True)
        self._log_metrics("val", metrics, on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx):
        total_loss, distill_loss, sigreg_loss, metrics = self.forward(batch[0])
        self.log("test_total_loss", total_loss, on_step=False, on_epoch=True)
        self.log("test_distillation_loss", distill_loss, on_step=False, on_epoch=True)
        self.log("test_sigreg_loss", sigreg_loss, on_step=False, on_epoch=True)
        self._log_metrics("test", metrics, on_step=False, on_epoch=True)


class FeatureVisualizationCallback(pl.Callback):
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
            image_tag = "COALA/context" if name == "context" else f"COALA/{name}_teacher_student_error"
            trainer.logger.experiment.add_image(image_tag, grid, global_step=log_step)
            if self.save_to_disk:
                suffix = "context" if name == "context" else f"{name}_teacher_student_error"
                image_name = f"epoch_{epoch}_{suffix}.png" if not is_sanity else f"pre_epoch_0_{suffix}.png"
                save_image(grid, os.path.join(trainer.logger.log_dir, image_name))


def train_coala(args):
    data_dir = os.path.abspath(args.data_dir)
    log_dir = dataset_log_dir(args.log_dir, args.dataset)
    os.makedirs(log_dir, exist_ok=True)
    train_loader, val_loader, test_loader = get_dataloaders(
        args.dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        root=data_dir,
    )
    vis_callback = FeatureVisualizationCallback(save_to_disk=True)
    save_callback = ModelCheckpoint(save_weights_only=True, mode="min", monitor="val_total_loss")
    trainer = pl.Trainer(
        default_root_dir=log_dir,
        accelerator="auto",
        max_epochs=args.epochs,
        callbacks=[save_callback, vis_callback],
        enable_progress_bar=args.progress_bar,
    )
    trainer.logger._default_hp_metric = None
    pl.seed_everything(args.seed)
    model = COALA(
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
        sigreg_loss_weight=args.sigreg_loss_weight,
        sigreg_max_samples=args.sigreg_max_samples,
        predictor_mode=args.predictor_mode,
        **masking_kwargs_from_args(args),
    )
    trainer.fit(model, train_loader, val_loader)
    model = COALA.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)
    return trainer.test(model, dataloaders=test_loader, verbose=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--epochs", default=21, type=int)
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--warmup_epochs", default=0, type=int)
    parser.add_argument("--weight_decay", default=0.0, type=float)
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--denoise", action="store_true")
    parser.set_defaults(denoise=True)
    add_masking_arguments(parser)
    parser.add_argument("--masked_loss_weight", default=1.0, type=float)
    parser.add_argument("--sigreg_loss_weight", default=0.1, type=float)
    parser.add_argument("--sigreg_max_samples", default=2048, type=int)
    parser.add_argument("--num_filters", default=32, type=int)
    parser.add_argument("--num_input_channels", default=1, type=int)
    parser.add_argument("--image_size", default=28, type=int)
    parser.add_argument("--dataset", default="mnist", type=str)
    parser.add_argument("--decoder_densify_mode", default="random", choices=("random", "token", "zero"), type=str)
    parser.add_argument("--upconv_method", default="upsample+conv", choices=("transposed_conv", "upsample+conv"), type=str)
    parser.add_argument("--norm_type", default="rmsnorm", choices=("layernorm", "rmsnorm"), type=str)
    parser.add_argument("--predictor_mode", default="predictor+decoder", choices=PREDICTOR_MODES, type=str)
    parser.add_argument("--no_skip", dest="use_skip", action="store_false")
    parser.set_defaults(use_skip=True)
    parser.add_argument("--data_dir", default=DATADIR, type=str)
    parser.add_argument("--num_workers", default=10, type=int)
    parser.add_argument("--log_dir", default=COALA_logs, type=str)
    parser.add_argument("--progress_bar", action="store_true")
    train_coala(parser.parse_args())
