# COALA: COllapsed Autoencoder with Local Adaptation
import argparse
import os
from collections.abc import Callable

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from torchvision.utils import make_grid, save_image

from cc.datasets.msmnist import msmnist
from cc.ml import COALA_logs
from cc.ml.architecture import (
    COALANet,
    create_temporal_prediction_figure,
    load_pretrained_weights,
    stack_temporal_logits,
)
from cc.ml.heads.classifier import TemporalCEloss


def _build_temporal_image_grid(
    images: torch.Tensor,
    num_examples: int = 8,
    max_time_steps: int = 16,
) -> torch.Tensor:
    images = images[:num_examples].detach().cpu()
    _, total_t, _, _, _ = images.shape
    num_time_steps = min(total_t, max_time_steps)
    time_idx = torch.linspace(0, total_t - 1, steps=num_time_steps).round().long()
    panel = images[:, time_idx].reshape(-1, *images.shape[2:])
    pad_value = float(panel.min().item() + 0.5 * (panel.max().item() - panel.min().item()))
    return make_grid(panel, nrow=num_time_steps, normalize=False, pad_value=pad_value)


class COALA(pl.LightningModule):
    """
    Lightning wrapper for task-specific COALA adaptation.
    """

    def __init__(
        self,
        model: COALANet,
        loss_fn: Callable = TemporalCEloss(),
        lr: float = 1e-3,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model", "loss_fn"])
        self.model = model
        self.loss_fn = loss_fn
        self.automatic_optimization = False
        self.model.set_adaptation_trainable(False)
        self.model.dynamic_updates = True

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return stack_temporal_logits(self.model(images))

    def configure_optimizers(self):
        return None

    def _labels_to_targets(self, labels: torch.Tensor, timesteps: int) -> torch.Tensor:
        if labels.ndim == 1:
            return labels.unsqueeze(1).expand(-1, timesteps)
        if labels.ndim == 2:
            if labels.shape[1] == timesteps:
                return labels
            if labels.shape[1] == 1:
                return labels.expand(-1, timesteps)
        raise ValueError(f"Unsupported target shape {tuple(labels.shape)} for {timesteps} timesteps.")

    def _predict_logits(self, images: torch.Tensor, dynamic_updates: bool) -> torch.Tensor:
        previous_dynamic_updates = self.model.dynamic_updates
        self.model.dynamic_updates = dynamic_updates
        self.model.reset_dynamic_state(ref_tensor=images)
        try:
            with torch.no_grad():
                return self(images)
        finally:
            self.model.reset_dynamic_state(ref_tensor=images)
            self.model.dynamic_updates = previous_dynamic_updates

    def _compute_metrics(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        targets = self._labels_to_targets(labels, timesteps=logits.shape[1])
        loss = self.loss_fn(logits, targets)
        probs = torch.softmax(logits, dim=-1)
        preds = logits.argmax(dim=-1)
        metrics = {
            "final_accuracy": (preds[:, -1] == targets[:, -1]).float().mean(),
            "temporal_accuracy": (preds == targets).float().mean(),
            "final_true_prob": probs[:, -1].gather(1, targets[:, -1].unsqueeze(1)).mean(),
        }
        return loss, metrics

    def _log_metrics(self, prefix: str, loss: torch.Tensor, metrics: dict[str, torch.Tensor], prog_bar: bool):
        self.log(f"{prefix}_classification_loss", loss, on_step=False, on_epoch=True, prog_bar=prog_bar)
        self.log(f"{prefix}_final_accuracy", metrics["final_accuracy"], on_step=False, on_epoch=True, prog_bar=prog_bar)
        self.log(f"{prefix}_temporal_accuracy", metrics["temporal_accuracy"], on_step=False, on_epoch=True)
        self.log(f"{prefix}_final_true_prob", metrics["final_true_prob"], on_step=False, on_epoch=True)

    def _shared_eval_step(self, batch, stage: str):
        images, labels = batch

        static_logits = self._predict_logits(images, dynamic_updates=False)
        static_loss, static_metrics = self._compute_metrics(static_logits, labels)
        self._log_metrics(f"{stage}_static", static_loss, static_metrics, prog_bar=False)

        dynamic_logits = self._predict_logits(images, dynamic_updates=True)
        dynamic_loss, dynamic_metrics = self._compute_metrics(dynamic_logits, labels)
        self._log_metrics(f"{stage}_dynamic", dynamic_loss, dynamic_metrics, prog_bar=(stage == "val"))
        return dynamic_loss

    def training_step(self, batch, batch_idx):
        images, labels = batch
        logits = self._predict_logits(images, dynamic_updates=True)
        loss, metrics = self._compute_metrics(logits, labels)
        self._log_metrics("train", loss, metrics, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        return self._shared_eval_step(batch, stage="val")

    def test_step(self, batch, batch_idx):
        return self._shared_eval_step(batch, stage="test")

    def on_train_epoch_start(self):
        self.model.set_adaptation_trainable(False)

    def on_validation_epoch_start(self):
        self.model.set_adaptation_trainable(False)

    def on_test_epoch_start(self):
        self.model.set_adaptation_trainable(False)


class COALADiagnosticsCallback(pl.Callback):
    """
    Logs masked temporal inputs, prediction curves, and adaptation parameters.
    """

    def __init__(
        self,
        every_n_epochs: int = 5,
        num_examples: int = 8,
        max_time_steps: int = 16,
        save_to_disk: bool = False,
    ):
        super().__init__()
        self.every_n_epochs = every_n_epochs
        self.num_examples = num_examples
        self.max_time_steps = max_time_steps
        self.save_to_disk = save_to_disk
        self._example_batch: tuple[torch.Tensor, torch.Tensor] | None = None

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        if batch_idx == 0:
            images, labels = batch
            self._example_batch = (
                images[:self.num_examples].detach().cpu(),
                labels[:self.num_examples].detach().cpu(),
            )

    @torch.no_grad()
    def on_validation_epoch_end(self, trainer, pl_module):
        import matplotlib.pyplot as plt

        if self._example_batch is None or trainer.logger is None:
            return
        if trainer.current_epoch % self.every_n_epochs != 0:
            return

        images_cpu, labels_cpu = self._example_batch
        images = images_cpu.to(pl_module.device)
        labels = labels_cpu.to(pl_module.device)

        epoch = trainer.current_epoch
        is_sanity = trainer.sanity_checking
        log_step = epoch + (0 if is_sanity else 1)

        grid = _build_temporal_image_grid(
            images_cpu,
            num_examples=self.num_examples,
            max_time_steps=self.max_time_steps,
        )
        trainer.logger.experiment.add_image("COALA/masked_sequence", grid, global_step=log_step)

        dynamic_logits = pl_module._predict_logits(images, dynamic_updates=True)
        dynamic_fig = create_temporal_prediction_figure(dynamic_logits, labels, max_examples=self.num_examples)
        trainer.logger.experiment.add_figure("COALA/dynamic_prediction_curves", dynamic_fig, global_step=log_step)

        static_logits = pl_module._predict_logits(images, dynamic_updates=False)
        static_fig = create_temporal_prediction_figure(static_logits, labels, max_examples=self.num_examples)
        trainer.logger.experiment.add_figure("COALA/static_prediction_curves", static_fig, global_step=log_step)

        for layer_idx, layer in enumerate(pl_module.model.iter_cc_modules()):
            trainer.logger.experiment.add_scalar(
                f"COALA/time_alpha/layer_{layer_idx}",
                float(layer.time_alpha.detach().cpu().item()),
                global_step=log_step,
            )
            trainer.logger.experiment.add_scalar(
                f"COALA/local_lr_ff/layer_{layer_idx}",
                float(layer.Lambda_FF.lr.detach().cpu().item()),
                global_step=log_step,
            )
            trainer.logger.experiment.add_scalar(
                f"COALA/local_lr_lat/layer_{layer_idx}",
                float(layer.Lambda_LAT.lr.detach().cpu().item()),
                global_step=log_step,
            )
            if layer.Lambda_FB is not None:
                trainer.logger.experiment.add_scalar(
                    f"COALA/local_lr_fb/layer_{layer_idx}",
                    float(layer.Lambda_FB.lr.detach().cpu().item()),
                    global_step=log_step,
                )

        if self.save_to_disk:
            image_name = f"epoch_{epoch}_masked_sequence.png" if not is_sanity else "pre_epoch_0_masked_sequence.png"
            save_image(grid, os.path.join(trainer.logger.log_dir, image_name))
            dynamic_fig.savefig(os.path.join(trainer.logger.log_dir, f"epoch_{epoch}_dynamic_prediction_curves.png"))
            static_fig.savefig(os.path.join(trainer.logger.log_dir, f"epoch_{epoch}_static_prediction_curves.png"))

        plt.close(dynamic_fig)
        plt.close(static_fig)


def _parse_masked_fill(masked_fill: str) -> str | float:
    if masked_fill == "random":
        return masked_fill
    return float(masked_fill)


def train_coala(args):
    os.makedirs(args.log_dir, exist_ok=True)
    train_loader, val_loader, test_loader = msmnist(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        root=args.data_dir,
        patch_size=args.patch_size,
        mask_ratio=args.mask_ratio,
        mask_pattern=args.mask_pattern,
        masked_fill=_parse_masked_fill(args.masked_fill),
        number_of_masks=args.number_of_masks,
        timesteps_per_mask=args.timesteps_per_mask,
        accepted_digits=args.accepted_digits,
    )

    diagnostics_callback = COALADiagnosticsCallback(
        every_n_epochs=args.log_every_n_epochs,
        num_examples=args.num_log_examples,
        max_time_steps=args.max_log_time_steps,
        save_to_disk=True,
    )
    save_callback = ModelCheckpoint(
        save_weights_only=True,
        mode="min",
        monitor="val_dynamic_classification_loss",
    )
    trainer = pl.Trainer(
        default_root_dir=args.log_dir,
        accelerator="auto",
        max_epochs=args.epochs,
        callbacks=[save_callback, diagnostics_callback],
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
    model = COALA(model=load_pretrained_weights(), lr=args.lr)
    trainer.fit(model, train_loader, val_loader)

    best_model_path = trainer.checkpoint_callback.best_model_path
    if best_model_path:
        checkpoint = torch.load(best_model_path, map_location="cpu")
        state_dict = checkpoint.get("state_dict", checkpoint)
        model.load_state_dict(state_dict, strict=True)
    model.model.set_adaptation_trainable(False)
    return trainer.test(model, dataloaders=test_loader, verbose=True)
# TODO: fix how slow this is (BPTT is the problem?)
# TODO: each batch sample should have its own lambdas, only loss should be averaged over the batch 
# (dynamic updates should happen 1 sample at a time)
# TODO: look at spectral radius of recurrent layers and clamp lambda so that eigenvales never get above 1
# to prevent explosion of activations
# TODO: consider EA and other black-box optimization for lambdas and learning rates
if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--lr", default=1e-3, type=float, help="Outer-loop optimizer learning rate.")
    parser.add_argument("--batch_size", default=1, type=int, help="Minibatch size.")
    parser.add_argument("--epochs", default=21, type=int, help="Max number of epochs.")
    parser.add_argument("--seed", default=42, type=int, help="Seed to use for reproducing results.")
    parser.add_argument("--num_workers",default=0,type=int,
        help="Number of workers to use in data loaders. For strict determinism set this to 0.",
    )
    parser.add_argument("--data_dir", default="../data/", type=str, help="Directory where to look for the data.")
    parser.add_argument("--log_dir", default=COALA_logs, type=str, help="Directory for PyTorch Lightning logs.")
    parser.add_argument("--progress_bar",action="store_true",
        help="Use a progress bar indicator for interactive experimentation. Not to be used with SLURM jobs.",
    )

    parser.add_argument("--patch_size", default=4, type=int, help="Patch size used for masking.")
    parser.add_argument("--mask_ratio", default=0.8, type=float, help="Fraction of masked patches.")
    parser.add_argument("--mask_pattern",default="random",choices=("random", "structured"),type=str,
        help="Patch-masking strategy for sequential MNIST.",
    )
    parser.add_argument("--masked_fill",default="random",type=str,
        help="Fill value for masked pixels, or 'random'. Use 0.0 to match the current architecture demo.",
    )
    parser.add_argument("--number_of_masks", default=10, type=int, help="Distinct masks per sample.")
    parser.add_argument("--timesteps_per_mask", default=10, type=int, help="How long each mask is reused.")
    parser.add_argument("--accepted_digits",nargs="*",type=int,default=None,
        help="Optional subset of MNIST digits to train on.",
    )

    parser.add_argument("--log_every_n_epochs", default=5, type=int, help="How often to log visual diagnostics.")
    parser.add_argument("--num_log_examples", default=8, type=int, help="Number of validation examples to plot.")
    parser.add_argument("--max_log_time_steps",default=10,type=int,
        help="Maximum number of timesteps to show in the masked-sequence image grid.",
    )

    args = parser.parse_args()
    train_coala(args)
