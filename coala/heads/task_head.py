import os
import inspect
from collections.abc import Callable
from typing import Any

import pytorch_lightning as pl
import torch
import torch.nn as nn
from pytorch_lightning.callbacks import ModelCheckpoint

from coala import Head_logs, dataset_log_dir

def _expects_keep_mask_arg(module: nn.Module) -> bool:
    try:
        return "keep_mask" in inspect.signature(module.forward).parameters
    except (TypeError, ValueError):
        return False

# General task head class
class TaskHead(pl.LightningModule):
    def __init__(
        self,
        backbone: nn.Module,
        head: nn.Module,
        feature_key: str | None = None,
        feature_selector: Callable[[Any], torch.Tensor] | None = None,
        lr: float = 1e-3,
        freeze_backbone: bool = False,
        task_name: str = "task_head",
        monitor_metric: str = "val_loss",
        monitor_mode: str = "min",
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["backbone", "head", "feature_selector"])
        self.backbone = backbone
        self.head = head
        self.feature_key = feature_key
        self.feature_selector = feature_selector
        self.task_name = task_name
        self.monitor_metric = monitor_metric
        self.monitor_mode = monitor_mode
        self._expects_keep_mask = _expects_keep_mask_arg(backbone)
        if freeze_backbone:
            self.backbone.requires_grad_(False)
            self.backbone.eval()

    def _forward_backbone(self, x: torch.Tensor) -> Any:
        if self._expects_keep_mask:
            keep_mask = torch.ones((x.shape[0], 1, x.shape[2], x.shape[3]), device=x.device, dtype=torch.bool)
            return self.backbone(x, keep_mask=keep_mask)
        return self.backbone(x)

    def _select_features(self, backbone_out: Any) -> torch.Tensor:
        if self.feature_selector is not None:
            return self.feature_selector(backbone_out)
        if self.feature_key is None:
            return backbone_out
        return backbone_out[self.feature_key]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self._select_features(self._forward_backbone(x))
        return self.head(features)

    def configure_optimizers(self):
        trainable_params = (p for p in self.parameters() if p.requires_grad)
        return torch.optim.Adam(trainable_params, lr=self.hparams.lr)

    def _shared_step(self, batch, stage: str):
        raise NotImplementedError("Task-specific head must implement _shared_step.")

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, stage="train")

    def validation_step(self, batch, batch_idx):
        self._shared_step(batch, stage="val")

    def test_step(self, batch, batch_idx):
        self._shared_step(batch, stage="test")


def create_task_head_trainer(
    model: TaskHead,
    log_root_dir: str = Head_logs,
    dataset_name: str = "mnist",
    callbacks: list[Any] | None = None,
    **trainer_kwargs: Any,
) -> pl.Trainer:
    task_log_dir = dataset_log_dir(os.path.join(log_root_dir, model.task_name), dataset_name)
    os.makedirs(task_log_dir, exist_ok=True)

    callback_list = list(callbacks or [])
    if not any(isinstance(cb, ModelCheckpoint) for cb in callback_list):
        callback_list.append(
            ModelCheckpoint(
                save_weights_only=True,
                monitor=model.monitor_metric,
                mode=model.monitor_mode,
            )
        )

    trainer = pl.Trainer(default_root_dir=task_log_dir, callbacks=callback_list, **trainer_kwargs)
    if trainer.logger is not None and hasattr(trainer.logger, "_default_hp_metric"):
        trainer.logger._default_hp_metric = None
    return trainer
