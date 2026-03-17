import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from cc.ml.heads.task_head import TaskHead
from cc.ml.sparse_cnn_unet import SparseCNNEncoder, SparseCNNUNet

# Specific task heads
class ClassifierHead(TaskHead):
    def __init__(
        self,
        encoder: SparseCNNEncoder,
        num_classes: int,
        latent_dim: int,
        lr: float = 1e-3,
        freeze_encoder: bool = True,
        feature_key: str = "feat4",
    ):
        head = nn.Sequential(
            nn.AdaptiveMaxPool2d((1, 1)), # better than average pool
            nn.Flatten(start_dim=1),
            nn.Linear(latent_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, latent_dim // 2),
            nn.LayerNorm(latent_dim // 2),
            nn.GELU(),
            nn.Linear(latent_dim // 2, num_classes),
        )
        super().__init__(
            backbone=encoder,
            head=head,
            feature_key=feature_key,
            lr=lr,
            freeze_backbone=freeze_encoder,
            task_name="classifier",
            monitor_metric="val_classification_loss",
            monitor_mode="min",
        )
        self.save_hyperparameters(ignore=["encoder"])

    @classmethod
    def from_pretrained_unet(
        cls,
        checkpoint_path: str,
        num_classes: int,
        latent_dim: int,
        lr: float = 1e-3,
        freeze_encoder: bool = True,
        num_input_channels: int = 1,
        num_filters: int = 32,
        map_location: str | torch.device = "cpu",
        upconv_method: str = "transposed_conv",
    ):
        checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
        hparams = checkpoint.get("hyper_parameters", {})
        num_input_channels = hparams.get("num_input_channels", num_input_channels)
        num_filters = hparams.get("num_filters", num_filters)
        decoder_densify_mode = str(hparams.get("decoder_densify_mode", "random"))
        use_skip = bool(hparams.get("use_skip", True))
        upconv_method = str(hparams.get("upconv_method", upconv_method))
        # Older MAE checkpoints predate this hparam and used LayerNorm throughout.
        norm_type = str(hparams.get("norm_type", "layernorm"))

        unet = SparseCNNUNet(
            num_input_channels=num_input_channels,
            num_output_channels=num_input_channels,
            num_filters=num_filters,
            decoder_densify_mode=decoder_densify_mode,
            use_skip=use_skip,
            upconv_method=upconv_method,
            norm_type=norm_type,
        )
        state_dict = checkpoint.get("state_dict", checkpoint)
        if any(k.startswith("model.") for k in state_dict):
            state_dict = {k[len("model."):]: v for k, v in state_dict.items() if k.startswith("model.")}
        unet.load_state_dict(state_dict, strict=False)
        return cls(
            encoder=unet.encoder,
            num_classes=num_classes,
            latent_dim=latent_dim,
            lr=lr,
            freeze_encoder=freeze_encoder,
        )

    def _shared_step(self, batch, stage: str):
        imgs, labels = batch
        logits = self.forward(imgs)
        loss = F.cross_entropy(logits, labels)
        acc = (logits.argmax(dim=1) == labels).float().mean()
        self.log(f"{stage}_classification_loss", loss, on_step=False, on_epoch=True)
        self.log(f"{stage}_accuracy", acc, on_step=False, on_epoch=True, prog_bar=(stage != "train"))
        return loss

class TemporalCEloss(nn.CrossEntropyLoss):
    def __init__(
        self,
        *args,
        plateau_fraction: float = 0.5,
        min_weight: float = 1e-2,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.plateau_fraction = plateau_fraction
        self.min_weight = min_weight

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # input shape: (B, T, C), target shape: (B, T)
        B, T, C = input.shape
        losses = F.cross_entropy(
            input.reshape(B * T, C),
            target.reshape(B * T),
            weight=self.weight,
            ignore_index=self.ignore_index,
            reduction="none",
            label_smoothing=self.label_smoothing,
        ).reshape(B, T)

        if T == 1:
            time_weights = losses.new_ones(1)
        else:
            plateau_fraction = max(self.plateau_fraction, 1e-3)
            time_steps = torch.arange(T, device=input.device, dtype=losses.dtype)
            normalized_time = time_steps / (T - 1)
            saturation_rate = math.log(100.0) / plateau_fraction
            time_weights = 1.0 - torch.exp(-saturation_rate * normalized_time)
            time_weights = self.min_weight + (1.0 - self.min_weight) * time_weights

        if self.ignore_index >= 0:
            valid_mask = target.ne(self.ignore_index)
            losses = losses * valid_mask
            denom = (valid_mask * time_weights).sum()
        else:
            denom = torch.full((), B * time_weights.sum(), device=losses.device, dtype=losses.dtype)

        weighted_losses = losses * time_weights
        if self.reduction == "none":
            return weighted_losses
        if self.reduction == "sum":
            return weighted_losses.sum()
        return weighted_losses.sum() / denom.clamp_min(1)
