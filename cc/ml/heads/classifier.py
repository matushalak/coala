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
        freeze_encoder: bool = False,
        feature_key: str = "feat4",
    ):
        head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(start_dim=1),
            nn.Linear(latent_dim, latent_dim // 2),
            nn.BatchNorm1d(latent_dim // 2),
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
    ):
        checkpoint = torch.load(checkpoint_path, map_location=map_location)
        hparams = checkpoint.get("hyper_parameters", {})
        num_input_channels = hparams.get("num_input_channels", num_input_channels)
        num_filters = hparams.get("num_filters", num_filters)

        unet = SparseCNNUNet(
            num_input_channels=num_input_channels,
            num_output_channels=num_input_channels,
            num_filters=num_filters,
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
