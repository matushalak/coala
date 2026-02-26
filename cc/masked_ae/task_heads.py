import torch
import torch.nn as nn
import torch.nn.functional as F

from cc.masked_ae.sparse_cnn_unet import SparseCNNUNet, SparseCNNEncoder
import pytorch_lightning as pl

class ClassifierHead(pl.LightningModule):
    def __init__(
        self,
        encoder: SparseCNNEncoder,
        num_classes: int,
        latent_dim: int,
        lr: float = 1e-3,
        freeze_encoder: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["encoder"])
        self.encoder = encoder
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(latent_dim, num_classes)
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

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

    def forward(self, x):
        keep_mask = torch.ones((x.shape[0], 1, x.shape[2], x.shape[3]), device=x.device, dtype=torch.bool)
        latent = self.encoder(x, keep_mask=keep_mask)["feat4"]
        latent = self.pool(latent).flatten(1)
        return self.classifier(latent)

    def configure_optimizers(self):
        trainable_params = (p for p in self.parameters() if p.requires_grad)
        return torch.optim.Adam(trainable_params, lr=self.hparams.lr)

    def _shared_step(self, batch, stage: str):
        imgs, labels = batch
        logits = self.forward(imgs)
        loss = F.cross_entropy(logits, labels)
        acc = (logits.argmax(dim=1) == labels).float().mean()
        self.log(f"{stage}_classification_loss", loss, on_step=False, on_epoch=True)
        self.log(f"{stage}_accuracy", acc, on_step=False, on_epoch=True, prog_bar=(stage != "train"))
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, stage="train")

    def validation_step(self, batch, batch_idx):
        self._shared_step(batch, stage="val")

    def test_step(self, batch, batch_idx):
        self._shared_step(batch, stage="test")
