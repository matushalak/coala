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


class MAE(pl.LightningModule):
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
        masking_strategy: str = "random",
        multi_block_scale_min: float = 0.15,
        multi_block_scale_max: float = 0.2,
        multi_block_aspect_ratio_min: float = 0.75,
        multi_block_aspect_ratio_max: float = 1.5,
        multi_block_square_aspect_ratio: float = 1.0,
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

    def reconstruct(
        self,
        imgs: torch.Tensor,
        keep_mask: torch.BoolTensor | None = None,
    ) -> tuple[torch.Tensor, torch.BoolTensor]:
        if keep_mask is None:
            keep_mask = self._mask(imgs)
        if self.hparams.denoise:
            noise = torch.randn_like(imgs)
            # dMAE-style noise
            imgs = torch.where(keep_mask, imgs + torch.randn_like(imgs), noise).clamp_(-1.0, 1.0)
        recon = self.model(imgs, keep_mask=keep_mask)
        return recon, keep_mask

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

    def forward(self, imgs: torch.Tensor) -> torch.Tensor:
        recon, keep_mask = self.reconstruct(imgs)
        return self._reconstruction_loss(recon, imgs, keep_mask)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

    def training_step(self, batch, batch_idx):
        loss = self.forward(batch[0])
        self.log("train_reconstruction_loss", loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.forward(batch[0])
        self.log("val_reconstruction_loss", loss)

    def test_step(self, batch, batch_idx):
        loss = self.forward(batch[0])
        self.log("test_reconstruction_loss", loss)


class ReconstructionCallback(pl.Callback):
    """
    Logs original / masked / reconstructed samples as a simple training sanity check.
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
        recon, keep_mask = pl_module.reconstruct(imgs)

        original = imgs.float()
        grey = 0.0
        noise = (0.0 if not pl_module.hparams.denoise else torch.randn_like(imgs).clamp_(-1.0, 1.0))
        masked = torch.where(keep_mask, imgs + noise, grey).float()
        reconstructed = recon.float()

        panel = torch.cat([original, masked, reconstructed], dim=0).detach().cpu()
        grid = make_grid(panel, nrow=self.num_images, normalize=True, pad_value=grey, value_range=(-1.0, 1.0))
        epoch = trainer.current_epoch
        is_sanity = trainer.sanity_checking
        log_step = epoch + (0 if is_sanity else 1)
        trainer.logger.experiment.add_image("MAE/original_masked_recon", grid, global_step=log_step)

        if self.save_to_disk:
            image_name = f"epoch_{epoch}_recon.png" if not is_sanity else "pre_epoch_0_recon.png"
            save_image(grid, os.path.join(trainer.logger.log_dir, image_name))


def train_mae(args):
    os.makedirs(args.log_dir, exist_ok=True)
    train_loader, val_loader, test_loader = mnist(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        root=args.data_dir,
    )

    recon_callback = ReconstructionCallback(save_to_disk=True)
    save_callback = ModelCheckpoint(save_weights_only=True, mode="min", monitor="val_reconstruction_loss")
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
    
    # Define model with the specified hyperparameters
    model = MAE(
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
        **masking_kwargs_from_args(args),
    )

    trainer.fit(model, train_loader, val_loader)

    model = MAE.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)
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
    parser.add_argument("--masked_loss_weight", default=4.0, type=float, help="Extra weight for masked pixels in MSE.")
    
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
    parser.add_argument("--no_skip", action="store_false", 
                        help="Whether to use skip connections in the decoder.")
    parser.set_defaults(no_skip=False)
    
    # Logging / other params
    parser.add_argument("--data_dir", default="../data/", type=str, help="Directory where to look for the data.")
    parser.add_argument("--num_workers",default=10,type=int,
                        help=("Number of workers to use in data loaders. For strict determinism set this to 0."),)
    parser.add_argument("--log_dir", default="MAE_logs", type=str, help="Directory for PyTorch Lightning logs.")
    parser.add_argument("--progress_bar",action="store_true",
        help=("Use a progress bar indicator for interactive experimentation. Not to be used with SLURM jobs."),)

    args = parser.parse_args()
    train_mae(args)
