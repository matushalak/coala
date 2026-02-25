import argparse
import os

import torch
import torch.nn.functional as F
from torchvision.utils import make_grid, save_image
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from cc.datasets.mnist import mnist
from cc.masked_ae.cnn_unet import CNNEncoder, CNNDecoder
from cc.masked_ae.utils import visualize_manifold

class AE(pl.LightningModule):

    def __init__(self, num_filters, z_dim, lr, num_input_channels=1):
        """
        PyTorch Lightning module that summarizes all components to train a VAE.
        Inputs:
            num_filters - Number of channels to use in a CNN encoder/decoder
            z_dim - Dimensionality of latent space
            lr - Learning rate to use for the optimizer
        """
        super().__init__()
        self.save_hyperparameters()
        self.zdim = z_dim
        self.encoder = CNNEncoder(z_dim=z_dim, num_filters=num_filters, num_input_channels=num_input_channels)
        self.decoder = CNNDecoder(z_dim=z_dim, num_filters=num_filters, num_input_channels=num_input_channels)

    def forward(self, imgs):
        """
        The forward function calculates the AE-loss for a given batch of images.
        Inputs:
            imgs - Batch of normalized images of shape [B,C,H,W].
        Ouptuts:
            L_rec - The average reconstruction loss of the batch. Shape: single scalar
        """
        # obtain latent representation from encoder
        z = self.encoder(imgs)
        # obtain reconstruction from decoder
        recon = self.decoder(z)
        L_rec = F.mse_loss(recon, imgs)
        return L_rec

    @torch.no_grad()
    def sample(self, batch_size):
        """
        Function for sampling a new batch of random images.
        Inputs:
            batch_size - Number of images to generate
        Outputs:
            x_samples - Sampled reconstructed images. Shape: [B,C,H,W]
        """
        # sample from standard normal
        z_samples = torch.randn((batch_size, self.zdim), device=self.decoder.device)
        # pass through decoder to obtain reconstructed samples
        x_samples = self.decoder(z_samples)
        return x_samples
    
    @torch.no_grad()
    def reconstruct_samples(self, imgs):
        z = self.encoder(imgs)
        return self.decoder(z)

    def configure_optimizers(self):
        # Create optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        return optimizer

    def training_step(self, batch, batch_idx):
        # Make use of the forward function, and add logging statements
        L_rec = self.forward(batch[0])
        self.log("train_reconstruction_loss", L_rec, on_step=False, on_epoch=True)
        return L_rec

    def validation_step(self, batch, batch_idx):
        # Make use of the forward function, and add logging statements
        L_rec = self.forward(batch[0])
        self.log("val_reconstruction_loss", L_rec)

    def test_step(self, batch, batch_idx):
        # Make use of the forward function, and add logging statements
        L_rec = self.forward(batch[0])
        self.log("test_reconstruction_loss", L_rec)


class GenerateCallback(pl.Callback):

    def __init__(self, batch_size=64, every_n_epochs=5, save_to_disk=False):
        """
        Inputs:
            batch_size - Number of images to generate
            every_n_epochs - Only save those images every N epochs (otherwise tensorboard gets quite large)
            save_to_disk - If True, the samples and image means should be saved to disk as well.
        """
        super().__init__()
        self.batch_size = batch_size
        self.every_n_epochs = every_n_epochs
        self.save_to_disk = save_to_disk
        self._val_examples = None

    def on_train_epoch_end(self, trainer, pl_module):
        """
        This function is called after every epoch.
        Call the save_and_sample function every N epochs.
        """
        if (trainer.current_epoch+1) % self.every_n_epochs == 0:
            self.sample_and_save(trainer, pl_module, trainer.current_epoch+1)
            self.reconstruct_and_save(trainer, pl_module, trainer.current_epoch+1)

    def sample_and_save(self, trainer, pl_module, epoch):
        """
        Function that generates and save samples from the VAE.
        The generated sample images should be added to TensorBoard and,
        if self.save_to_disk is True, saved inside the logging directory.
        Inputs:
            trainer - The PyTorch Lightning "Trainer" object.
            pl_module - The VAE model that is currently being trained.
            epoch - The epoch number to use for TensorBoard logging and saving of the files.
        """
        samples = pl_module.sample(self.batch_size)
        grid = make_grid(samples, nrow=8, normalize=True, pad_value=0.5)
        grid = grid.detach().cpu()
        trainer.logger.experiment.add_image("Samples", grid, global_step=epoch)
        if self.save_to_disk:
            save_image(grid,
                        os.path.join(trainer.logger.log_dir, f"epoch_{epoch}_samples.png"))

    def _get_validation_examples(self, trainer):
        if self._val_examples is not None:
            return self._val_examples

        val_loaders = trainer.val_dataloaders
        if val_loaders is None:
            return None
        val_loader = val_loaders[0] if isinstance(val_loaders, (list, tuple)) else val_loaders
        if val_loader is None:
            return None

        try:
            batch = next(iter(val_loader))
        except StopIteration:
            return None

        imgs = batch[0] if isinstance(batch, (tuple, list)) else batch
        self._val_examples = imgs[:self.batch_size].detach().cpu()
        return self._val_examples

    @torch.no_grad()
    def reconstruct_and_save(self, trainer, pl_module, epoch):
        imgs = self._get_validation_examples(trainer)
        if imgs is None or imgs.numel() == 0:
            return

        imgs = imgs.to(pl_module.device)
        recon = pl_module.reconstruct_samples(imgs)

        originals = imgs.float()
        reconstructions = recon.float()
        paired = torch.stack((originals, reconstructions), dim=1).flatten(0, 1).detach().cpu()
        nrow = max(2, 2 * min(8, originals.shape[0]))
        grid = make_grid(paired, nrow=nrow, normalize=True, pad_value=0.5)

        trainer.logger.experiment.add_image("Reconstructions", grid, global_step=epoch)
        if self.save_to_disk:
            save_image(grid,
                        os.path.join(trainer.logger.log_dir, f"epoch_{epoch}_reconstructions.png"))


def train_ae(args):
    """
    Function for training and testing a VAE model.
    Inputs:
        args - Namespace object from the argument parser
    """

    os.makedirs(args.log_dir, exist_ok=True)
    train_loader, val_loader, test_loader = mnist(batch_size=args.batch_size,
                                                   num_workers=args.num_workers,
                                                   root=args.data_dir)

    # Create a PyTorch Lightning trainer with the generation callback
    gen_callback = GenerateCallback(save_to_disk=True)
    save_callback = ModelCheckpoint(save_weights_only=True, mode="min", monitor="val_reconstruction_loss")
    trainer = pl.Trainer(default_root_dir=args.log_dir,
                         accelerator="auto",
                         max_epochs=args.epochs,
                         callbacks=[save_callback, gen_callback],
                         enable_progress_bar=args.progress_bar)
    trainer.logger._default_hp_metric = None  # Optional logging argument that we don't need
    if not args.progress_bar:
        print("[INFO] The progress bar has been suppressed. For updates on the training " + \
              f"progress, check the TensorBoard file at {trainer.logger.log_dir}. If you " + \
              "want to see the progress bar, use the argparse option \"progress_bar\".\n")

    # Create model
    pl.seed_everything(args.seed)  # To be reproducible
    model = AE(num_filters=args.num_filters,
                z_dim=args.z_dim,
                lr=args.lr,
                num_input_channels=args.num_input_channels)

    # Training
    gen_callback.sample_and_save(trainer, model, epoch=0)  # Initial sample
    gen_callback.reconstruct_and_save(trainer, model, epoch=0)  # Initial sample
    trainer.fit(model, train_loader, val_loader)

    # Testing
    model = AE.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)
    test_result = trainer.test(model, dataloaders=test_loader, verbose=True)

    # Manifold generation
    if args.z_dim == 2:
        img_grid = visualize_manifold(model.decoder)
        save_image(img_grid,
                   os.path.join(trainer.logger.log_dir, 'ae_manifold.png'),
                   normalize=False)

    return test_result


if __name__ == '__main__':
    # Feel free to add more argument parameters
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Model hyperparameters
    parser.add_argument('--z_dim', default=20, type=int,
                        help='Dimensionality of latent space')
    parser.add_argument('--num_filters', default=32, type=int,
                        help='Number of channels/filters to use in the CNN encoder/decoder.')
    parser.add_argument('--num_input_channels', default=1, type=int,
                        help='Number of image channels to reconstruct (1 for MNIST/FashionMNIST, 3 for CIFAR/SVHN).')

    # Optimizer hyperparameters
    parser.add_argument('--lr', default=1e-3, type=float,
                        help='Learning rate to use')
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Minibatch size')

    # Other hyperparameters
    parser.add_argument('--data_dir', default='../data/', type=str,
                        help='Directory where to look for the data. For jobs on Lisa, this should be $TMPDIR.')
    parser.add_argument('--epochs', default=21, type=int,
                        help='Max number of epochs')
    parser.add_argument('--seed', default=42, type=int,
                        help='Seed to use for reproducing results')
    parser.add_argument('--num_workers', default=10, type=int,
                        help='Number of workers to use in the data loaders. To have a truly deterministic run, this has to be 0. ' + \
                             'For your assignment report, you can use multiple workers (e.g. 4) and do not have to set it to 0.')
    parser.add_argument('--log_dir', default='AE_logs', type=str,
                        help='Directory where the PyTorch Lightning logs should be created.')
    parser.add_argument('--progress_bar', action='store_true',
                        help=('Use a progress bar indicator for interactive experimentation. '
                              'Not to be used in conjuction with SLURM jobs'))

    args = parser.parse_args()

    train_ae(args)
