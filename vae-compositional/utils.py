################################################################################
# MIT License
#
# Copyright (c) 2022
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course | Autumn 2022
# Date Created: 2022-11-25
################################################################################

import torch
import torch.nn.functional as F
from torchvision.utils import make_grid
import numpy as np


def sample_reparameterize(mean, std):
    """
    Perform the reparameterization trick to sample from a distribution with the given mean and std
    Inputs:
        mean - Tensor of arbitrary shape and range, denoting the mean of the distributions
        std - Tensor of arbitrary shape with strictly positive values. Denotes the standard deviation
              of the distribution
    Outputs:
        z - A sample of the distributions, with gradient support for both mean and std.
            The tensor should have the same shape as the mean and std input tensors.
    """
    assert not (std < 0).any().item(), "The reparameterization trick got a negative std as input. " + \
                                       "Are you sure your input is std and not log_std?"
    # reparametrization trick: z = mu + sigma * eps     with eps ~ N(0,I)
    z = mean + (std * torch.randn(mean.shape, device=mean.device))
    return z


def KLD(mean, log_std):
    """
    Calculates the Kullback-Leibler divergence of given distributions to unit Gaussians over the last dimension.
    See the definition of the regularization loss in Section 1.4 for the formula.
    Inputs:
        mean - Tensor of arbitrary shape and range, denoting the mean of the distributions.
        log_std - Tensor of arbitrary shape and range, denoting the log standard deviation of the distributions.
    Outputs:
        KLD - Tensor with one less dimension than mean and log_std (summed over last dimension).
              The values represent the Kullback-Leibler divergence to unit Gaussians.
    """
    # KLD of 2 multivar gaussians expressed as sum of univar gaussians
    # solely with mean and log_std along each dimension
    KLD:torch.Tensor = 0.5 * (torch.exp(2*log_std) + torch.pow(mean, 2) - 1 - (2*log_std))
    return torch.sum(KLD,dim = -1)


def elbo_to_bpd(elbo, img_shape):
    """
    Converts the summed negative log likelihood given by the ELBO into the bits per dimension score.
    Inputs:
        elbo - Tensor of shape [batch_size]
        img_shape - Shape of the input images, representing [batch, channels, height, width]
    Outputs:
        bpd - The negative log likelihood in bits per dimension for the given image.
    """
    # change log base from e to 2
    nll = elbo * torch.log2(torch.tensor(torch.e, device=elbo.device))
    # exclude first batch dim
    image_dims = torch.tensor(img_shape[1:], device = elbo.device)
    # bits per dimension score
    bpd = nll * (1 / torch.prod(image_dims, dim=0))
    return bpd

@torch.no_grad()
def visualize_manifold(decoder, grid_size=16):
    """
    Visualize a manifold over a 2 dimensional latent space. The images in the manifold
    should represent the decoder's output means (not binarized samples of those).
    Inputs:
        decoder - Decoder model such as LinearDecoder or ConvolutionalDecoder.
        grid_size - Number of steps/images to have per axis in the manifold.
                    Overall you need to generate grid_size**2 images, and the distance
                    between different latents in percentiles is 1/grid_size
    Outputs:
        img_grid - Grid of images representing the manifold.
    """
    # percentile range with significant density
    percentiles = torch.linspace(0.01/grid_size, (grid_size-0.01)/grid_size, grid_size)
    standard_normal = torch.distributions.Normal(loc=0, scale=1)
    zvals = standard_normal.icdf(percentiles)
    z1grid, z2grid = torch.meshgrid(zvals, zvals, indexing='ij')
    # reformat to collapse grid to one batch and match latent dimensionality
    zgrid = torch.stack([z1grid.flatten(), z2grid.flatten()], dim=1) # (grid**2, 2)
    # can pass to decoder (B=grid**2, zdim = 2)
    logits = decoder(zgrid) # (grid**2, 2) => (grid**2, 16, 28, 28)
    probs = torch.softmax(logits, dim = 1)
    # compute output means
    values = torch.arange(16)
    output_means = probs * values[None, :, None, None]
    output_means = output_means.sum(dim = 1).float().unsqueeze(1)
    # make grid with torchvision make_grid
    img_grid = make_grid(output_means, nrow=grid_size, padding=0, normalize=True, value_range=(0,1))

    return img_grid

@torch.no_grad()
def visualize_3d_manifold(decoder, grid_size=25):
    """
    Visualize a manifold over a 3 dimensional latent space. The images in the manifold
    should represent the decoder's output means (not binarized samples of those).
    Inputs:
        decoder - Decoder model such as LinearDecoder or ConvolutionalDecoder.
        grid_size - Number of steps/images to have per axis in the manifold.
                    Overall you need to generate grid_size**3 images, and the distance
                    between different latents in percentiles is 1/grid_size
    Outputs:
        img_grid - Grid of images representing the manifold.
    """
    # percentile range with significant density
    percentiles = torch.linspace(0.01/grid_size, (grid_size-0.01)/grid_size, grid_size)
    standard_normal = torch.distributions.Normal(loc=0, scale=1)
    zvals = standard_normal.icdf(percentiles)
    z1grid, z2grid = torch.meshgrid(zvals, zvals, indexing='ij')

    # Build latent batch as 9 z3-slices; each slice is a (grid_size x grid_size) manifold over z1/z2.
    slice_latents = []
    for z3 in zvals:
        z_slice = torch.stack(
            [z1grid.flatten(), z2grid.flatten(), torch.full_like(z1grid.flatten(), z3)],
            dim=1
        )
        slice_latents.append(z_slice)
    zgrid = torch.cat(slice_latents, dim=0)  # (grid_size**3, 3)

    # can pass to decoder (B=grid_size**3, zdim=3)
    logits = decoder(zgrid)
    probs = torch.softmax(logits, dim = 1)
    # compute output means
    values = torch.arange(16, device=probs.device)
    output_means = probs * values[None, :, None, None]
    output_means = output_means.sum(dim = 1).float().unsqueeze(1)
    output_means /= 15

    # First, make one 2D manifold image per z3-slice (9x9 each by default).
    slice_grids = []
    imgs_per_slice = grid_size * grid_size
    for i in range(grid_size):
        start = i * imgs_per_slice
        end = (i + 1) * imgs_per_slice
        slice_grid = make_grid(
            output_means[start:end],
            nrow=grid_size,
            padding=0,
            normalize=True,
            value_range=(0, 1)
        )
        slice_grids.append(slice_grid)
    slice_grids = torch.stack(slice_grids, dim=0)

    # Arrange the z3-slices as a 3x3 panel (for grid_size=9).
    panel_cols = int(np.sqrt(grid_size))
    if panel_cols * panel_cols != grid_size:
        panel_cols = grid_size
    img_grid = make_grid(slice_grids, nrow=panel_cols, padding=4, normalize=False)

    return img_grid

@torch.no_grad()
def visualize_reconstructions(model, data_loader, n_images=12):
    """
    Visualizes reconstructions of the model on a batch of images from the data_loader.
    Inputs:
        model - VAE model with encoder and decoder, such as VAE or CVAE class in train_pl.py
        data_loader - DataLoader to sample the images to reconstruct from. The images should be
                      normalized to [0,1] and have shape [B,C,H,W].
        n_images - Number of images to visualize in the manifold. Should be less than batch size
    """
    batch = next(iter(data_loader))
    images = batch[0][:n_images].to(model.device) # (B,C,H,W)
    b, c, h, w = images.shape
    # obtain variational distribution parameters (mu, log(std))
    mean, logstd = model.encoder(images)
    # sample in latent space using mu, std
    z = sample_reparameterize(mean, torch.exp(logstd))
    # obtain logits from decoder
    logits = model.decoder(z)

    # reconstruction loss: obtain negative log likelihood by summing over CE over all pixels and possible values
    # sum over pixels and possible values, but not over batch dimension, to get NLL per image
    L_rec = F.cross_entropy(logits.view(b, -1, h*w), images.view(b, h*w), reduction= 'none').sum(axis = -1)
    
    # convert logits to probabilities and then to 4-bit images
    probs = torch.softmax(logits, dim=1)
    values = torch.arange(16)
    output_means = probs * values[None, :, None, None]
    output_means = output_means.sum(dim=1).float().unsqueeze(1) / 15.0  # Normalize to [0,1]
    
    original_images = images.float()
    if original_images.max() > 1:
        original_images = original_images / 15.0

    paired_images = torch.empty(
        (2 * b, output_means.shape[1], h, w),
        device=output_means.device,
        dtype=output_means.dtype
    )
    paired_images[0::2] = original_images
    paired_images[1::2] = output_means
    img_grid = make_grid(paired_images, nrow=2, padding=2, normalize=False)

    return img_grid

def map_high_dimensional_latent_reconstructions_to_Nd(model, test_loader, Nd:int = 3):
    from tqdm import tqdm
    latents = []
    labels = []

    # Collect latents and labels for the whole test set
    for i, (imgs, (digit_lbls, color_lbls)) in enumerate(tqdm(test_loader, desc="Collecting latents and labels")):
        if i % 500 == 0:
            continue  # Skip some batches to reduce computation for large test sets
            
        images = imgs.to(model.device) # (B,C,H,W)
        b, c, h, w = images.shape
        # obtain variational distribution parameters (mu, log(std))
        mean, logstd = model.encoder(images)
        # sample in latent space using mu, std
        z = sample_reparameterize(mean, torch.exp(logstd))
        latents.append(z)
        labels.append(torch.cat([digit_lbls[:, None], color_lbls[:, None]], dim=1))  # (B, 2) with digit and color labels
    
    # Turn into big tensors
    latents = torch.cat(latents, dim=0).detach().cpu().numpy()
    labels = torch.cat(labels, dim=0).detach().cpu().numpy()

    # Turn 2d labels into a single label for coloring (e.g., digit*10 + color)
    labels = labels[:, 0] * 10 + labels[:, 1]

    # Use PCA to reduce to Nd dimensions
    from sklearn.decomposition import PCA
    pca = PCA(n_components=Nd)
    latents_Nd = pca.fit_transform(latents)
    
    if Nd == 2:
        # For 2D, we can visualize the latent space with a scatter plot
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(latents_Nd[:, 0], latents_Nd[:, 1], c=labels, cmap='viridis', alpha=0.7)
        plt.colorbar(scatter, ticks=np.unique(labels))
        plt.title('2D PCA of Latent Space')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.grid()
        plt.show()
    
    if Nd == 3:
        # For 3D, we can visualize the latent space with a 3D scatter plot
        from mpl_toolkits.mplot3d import Axes3D
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(latents_Nd[:, 0], latents_Nd[:, 1], latents_Nd[:, 2], c=labels, cmap='viridis', alpha=0.7)
        fig.colorbar(scatter, ticks=np.unique(labels))
        ax.set_title('3D PCA of Latent Space')
        ax.set_xlabel('Principal Component 1')
        ax.set_ylabel('Principal Component 2')
        ax.set_zlabel('Principal Component 3')
        plt.show()
    
    return latents_Nd, labels