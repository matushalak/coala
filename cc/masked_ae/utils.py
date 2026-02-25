import torch
from torchvision.utils import make_grid

@torch.no_grad()
def visualize_manifold(decoder, grid_size=20):
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
    percentiles = torch.linspace(0.5/grid_size, (grid_size-0.5)/grid_size, grid_size)
    standard_normal = torch.distributions.Normal(loc=0, scale=1)
    zvals = standard_normal.icdf(percentiles)
    z1grid, z2grid = torch.meshgrid(zvals, zvals, indexing='ij')
    # reformat to collapse grid to one batch and match latent dimensionality
    zgrid = torch.stack([z1grid.flatten(), z2grid.flatten()], dim=1) # (grid**2, 2)
    # can pass to decoder (B=grid**2, zdim = 2)
    output_means = decoder(zgrid).float()
    # make grid with torchvision make_grid
    img_grid = make_grid(output_means, nrow=grid_size, padding=0, normalize=True)

    return img_grid
