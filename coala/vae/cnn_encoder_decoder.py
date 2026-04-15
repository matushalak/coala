import torch
import torch.nn as nn
import numpy as np


def _downsample_dim(size: int) -> int:
    return (int(size) + 1) // 2


def _encoder_spatial_dims(image_size: int) -> tuple[int, int, int]:
    dim1 = _downsample_dim(image_size)
    dim2 = _downsample_dim(dim1)
    dim3 = _downsample_dim(dim2)
    return dim1, dim2, dim3


def _output_padding(input_dim: int, target_dim: int) -> int:
    output_padding = int(target_dim) - (2 * int(input_dim) - 1)
    if output_padding not in {0, 1}:
        raise ValueError(
            f"Cannot reach target_dim={target_dim} from input_dim={input_dim} with the current ConvTranspose2d setup."
        )
    return output_padding


class CNNEncoder(nn.Module):
    def __init__(self, 
                 num_input_channels: int = 1, 
                 num_filters: int = 32,
                 z_dim: int = 20,
                 num_values: int = 16,
                 image_size: int = 28):
        """Encoder with a CNN network
        Inputs:
            num_input_channels - Number of input channels of the image. For
                                 MNIST, this parameter is 1
            num_filters - Number of channels we use in the first convolutional
                          layers. Deeper layers might use a duplicate of it.
            z_dim - Dimensionality of latent representation z
        """
        super().__init__()
        self.num_values = num_values
        self.image_size = int(image_size)
        # latent dimension
        self.z_dim = z_dim
        dim1, dim2, dim3 = _encoder_spatial_dims(self.image_size)
        # Architecture from Tutorial 9 adapted to 28x28 MNIST
        # + added LayerNorm before nonlinearities
        # + added residual connections around Conv blocks that dont change shape
        self.encoder = nn.Sequential(
            # downsample res
            nn.Conv2d(num_input_channels, num_filters, kernel_size=3, padding=1, stride=2),
            nn.LayerNorm((num_filters, dim1, dim1)),
            nn.GELU(),
            # same res conv + residual connection
            ResidualConv2d(n_channels=num_filters, spatial_dim=(dim1, dim1), kernel_size=3),
            nn.LayerNorm((num_filters, dim1, dim1)),
            nn.GELU(),
            # downsample res
            nn.Conv2d(num_filters, 2*num_filters, kernel_size=3, padding=1, stride=2),
            nn.LayerNorm((2*num_filters, dim2, dim2)),
            nn.GELU(),
            # same res conv + residual connection
            ResidualConv2d(n_channels=2*num_filters, spatial_dim=(dim2, dim2), kernel_size=3),
            nn.LayerNorm((2*num_filters, dim2, dim2)),
            nn.GELU(),
            # downsample res
            nn.Conv2d(2*num_filters, 4*num_filters, kernel_size=3, padding=1, stride=2),
            nn.LayerNorm((4*num_filters, dim3, dim3)),
            nn.GELU(),
            nn.Flatten(),
            # want to predict both mean and log(std), learn one linear layer for both
            # log(std) \in R^D can be obtained with normal linear layer, and converted to
            # std \in R_+^D using exp(log(std))
            nn.Linear(dim3 * dim3 * 4 * num_filters, z_dim * 2)
        )

    def forward(self, x):
        """
        Inputs:
            x - Input batch with images of shape [B,C,H,W] of type long with values between 0 and 15.
        Outputs:
            mean - Tensor of shape [B,z_dim] representing the predicted mean of the latent distributions.
            log_std - Tensor of shape [B,z_dim] representing the predicted log standard deviation
                      of the latent distributions.
        """
        x = x.float() / (self.num_values-1) * 2.0 - 1.0  # Move images between -1 and 1
        # run batch of images through network
        latent_params = self.encoder(x) # (B, 2*z_dim)
        mean = latent_params[..., :self.z_dim] # (B, z_dim)
        log_std = latent_params[..., self.z_dim:] # (B, z_dim)
        return mean, log_std

class ResidualConv2d(nn.Module):
    def __init__(self, 
                 n_channels: int,
                 spatial_dim: tuple[int, int],
                 kernel_size: int):
        '''
        Convolution block that doesn't change channel/spatial dimensions 
        with residual connection around it
        '''
        super().__init__()

        # Keep same shape just pass through a series of convolutions
        self.resblock = nn.Sequential(
            nn.Conv2d(n_channels, n_channels, kernel_size=kernel_size, padding='same', stride = 1),
            nn.LayerNorm((n_channels, *spatial_dim)),
            nn.GELU(),
            nn.Conv2d(n_channels, n_channels, kernel_size=kernel_size, padding='same', stride = 1),
        )
    def forward(self, x):
        # Residual connection
        return x + self.resblock(x)

class CNNDecoder(nn.Module):
    def __init__(self, 
                 num_input_channels: int = 16, 
                 num_filters: int = 32,
                 z_dim: int = 20,
                 image_size: int = 28):
        """Decoder with a CNN network.
        Inputs:
            num_input_channels - Number of channels of the image to
                                 reconstruct. For a 4-bit MNIST, this parameter is 16
            num_filters - Number of filters we use in the last convolutional
                          layers. Early layers might use a duplicate of it.
            z_dim - Dimensionality of latent representation z
        """
        super().__init__()
        self.image_size = int(image_size)
        dim1, dim2, dim3 = _encoder_spatial_dims(self.image_size)
        output_padding_1 = _output_padding(dim3, dim2)
        output_padding_2 = _output_padding(dim2, dim1)
        output_padding_3 = _output_padding(dim1, self.image_size)

        self.expand = nn.Linear(z_dim, dim3 * dim3 * 4 * num_filters)
        # Architecture from Tutorial 9 adapted to 28x28 MNIST
        # + added LayerNorm before nonlinearities
        # + added residual connections around Conv blocks that dont change shape
        self.decoder = nn.Sequential(
            nn.LayerNorm((4*num_filters, dim3, dim3)),
            nn.GELU(),
            # upsample with transposed conv
            nn.ConvTranspose2d(4*num_filters, 2*num_filters, kernel_size=3, output_padding=output_padding_1, padding=1, stride=2),
            nn.LayerNorm((2*num_filters, dim2, dim2)),
            nn.GELU(),
            # dont change shape - normal conv with residual connection
            ResidualConv2d(n_channels=2*num_filters, spatial_dim=(dim2, dim2), kernel_size=3),
            nn.LayerNorm((2*num_filters, dim2, dim2)),
            nn.GELU(),
            # upsample with transposed conv
            nn.ConvTranspose2d(2*num_filters, num_filters, kernel_size=3, output_padding=output_padding_2, padding=1, stride=2),
            nn.LayerNorm((num_filters, dim1, dim1)),
            nn.GELU(),
            # dont change shape - normal conv with residual connection
            ResidualConv2d(n_channels=num_filters, spatial_dim=(dim1, dim1), kernel_size=3),
            nn.LayerNorm((num_filters, dim1, dim1)),
            nn.GELU(),
            # upsample with transposed conv
            nn.ConvTranspose2d(num_filters, num_input_channels, kernel_size=3, output_padding=output_padding_3, padding=1, stride=2),
        )

    def forward(self, z):
        """
        Inputs:
            z - Latent vector of shape [B,z_dim]
        Outputs:
            x - Prediction of the reconstructed image based on z.
                This should be a logit output *without* a softmax applied on it.
                Shape: [B,num_input_channels,28,28]
        """
        _, _, dim3 = _encoder_spatial_dims(self.image_size)
        x = self.expand(z)
        x = x.reshape(x.shape[0], -1, dim3, dim3)
        # get logits for all possible 4-bit values for each pixel in x
        # by passing (reshaped) latents (z) through decoder
        x = self.decoder(x)
        return x

    @property
    def device(self):
        """
        Property function to get the device on which the decoder is.
        Might be helpful in other functions.
        """
        return next(self.parameters()).device
