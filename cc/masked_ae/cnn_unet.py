# Simple U-Net implementation
import torch.nn as nn

class CNNEncoder(nn.Module):
    def __init__(self, 
                 num_input_channels: int = 1, 
                 num_filters: int = 32,
                 z_dim: int = 20):
        """Encoder with a CNN network
        Inputs:
            num_input_channels - Number of input channels of the image. For
                                 MNIST, this parameter is 1
            num_filters - Number of channels we use in the first convolutional
                          layers. Deeper layers might use a duplicate of it.
            z_dim - Dimensionality of latent representation z
        """
        super().__init__()
        
        # latent dimension
        self.z_dim = z_dim
        # Architecture adapted to 28x28 MNIST
        # + added LayerNorm before nonlinearities
        # + added residual connections around Conv blocks that dont change shape
        self.encoder = nn.Sequential(
            # downsample res
            nn.Conv2d(num_input_channels, num_filters, kernel_size=3, padding=1, stride=2), # 28x28 => 14x14
            nn.LayerNorm((num_filters, 14, 14)),
            nn.GELU(),
            # same res conv + residual connection
            ResidualConv2d(n_channels=num_filters, spatial_dim=(14,14), kernel_size=3), # 14x14 => 14x14
            nn.LayerNorm((num_filters, 14, 14)),
            nn.GELU(),
            # downsample res
            nn.Conv2d(num_filters, 2*num_filters, kernel_size=3, padding=1, stride=2), # 14x14 => 7x7
            nn.LayerNorm((2*num_filters, 7, 7)),
            nn.GELU(),
            # same res conv + residual connection
            ResidualConv2d(n_channels=2*num_filters, spatial_dim=(7,7), kernel_size=3), # 7x7 => 7x7
            nn.LayerNorm((2*num_filters, 7, 7)),
            nn.GELU(),
            # downsample res
            nn.Conv2d(2*num_filters, 2*num_filters, kernel_size=3, padding=1, stride=2), # 7x7 => 4x4
            nn.LayerNorm((2*num_filters, 4, 4)),
            nn.GELU(),
            nn.Flatten(), # Image grid to single feature vector 4x4 => (16,)
            nn.Linear(16*2*num_filters, z_dim)
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
        x = x.float() / 15 * 2.0 - 1.0  # Move images between -1 and 1
        # run batch of images through network
        latent_params = self.encoder(x) # (B, z_dim)
        mean = latent_params[..., :self.z_dim] # (B, z_dim)
        return mean

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
                 z_dim: int = 20):
        """Decoder with a CNN network.
        Inputs:
            num_input_channels - Number of channels of the image to
                                 reconstruct. For a 4-bit MNIST, this parameter is 16
            num_filters - Number of filters we use in the last convolutional
                          layers. Early layers might use a duplicate of it.
            z_dim - Dimensionality of latent representation z
        """
        super().__init__()

        self.expand = nn.Linear(z_dim, 16*2*num_filters) # (20,) => (16*2*32,) => (2*32, 4, 4)
        # Architecture from Tutorial 9 adapted to 28x28 MNIST
        # + added LayerNorm before nonlinearities
        # + added residual connections around Conv blocks that dont change shape
        self.decoder = nn.Sequential(
            nn.LayerNorm((2*num_filters, 4, 4)),
            nn.GELU(),
            # upsample with transposed conv
            nn.ConvTranspose2d(2*num_filters, 2*num_filters, kernel_size=3, output_padding=0, padding=1, stride=2), # 4x4 => 7x7
            nn.LayerNorm((2*num_filters, 7, 7)),
            nn.GELU(),
            # dont change shape - normal conv with residual connection
            ResidualConv2d(n_channels=2*num_filters, spatial_dim=(7,7), kernel_size=3), # 7x7 => 7x7
            nn.LayerNorm((2*num_filters, 7, 7)),
            nn.GELU(),
            # upsample with transposed conv
            nn.ConvTranspose2d(2*num_filters, num_filters, kernel_size=3, output_padding=1, padding=1, stride=2), # 7x7 => 14x14
            nn.LayerNorm((num_filters, 14, 14)),
            nn.GELU(),
            # dont change shape - normal conv with residual connection
            ResidualConv2d(n_channels=num_filters, spatial_dim=(14,14), kernel_size=3), # 14x14 => 14x14
            nn.LayerNorm((num_filters, 14, 14)),
            nn.GELU(),
            # upsample with transposed conv
            nn.ConvTranspose2d(num_filters, num_input_channels, kernel_size=3, output_padding=1, padding=1, stride=2), # 14x14 => 28x28
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
        x = self.expand(z) # (B, 20) => (B, 16*2*32)
        x = x.reshape(x.shape[0], -1, 4, 4) # (B, 16*2*32) => (B, 2*32, 4, 4)
        # get logits for all possible 4-bit values for each pixel in x
        # by passing (reshaped) latents (z) through decoder
        x = self.decoder(x) # (B, 2*32, 4, 4) => (B, 16, 28, 28)
        return x

    @property
    def device(self):
        """
        Property function to get the device on which the decoder is.
        Might be helpful in other functions.
        """
        return next(self.parameters()).device
