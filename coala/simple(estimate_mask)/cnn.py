'''
Learned-Sparse CNN Autoencoder
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

class EncoderLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(EncoderLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.up_mix = nn.Linear(out_channels, 2*out_channels)
        self.down_mix = nn.Linear(2*out_channels, out_channels)
        self.relu = nn.ReLU()

    def forward(self, x:torch.Tensor, mask:torch.Tensor):
        x = self.conv(x) * mask # (B, C, H, W)
        x = self.relu(x)
        y = self.up_mix(x.permute(0, 2, 3, 1)) * mask.permute(0, 2, 3, 1)  # (B, H, W, C)
        y = self.relu(y)
        y = self.down_mix(y) * mask.permute(0, 2, 3, 1)  # (B, H, W, C)
        return x + y.permute(0, 2, 3, 1)  # (B, C, H, W)
    
class DecoderLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(DecoderLayer, self).__init__()
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding)
        self.up_mix = nn.Linear(out_channels, 2*out_channels)
        self.down_mix = nn.Linear(2*out_channels, out_channels)
        self.relu = nn.ReLU()
        self.mask_token = nn.Parameter(torch.zeros(1, out_channels, 1, 1))  # (1, C, 1, 1)

    def forward(self, x:torch.Tensor, mask:torch.Tensor, skip:torch.Tensor):
        x = self.conv(x) # (B, C, H, W)
        x = self.relu(x)
        skip = (mask * skip) + ((1 - mask) * self.mask_token)  # (B, C, H, W)
        x += skip 
        y = self.up_mix(x.permute(0, 2, 3, 1))  # (B, H, W, C)
        y = self.relu(y)
        y = self.down_mix(y)  # (B, H, W, C)
        return x + y.permute(0, 2, 3, 1)  # (B, C, H, W)
    
class PredictorLayer(nn.Module):
    def __init__(self, latent_dims:list[int]):
        super(PredictorLayer, self).__init__()
        # TODO
        # do "unpooling" to broadcast low resolution to high resolution
        # Stack hierarchical encoder layers across channel dimension
        # perform self attention across spatial dimension
        # followed by MLP to predict the occluded regions of image

    def forward(self, latents:list[torch.Tensor])->torch.Tensor:
        raise NotImplementedError()
    
class Net(nn.Module):
    def __init__(self, encoder_dims:list[int], decoder_dims:list[int]):
        super(Net, self).__init__()
        self.encoders = nn.ModuleList([EncoderLayer(encoder_dims[i], encoder_dims[i+1]) for i in range(len(encoder_dims)-1)])
        self.decoders = nn.ModuleList([DecoderLayer(decoder_dims[i], decoder_dims[i+1]) for i in range(len(decoder_dims)-1)])
        self.predictor = PredictorLayer(encoder_dims)

    def forward(self, x:torch.Tensor, n_iter:int=5):
        mask = torch.ones_like(x)  # (B, C, H, W)
        outputs = []
        # Recursive inference loop
        for _ in range(n_iter):
            skips = []
            # Encoder
            for encoder in self.encoders:
                x = encoder(x, mask)
                skips.append(x)
            # Predictor
            mask = self.predictor(skips)  # (B, C, H, W)
            # Decoder
            for decoder in self.decoders:
                skip = skips.pop()
                x = decoder(x, mask, skip)
            outputs.append(x)
        return outputs  # list of (B, C, H, W) for each iteration