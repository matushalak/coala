import torch
import torch.nn as nn

from cc.masked_ae.sparse_cnn_unet import SparseLocalStage, SparseConv2d

class hRCNN(nn.Module):
    '''
    hRCNN architecture collapses a pretrained CNN autoencoder into a recurrent CNN 
        by reusing (and freezing) the encoder (feedforward) and decoder (feedback) weights.
    '''
    def __init__(self, encoder, decoder, head, config:dict = None):
        super(hRCNN, self).__init__()
        # Extract feedforward (encoder) weights
        self.encoder = encoder
        self.ff_local_processing = {}
        self.ff_downsample_convs = {}
        
        for name, module in (self.encoder.named_modules()):
            if isinstance(module, SparseLocalStage):
                self.ff_local_processing[name] = module
            elif isinstance(module, nn.Conv2d) and module.stride == (2, 2):
                self.ff_downsample_convs[name] = module

        # Extract feedback (decoder) weights
        self.decoder = decoder
        self.fb_local_processing = {}
        self.fb_upsample_convs = {}

        for name, module in (self.decoder.named_modules()):
            if isinstance(module, SparseLocalStage):
                self.fb_local_processing[name] = module
            elif isinstance(module, nn.ConvTranspose2d) and module.stride == (2, 2):
                self.fb_upsample_convs[name] = module
        
        # Task-specific head (eg. classification), starting from the latent dimension
        self.head = head

