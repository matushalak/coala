import torch
import torch.nn as nn

class hRCNN(nn.Module):
    '''
    hRCNN architecture collapses a pretrained CNN autoencoder into a recurrent CNN 
        by reusing (and freezing) the encoder (feedforward) and decoder (feedback) weights.
    '''
    def __init__(self, encoder, decoder):
        super(hRCNN, self).__init__()
        # Extract feedforward (encoder) weights
        self.encoder = encoder
        
        
        # Extract feedback (decoder) weights
        self.decoder = decoder