from torch import nn

class HierarchicalAutoencoder(nn.Module):
    '''
    Define the Autoencoder architecture for pretraining, agostic to architecture
    '''
    def __init__(self,
                 n_layers:int,
                 d_layers:list[tuple],
                 E_kwargs:dict,
                 layers_E:list[nn.Module],
                 layers_E_kwargs:list[dict],
                 D_kwargs:dict,
                 layers_D:list[nn.Module],
                 layers_D_kwargs:list[dict]):
        pass

    def forward(self, x):
        # Implement the forward pass of the HierarchicalAutoencoder architecture
        latents = encoder(x)
        recon = decoder(latents)
        return recon, latents