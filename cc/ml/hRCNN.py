import torch
import torch.nn as nn

from cc.ml.sparse_cnn_unet import SparseCNNEncoder, SparseCNNDecoder, SparseCNNUNet, SparseLocalStage, DenseLocalStage, DenseUpConv2d, SparseConv2d
from cc.ml import MAE_logs, Classifier_logs
from cc.ml.utils import load_checkpoint
from cc.ml.heads.classifier import ClassifierHead
from cc.ml.heads.task_head import TaskHead
from cc.ml.cc_modules import CCModule
from cc.datasets.msmnist import msmnist, visualize_msmnist_examples

class hRCNN(nn.Module):
    '''
    hRCNN architecture collapses a pretrained CNN autoencoder into a recurrent CNN 
        by reusing (and freezing) the encoder (feedforward) and decoder (feedback) weights.
    '''
    def __init__(self, encoder:SparseCNNEncoder, decoder:SparseCNNDecoder, head:TaskHead, 
                 data_dims:tuple[int, int] = (1,28,28), config:dict = {}):
        super(hRCNN, self).__init__()
        self.data_dims, self.config = data_dims, config
        
        # Load the pre-trained encoder, decoder, and head weights into the hRCNN architecture
        self._load_pretrained(encoder, decoder, head)

        # Define hierarchical recurrent layers
        self._define_hierarchical_layers()

    def forward(self, x:torch.Tensor)->torch.Tensor:
        ''''
        Expects tensor with temporal dimension of shape (B, T, C, H, W).
        '''
        # Initialize hidden state (all layers start with zero hidden state)
        Y_old_activations = {f'L{i}': torch.zeros((x.shape[0], *l.dims_), device=x.device, dtype=x.dtype) 
                             for i, l in enumerate(self.cc_layers)}
        logs = []
        for t in range(x.shape[1]): # iterate over time dimension
            Y_new_activations = {f'L{i}': None  for i in range(self.n_layers)}
            yff = x[:, t, ...]
            # Process all recurrent layers at once
            for il, layer in enumerate(self.cc_layers):
                yfb = Y_old_activations.get(f'L{il+1}', None) 
                Y_new_activations[f'L{il}'] = layer(yff, yfb, Y_old_activations[f'L{il}'], 
                                                    train = True)
                # Feed current-step activation to the next (higher) layer.
                yff = Y_old_activations[f'L{il}']
            # Pass output to head from current-step top-layer activations.
            top_state = Y_old_activations[f'L{self.n_layers-1}']
            if top_state is None:
                top_state = torch.zeros((x.shape[0], *self.cc_layers[-1].dims_),
                                        device=x.device,dtype=x.dtype,)
            out = self.head(top_state)
            # Update activations for all layers at once
            Y_old_activations.update(Y_new_activations)
            logs.append(out)
        return logs

    def _define_hierarchical_layers(self)->None:
        self.n_layers = len(self.ff_local_processing) # number of hierarchical layers based on the feedforward local processing stages
        assert self.n_layers == len(self.fb_local_processing), "Mismatch in number of feedforward and feedback local processing stages."
        assert self.n_layers == len(self.ff_downsample_convs) == len(self.fb_upsample_convs) + 1, "There should be one more set of FF weights (input->V1) than FB weights."
        # NOTE: assume halving in spatial dimensions and doubling channel dimensions each layer (V1 starts at data resolution)
        spatial_dims = [(self.data_dims[1] // (2 ** i), self.data_dims[2] // (2 ** i)) 
                        if self.data_dims[1] % (2 ** i) == 0 and self.data_dims[2] % (2 ** i) == 0 
                        else ((self.data_dims[1] // (2 ** i))+1, (self.data_dims[2] // (2 ** i))+1)
                        for i in range(self.n_layers)]

        self.cc_layers = nn.ModuleList()
        src = self.data_dims[1]
        
        for il, spatial_dim in enumerate(spatial_dims):
            dst = spatial_dim[0]
            # Bottom-up FF input from layer below
            FF_conv = self.ff_downsample_convs.get(f"down{src}_to_{dst}", None)
            FF_local = self.ff_local_processing.get(f"local{dst}", None)
            assert FF_conv is not None and FF_local is not None, f"Missing FF conv or local processing for down{src}_to_{dst} and local{dst}"
            FF_conv = FF_Conv2d(FF_conv, FF_local)
            # Top-down FB context from layer above (if exists). Top layer has no feedback source.
            if il < self.n_layers - 1:
                next_dim = spatial_dims[il + 1][0]
            else:
                next_dim = None
            FB_local = self.fb_local_processing.get(f"local{next_dim}", None)
            FB_conv = self.fb_upsample_convs.get(f"up{next_dim}_to_{dst}", None)
            FB_local2 = self.fb_local_processing.get(f"local{dst}", None) if il == 0 else None
            FB_conv = FB_Conv2d(FB_conv, FB_local, FB_local2) if FB_conv is not None else None
            # print(f"Defining CC layer with FF_conv: {FF_conv}, FB_conv: {FB_conv}")
            layer = CCModule(spatial_dim, FF_conv, FB_conv)
            self.cc_layers.append(layer)
            src = dst
        
    def _load_pretrained(self, encoder:SparseCNNEncoder, decoder:SparseCNNDecoder, head:TaskHead,
                         freeze_encoder:bool = True, freeze_decoder:bool = True, freeze_head:bool = True)->None:
        # Extract feedforward (encoder) weights
        encoder = encoder
        if freeze_encoder:
            for param in encoder.parameters():
                param.requires_grad = False
            encoder.eval()
        self.ff_local_processing = {}
        self.ff_downsample_convs = {}
        
        for name, module in (encoder.named_modules()):
            if isinstance(module, SparseLocalStage):
                self.ff_local_processing[name] = module
            elif isinstance(module, nn.Conv2d) and (module.stride == (2, 2) or 'down' in name):
                self.ff_downsample_convs[name] = module
        
        # Extract feedback (decoder) weights
        decoder = decoder
        if freeze_decoder:
            for param in decoder.parameters():
                param.requires_grad = False
            decoder.eval()
        self.fb_local_processing = {}
        self.fb_upsample_convs = {}

        for name, module in (decoder.named_modules()):
            if isinstance(module, DenseLocalStage):
                self.fb_local_processing[name] = module
            elif isinstance(module, (DenseUpConv2d, nn.ConvTranspose2d)) and (module.stride == (2, 2) or 'up' in name):
                self.fb_upsample_convs[name] = module
        
        self.ff_names = list(self.ff_local_processing.keys())
        self.fb_names = list(self.fb_local_processing.keys())

        # Task-specific head (eg. classification), starting from the latent dimension
        self.head = head
        if freeze_head:
            for param in self.head.parameters():
                param.requires_grad = False
            self.head.eval()

class FF_Conv2d(nn.Module):
    def __init__(self, conv:SparseConv2d, local_processing:SparseLocalStage):
        super(FF_Conv2d, self).__init__()
        self.conv = conv
        self.local_processing = local_processing
        self.out_channels = conv.out_channels

    def forward(self, x:torch.Tensor, keep_mask:torch.Tensor)->torch.Tensor:
        x = self.conv(x, keep_mask)
        x = self.local_processing(x, keep_mask)
        return x

class FB_Conv2d(nn.Module):
    def __init__(self, conv:nn.ConvTranspose2d|DenseUpConv2d, local_processing:DenseLocalStage, 
                 local_processing2:DenseLocalStage|None = None):
        super(FB_Conv2d, self).__init__()
        self.conv = conv
        self.local_processing = local_processing
        self.local_processing2 = local_processing2
        self.out_channels = conv.out_channels

    def forward(self, x:torch.Tensor)->torch.Tensor:
        x = self.local_processing(x)
        x = self.conv(x)
        if self.local_processing2 is not None:
            x = self.local_processing2(x)
        return x

# --- hRCNN utils ---
def load_pretrained_weights()->hRCNN:
    # Define pre-trained autoencoder model
    mae_model = SparseCNNUNet(num_input_channels=1, num_output_channels=1, num_filters=32)
    # Load pre-trained autoencoder weights (replace with actual checkpoint path)
    mae_checkpoint_path = f"{MAE_logs}/version_11/checkpoints/epoch=15-step=6752.ckpt"
    load_checkpoint(mae_model, mae_checkpoint_path)
    encoder:SparseCNNEncoder = mae_model.encoder
    decoder:SparseCNNDecoder = mae_model.decoder
    
    # Load pre-trained head
    mnist_classifier_head = ClassifierHead.from_pretrained_unet(
        checkpoint_path=mae_checkpoint_path,
        num_classes=10, # MNIST has 10 classes
        latent_dim=32*4, # latent dimension from the MAE model 
        lr=1e-3,
        freeze_encoder=True,
    ).head
    # Load pre-trained head weights
    head_checkpoint_path = f"{Classifier_logs}/version_21/checkpoints/epoch=5-step=2532.ckpt"
    load_checkpoint(mnist_classifier_head, head_checkpoint_path, weights_only=False)
    
    # Instantiate hRCNN with the loaded encoder, decoder, and head
    model = hRCNN(encoder, decoder, mnist_classifier_head)

    return model

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    hrcnn = load_pretrained_weights()
    examples, labels = visualize_msmnist_examples(num_examples=4, num_timeframes=100, mask_ratio=0.5)
    with torch.no_grad():
        logs = hrcnn(examples)
    # logs is a list[T] of tensors with shape (B, num_classes)
    logits = torch.stack(logs, dim=1)  # (B, T, num_classes)
    probs = torch.softmax(logits, dim=-1)
    preds = probs.argmax(dim=-1)  # (B, T)
    timesteps = torch.arange(logits.shape[1]).cpu().numpy()

    print("labels:", labels.tolist())
    print("preds per sample/time:", preds.tolist())

    fig, axes = plt.subplots(labels.shape[0], 1, figsize=(10, 2.2 * labels.shape[0]), sharex=True)
    if labels.shape[0] == 1:
        axes = [axes]
    for i, ax in enumerate(axes):
        true_label = int(labels[i].item())
        true_prob = probs[i, :, true_label].cpu().numpy()
        pred_prob = probs[i].max(dim=-1).values.cpu().numpy()
        ax.plot(timesteps, true_prob, marker="o", label=f"P(true={true_label})")
        ax.plot(timesteps, pred_prob, marker="x", linestyle="--", label="P(pred)")
        ax.set_ylim(0.0, 1.0)
        ax.set_ylabel(f"sample {i}")
        ax.grid(alpha=0.25)
        ax.set_title(f"label={true_label} | preds={preds[i].tolist()}")
        ax.legend(loc="lower right")

    axes[-1].set_xlabel("time step")
    fig.tight_layout()
    plt.show()
    
