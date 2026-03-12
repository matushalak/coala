import torch
import torch.nn as nn

from cc.ml.sparse_cnn_unet import SparseCNNEncoder, SparseCNNDecoder, SparseCNNUNet, SparseLocalStage, DenseLocalStage, DenseUpConv2d, SparseConv2d
from cc.ml import MAE_logs, Classifier_logs
from cc.ml.utils import load_checkpoint
from cc.ml.heads.classifier import ClassifierHead
from cc.ml.heads.task_head import TaskHead
from cc.ml.cc_modules import CCModule
from cc.datasets.msmnist import msmnist, visualize_msmnist_examples

class COALANet(nn.Module):
    '''
    COALA-Net:
    Collapsed Autoencoder with Local Adaptation
        Hierarchical CNN architecture collapses a pretrained CNN autoencoder into a recurrent CNN 
        by reusing (and freezing) the encoder (feedforward) and decoder (feedback) weights.
    '''
    def __init__(self, encoder:SparseCNNEncoder, decoder:SparseCNNDecoder, head:TaskHead, 
                 data_dims:tuple[int, int] = (1,28,28), config:dict = {}):
        super(COALANet, self).__init__()
        self.data_dims, self.config = data_dims, config
        
        # Load the pre-trained encoder, decoder, and head weights into the COALANet architecture
        self._load_pretrained(encoder, decoder, head)

        # Define hierarchical recurrent layers
        self._define_hierarchical_layers()

        self.dynamic_updates = True
        self.training_params = False

        self.eval() # by default keep all parameters frozen

    @torch.no_grad()
    def forward(self, x:torch.Tensor)->torch.Tensor:
        ''''
        Expects tensor with temporal dimension of shape (B, T, C, H, W).
        Returns outputs for each timestep in the batch
        '''
        # Initialize hidden state (all layers start with zero hidden state)
        Y_old_activations = {f'L{i}': torch.zeros((x.shape[0], *l.dims_), device=x.device, dtype=x.dtype) 
                             for i, l in enumerate(self.cc_layers)}
        logits_per_timestep = []
        for t in range(x.shape[1]): # iterate over time dimension
            Y_new_activations = {f'L{i}': None  for i in range(self.n_layers)}
            yff = x[:, t, ...]
            # Process all recurrent layers at once
            for il, layer in enumerate(self.cc_layers):
                yfb = Y_old_activations.get(f'L{il+1}', None) 
                layer_outputs = layer(yff, yfb, Y_old_activations[f'L{il}'])
                Y_new_activations[f'L{il}'] = layer_outputs[0]
                if self.dynamic_updates:
                    layer.update(*layer_outputs)

                # Feed current-step activation to the next (higher) layer.
                yff = Y_new_activations[f'L{il}']
            # Pass output to head from current-step top-layer activations.
            top_state = Y_new_activations[f'L{self.n_layers-1}']
            assert top_state is not None, "Top layer activations should not be None."
            out = self.head(top_state)
            # Update activations for all layers at once
            Y_old_activations.update(Y_new_activations)
            logits_per_timestep.append(out)
        return logits_per_timestep

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
            FF_conv = FF_Conv2d(conv=FF_conv, local_processing=FF_local)
            # Top-down FB context from layer above (if exists). Top layer has no feedback source.
            if il < self.n_layers - 1:
                next_dim = spatial_dims[il + 1][0]
            else:
                next_dim = None
            FB_local0 = self.fb_local_processing.get(f"local{next_dim}", None) if il == len(spatial_dims)-2 else None
            FB_local = self.fb_local_processing.get(f"local{dst}", None)
            FB_upconv = self.fb_upsample_convs.get(f"up{next_dim}_to_{dst}", None)
            FB_conv = FB_Conv2d(upconv=FB_upconv, local_processing=FB_local, local_processing0=FB_local0
                                ) if FB_upconv is not None else None
            print(f"Defining CC layer with FF_conv: {FF_conv}, FB_conv: {FB_conv}")
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

    def iter_cc_modules(self):
        return iter(self.cc_layers)

    def adaptation_parameters(self):
        params = []
        for layer in self.cc_layers:
            params.append(layer.time_alpha)
            params.append(layer.Lambda_FF.raw_lr)
            params.append(layer.Lambda_LAT.raw_lr)
            if layer.Lambda_FB is not None:
                params.append(layer.Lambda_FB.raw_lr)
        return params

    def set_adaptation_trainable(self, trainable:bool = True)->None:
        self.training_params = False
        for param in self.parameters():
            param.requires_grad = False
        self.eval()

    def reset_dynamic_state(self, ref_tensor:torch.Tensor|None = None)->None:
        for layer in self.cc_layers:
            layer.reset_dynamic_state(ref_tensor=ref_tensor)

class FF_Conv2d(nn.Module):
    ''''
    Downsampling convolution + local processing
    '''
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
    '''
    Upsampling convolution + local processing (w optional skip connection)
    '''
    def __init__(self, upconv:nn.ConvTranspose2d|DenseUpConv2d, local_processing:DenseLocalStage, 
                 local_processing0:DenseLocalStage|None = None):
        super(FB_Conv2d, self).__init__()
        self.upconv = upconv
        self.local_processing = local_processing
        self.local_processing0 = local_processing0
        self.out_channels = upconv.out_channels

    def forward(self, x:torch.Tensor, skip:torch.Tensor|None)->torch.Tensor:
        if self.local_processing0 is not None:
            x = self.local_processing0(x)
        x = self.upconv(x)
        if skip is not None:
            x = x + skip
        x = self.local_processing(x)
        return x

# --- COALANet utils ---
def load_pretrained_weights()->COALANet:
    # Define pre-trained autoencoder model
    mae_model = SparseCNNUNet(num_input_channels=1, num_output_channels=1, num_filters=32,
                              upconv_method="upsample+conv")
    # Load pre-trained autoencoder weights (replace with actual checkpoint path)
    mae_checkpoint_path = f"{MAE_logs}/version_13/checkpoints/epoch=19-step=8440.ckpt"
    load_checkpoint(mae_model, mae_checkpoint_path)
    encoder:SparseCNNEncoder = mae_model.encoder
    decoder:SparseCNNDecoder = mae_model.decoder
    
    # Load pre-trained classifier with head
    mnist_classifier = ClassifierHead.from_pretrained_unet(
        checkpoint_path=mae_checkpoint_path,
        num_classes=10, # MNIST has 10 classes
        latent_dim=32*4, # latent dimension from the MAE model 
        lr=1e-3,
        freeze_encoder=True,
        upconv_method = 'upsample+conv')
    # Load pre-trained classifier weights
    classifier_checkpoint_path = f"{Classifier_logs}/version_23/checkpoints/epoch=9-step=4220.ckpt"
    load_checkpoint(mnist_classifier, classifier_checkpoint_path, weights_only=False)
    # Extract the classifier head
    mnist_classifier_head = mnist_classifier.head

    # Instantiate COALANet with the loaded encoder, decoder, and head
    model = COALANet(encoder, decoder, mnist_classifier_head)

    return model


def stack_temporal_logits(logits:list[torch.Tensor] | torch.Tensor)->torch.Tensor:
    if isinstance(logits, torch.Tensor):
        if logits.dim() != 3:
            raise ValueError(f"Expected logits tensor of shape (B, T, C), got {tuple(logits.shape)}.")
        return logits
    return torch.stack(logits, dim=1)


def create_temporal_prediction_figure(
    logits:list[torch.Tensor] | torch.Tensor,
    labels:torch.Tensor,
    max_examples:int|None = None,
):
    import matplotlib.pyplot as plt

    logits = stack_temporal_logits(logits).detach().cpu()
    labels = labels.detach().cpu()
    if max_examples is not None:
        logits = logits[:max_examples]
        labels = labels[:max_examples]

    probs = torch.softmax(logits, dim=-1)
    preds = probs.argmax(dim=-1)
    timesteps = torch.arange(logits.shape[1]).cpu().numpy()
    tick_step = max(1, logits.shape[1] // 12)
    fig_width = min(18.0, max(10.0, 0.35 * logits.shape[1]))
    fig, axes = plt.subplots(labels.shape[0], 1, figsize=(fig_width, 2.2 * labels.shape[0]), sharex=True)
    if labels.shape[0] == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        true_label = int(labels[i].item())
        true_prob = probs[i, :, true_label].numpy()
        pred_prob = probs[i].max(dim=-1).values.numpy()
        ax.plot(timesteps, true_prob, marker="o", label=f"P(true={true_label})")
        ax.plot(timesteps, pred_prob, marker="x", linestyle="--", label="P(pred)")
        ax.set_xlim(0, logits.shape[1] - 1)
        ax.set_ylabel(f"sample {i}")
        ax.grid(alpha=0.25)
        ax.set_title(f"label={true_label} | final_pred={int(preds[i, -1].item())}")
        ax.legend(loc="lower right")

    axes[-1].set_xlabel("time step")
    axes[-1].set_xticks(timesteps[::tick_step])
    fig.tight_layout()
    return fig

# TODO: clamp V1 output; change task to reconstruction

if __name__ == "__main__":
    coalanet = load_pretrained_weights()
    # ablation results
    # can perform just as well (and mostly better with leaky integration of FF weights)
    # TODO: need to measure success rate systematically with loss from COALAmodel
    examples, labels = visualize_msmnist_examples(num_examples=1, 
                                                  number_of_masks=200, timesteps_per_mask=1,
                                                  mask_ratio=0.5,
                                                  masked_fill='random',
                                                  accepted_digits=None, show=True)
    with torch.no_grad():
        logits = stack_temporal_logits(coalanet(examples))
    fig = create_temporal_prediction_figure(logits, labels)
    import matplotlib.pyplot as plt
    plt.show()
    
