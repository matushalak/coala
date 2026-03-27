import argparse
from pathlib import Path
import torch
import torch.nn as nn
from typing import Literal

from cc.ml.sparse_cnn_unet import SparseCNNEncoder, SparseCNNDecoder, SparseCNNUNet, SparseLocalStage, DenseLocalStage, DenseUpConv2d, SparseConv2d
from cc.ml import MAE_logs, Classifier_logs, FM_logs
from cc.ml.utils import load_checkpoint
from cc.ml.heads.classifier import ClassifierHead
from cc.ml.cc_modules import CCModule
from cc.datasets.msmnist import msmnist, visualize_msmnist_examples
from cc.ml.visualize_activation_maps import create_activation_input_figure, create_activation_map_figure

class COALANet(nn.Module):
    '''
    COALA-Net:
    Collapsed Autoencoder with Local Adaptation
        Hierarchical CNN architecture collapses a pretrained CNN autoencoder into a recurrent CNN 
        by reusing (and freezing) the encoder (feedforward) and decoder (feedback) weights.
    '''
    MODES = ("discriminative", "generative")

    def __init__(
        self,
        encoder: SparseCNNEncoder,
        decoder: SparseCNNDecoder,
        head: nn.Module | None = None,
        data_dims: tuple[int, int] = (1, 28, 28),
        config: dict | None = None,
        mode: Literal["discriminative", "generative"] = "discriminative",
    ):
        super(COALANet, self).__init__()
        self.data_dims = data_dims
        self.config = {} if config is None else config
        self.mode = "discriminative"
        self.head: nn.Module | None = None
        self.discriminative_head: nn.Module | None = None
        self.generative_head: nn.Module | None = None
        
        # Load the pre-trained encoder, decoder, and head weights into the COALANet architecture
        self._load_pretrained(encoder, decoder, head)

        # Define hierarchical recurrent layers
        self._define_hierarchical_layers()
        self.set_mode(mode)

        self.dynamic_updates = True
        self.training_params = False

        self.eval() # by default keep all parameters frozen

    @torch.no_grad()
    def forward(
        self,
        x: torch.Tensor,
        return_activation_maps: bool = False,
    ) -> list[torch.Tensor] | dict[str, list[torch.Tensor] | dict[str, dict[str, torch.Tensor]]]:
        ''''
        Expects tensor with temporal dimension of shape (B, T, C, H, W).
        Returns outputs for each timestep in the batch.
        When ``return_activation_maps`` is True, also returns per-layer activation maps
        for ``Y``, ``y_FF``, and ``y_FB`` averaged across channels with shape (B, T, H, W).
        '''
        # Initialize hidden state (all layers start with zero hidden state)
        Y_old_activations = {f'L{i}': torch.zeros((x.shape[0], *l.dims_), device=x.device, dtype=x.dtype) 
                             for i, l in enumerate(self.cc_layers)}
        outputs_per_timestep = []
        activation_maps = None
        if return_activation_maps:
            activation_maps = {
                signal_name: {f"L{i}": [] for i in range(self.n_layers)}
                for signal_name in ("Y", "y_FF", "y_FB")
            }
        for t in range(x.shape[1]): # iterate over time dimension
            Y_new_activations = {f'L{i}': None  for i in range(self.n_layers)}
            yff = x[:, t, ...]
            readout_source = None
            # Process all recurrent layers at once
            for il, layer in enumerate(self.cc_layers):
                yfb = Y_old_activations.get(f'L{il+1}', None) 
                layer_outputs = layer(yff, yfb, Y_old_activations[f'L{il}'])
                Y_new_activations[f'L{il}'] = layer_outputs[0]
                if activation_maps is not None:
                    activation_maps["Y"][f"L{il}"].append(layer_outputs[0].mean(dim=1))
                    activation_maps["y_FF"][f"L{il}"].append(layer_outputs[1].mean(dim=1))
                    activation_maps["y_FB"][f"L{il}"].append(layer_outputs[2].mean(dim=1))
                if self.dynamic_updates:
                    layer.update(*layer_outputs)
                if self.mode == "generative" and il == 0:
                    # readout_source = layer_outputs[2] # V2->V1 feedback signal
                    readout_source = layer_outputs[0] # temporally integrated V1 activity

                # Feed current-step activation to the next (higher) layer.
                yff = Y_new_activations[f'L{il}']
            if self.mode == "discriminative":
                readout_source = Y_new_activations[f'L{self.n_layers-1}']
                assert readout_source is not None, "Top layer activations should not be None."
            assert self.head is not None, "COALANet head has not been configured."
            assert readout_source is not None, "Readout source should not be None."
            out = self.head(readout_source)
            # Update activations for all layers at once
            Y_old_activations.update(Y_new_activations)
            outputs_per_timestep.append(out)
        if activation_maps is None:
            return outputs_per_timestep
        return {
            "outputs_per_timestep": outputs_per_timestep,
            "activation_maps": stack_temporal_activation_maps(activation_maps),
        }

    def set_mode(self, mode: Literal["discriminative", "generative"]) -> None:
        if mode not in self.MODES:
            raise ValueError(f"mode must be one of {self.MODES}, got {mode!r}.")
        if mode == "discriminative" and self.discriminative_head is None:
            raise ValueError("Discriminative mode requires a discriminative head.")
        if mode == "generative" and self.generative_head is None:
            raise ValueError("Generative mode requires a generative head.")
        self.mode = mode
        self.head = self.discriminative_head if mode == "discriminative" else self.generative_head

    def _define_hierarchical_layers(self)->None:
        self.n_layers = len(self.ff_local_processing) # number of hierarchical layers based on the feedforward local processing stages
        assert self.n_layers == len(self.fb_local_processing), "Mismatch in number of feedforward and feedback local processing stages."
        assert self.n_layers == len(self.ff_downsample_convs) == len(self.fb_upsample_convs) + 1, "There should be one more set of FF weights (input->V1) than FB weights."
        # NOTE: assume halving in spatial dimensions and doubling channel dimensions each layer (V1 starts at data resolution)
        spatial_dims = [(self.data_dims[1] // (2 ** i), self.data_dims[2] // (2 ** i)) 
                        if self.data_dims[1] % (2 ** i) == 0 and self.data_dims[2] % (2 ** i) == 0 
                        else ((self.data_dims[1] // (2 ** i))+1, (self.data_dims[2] // (2 ** i))+1)
                        for i in range(self.n_layers)]
        
        # NOTE: slower time constants deeper in network
        time_constants = [0.05, 0.04, 0.03, 0.02]
        # time_constants = [0.05] * self.n_layers # same time constant across network layers
        
        # Lateral inhibition kernel sizes
        lateral_inhibition_kernels = [(7,7), (5, 5), (3, 3), (1, 1)]


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
            # print(f"Defining CC layer with FF_conv: {FF_conv}, FB_conv: {FB_conv}")
            layer = CCModule(spatial_dim, FF_conv, FB_conv, 
                             time_alpha=time_constants[il], 
                             LAT_ksize=lateral_inhibition_kernels[il])
            self.cc_layers.append(layer)
            src = dst
        
    def _load_pretrained(self, encoder:SparseCNNEncoder, decoder:SparseCNNDecoder, head:nn.Module | None,
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

        # Task-specific top readout (V4) for discriminative mode.
        self.discriminative_head = head
        if self.discriminative_head is not None and freeze_head:
            for param in self.discriminative_head.parameters():
                param.requires_grad = False
            self.discriminative_head.eval()

        # Generative readout clamps the V1 feedback state through the pretrained decoder output conv.
        self.generative_head = decoder.up28_to_out

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
def load_pretrained_weights(
    mode: Literal["discriminative", "generative"] = "discriminative",
)->COALANet:
    # MAE models
    # mae_checkpoint_path = f"{MAE_logs}/lightning_logs/version_13/checkpoints/epoch=19-step=8440.ckpt"
    # mae_checkpoint_path = f"{MAE_logs}/lightning_logs/version_18/checkpoints/epoch=19-step=8440.ckpt" # improved MAE
    
    # dMAE models
    # mae_checkpoint_path = f"{MAE_logs}/lightning_logs/version_26/checkpoints/epoch=49-step=21100.ckpt" # denoising MAE
    mae_checkpoint_path = f"{MAE_logs}/lightning_logs/version_21/checkpoints/epoch=19-step=8440.ckpt" # denoising MAE, more noise
    # mae_checkpoint_path = f"{MAE_logs}/lightning_logs/version_27/checkpoints/epoch=50-step=21522.ckpt" # "FM" dMAE pretraining, also good
    
    # FM models
    # mae_checkpoint_path = f"{FM_logs}/lightning_logs/version_1/checkpoints/epoch=18-step=16036.ckpt" # FM model, pretrained for 21 epochs
    
    mae_checkpoint = torch.load(mae_checkpoint_path, map_location="cpu", weights_only=False)
    mae_hparams = dict(mae_checkpoint.get("hyper_parameters", {}))
    mae_num_input_channels = int(mae_hparams.get("num_input_channels", 1))
    mae_num_filters = int(mae_hparams.get("num_filters", 32))
    mae_decoder_densify_mode = str(mae_hparams.get("decoder_densify_mode", "random"))
    mae_use_skip = bool(mae_hparams.get("use_skip", True))
    mae_upconv_method = str(mae_hparams.get("upconv_method", "upsample+conv"))
    # Older MAE checkpoints predate this hparam and used LayerNorm throughout.
    mae_norm_type = str(mae_hparams.get("norm_type", "layernorm"))

    # Define pre-trained autoencoder model
    mae_model = SparseCNNUNet(
        num_input_channels=mae_num_input_channels,
        num_output_channels=mae_num_input_channels,
        num_filters=mae_num_filters,
        decoder_densify_mode=mae_decoder_densify_mode,
        use_skip=mae_use_skip,
        upconv_method=mae_upconv_method,
        norm_type=mae_norm_type,
    )
    # Load pre-trained autoencoder weights (replace with actual checkpoint path)
    load_checkpoint(mae_model, mae_checkpoint_path, strict=False)
    encoder:SparseCNNEncoder = mae_model.encoder
    decoder:SparseCNNDecoder = mae_model.decoder
    
    mnist_classifier_head = None
    if mode == "discriminative":
        # Load pre-trained classifier with head
        mnist_classifier = ClassifierHead.from_pretrained_unet(
            checkpoint_path=mae_checkpoint_path,
            num_classes=10, # MNIST has 10 classes
            latent_dim=32*4, # latent dimension from the MAE model 
            lr=1e-3,
            freeze_encoder=True,
            upconv_method=mae_upconv_method,
        )
        # Load pre-trained classifier weights
        classifier_checkpoint_path = f"{Classifier_logs}/lightning_logs/version_24/checkpoints/epoch=8-step=3798.ckpt"
        load_checkpoint(mnist_classifier, classifier_checkpoint_path, weights_only=False, strict=False)
        # Extract the classifier head
        mnist_classifier_head = mnist_classifier.head

    # Instantiate COALANet with the loaded encoder, decoder, and head
    model = COALANet(encoder, decoder, mnist_classifier_head, mode=mode)

    return model


def stack_temporal_outputs(
    outputs: list[torch.Tensor] | torch.Tensor,
    expected_ndim: int | None = None,
) -> torch.Tensor:
    if isinstance(outputs, torch.Tensor):
        stacked = outputs
    else:
        stacked = torch.stack(outputs, dim=1) # introduce "time" dimension 1 (B, T, C, H, W)
    if expected_ndim is not None and stacked.dim() != expected_ndim:
        raise ValueError(f"Expected tensor with {expected_ndim} dims, got shape {tuple(stacked.shape)}.")
    return stacked


def stack_temporal_activation_maps(
    activation_maps: dict[str, dict[str, list[torch.Tensor] | torch.Tensor]],
) -> dict[str, dict[str, torch.Tensor]]:
    stacked_maps: dict[str, dict[str, torch.Tensor]] = {}
    for signal_name, per_layer_maps in activation_maps.items():
        stacked_maps[signal_name] = {}
        for layer_name, maps in per_layer_maps.items():
            stacked_maps[signal_name][layer_name] = stack_temporal_outputs(maps, expected_ndim=4)
    return stacked_maps


def stack_temporal_logits(logits:list[torch.Tensor] | torch.Tensor)->torch.Tensor:
    return stack_temporal_outputs(logits, expected_ndim=3)


def stack_temporal_reconstructions(
    reconstructions: list[torch.Tensor] | torch.Tensor,
) -> torch.Tensor:
    return stack_temporal_outputs(reconstructions, expected_ndim=5)


def _to_display_range(images: torch.Tensor) -> torch.Tensor:
    return ((images + 1.0) * 0.5).clamp_(0.0, 1.0)


def compute_temporal_reconstruction_loss(
    reconstructions: list[torch.Tensor] | torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """
    Per-example, per-timestep MSE for [-1, 1] targets.
    """
    reconstructions = stack_temporal_reconstructions(reconstructions)
    if targets.dim() == 4:
        per_pixel_mse = (reconstructions - targets.unsqueeze(1)).pow(2).mean(dim=2)
    elif targets.dim() == 5:
        per_pixel_mse = (reconstructions - targets).pow(2).mean(dim=2)
    else:
        raise ValueError(f"Expected targets tensor of shape (B, C, H, W), or (B, E, C, H, W) got {tuple(targets.shape)}.")
    
    return per_pixel_mse.mean(dim=(-2, -1))


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


def create_temporal_reconstruction_figure(
    reconstructions: list[torch.Tensor] | torch.Tensor,
    targets: torch.Tensor,
    max_examples: int | None = None,
):
    import matplotlib.pyplot as plt

    losses = compute_temporal_reconstruction_loss(reconstructions, targets).detach().cpu()
    if max_examples is not None:
        losses = losses[:max_examples]

    timesteps = torch.arange(losses.shape[1]).cpu().numpy()
    tick_step = max(1, losses.shape[1] // 12)
    fig_width = min(18.0, max(10.0, 0.35 * losses.shape[1]))
    fig, axes = plt.subplots(losses.shape[0], 1, figsize=(fig_width, 2.2 * losses.shape[0]), sharex=True)
    if losses.shape[0] == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        series = losses[i].numpy()
        ax.plot(timesteps, series, marker="o", label="reconstruction MSE")
        ax.set_xlim(0, losses.shape[1] - 1)
        ax.set_ylabel(f"sample {i}")
        ax.grid(alpha=0.25)
        ax.set_title(f"final_loss={series[-1]:.4f}")
        ax.legend(loc="upper right")

    axes[-1].set_xlabel("time step")
    axes[-1].set_xticks(timesteps[::tick_step])
    fig.tight_layout()
    return fig


def create_temporal_reconstruction_grid(
    target_images: torch.Tensor,
    masked_images: torch.Tensor,
    reconstructions: list[torch.Tensor] | torch.Tensor,
    num_examples: int = 4,
    max_time_steps: int = 16,
) -> torch.Tensor:
    from torchvision.utils import make_grid

    reconstructions = _to_display_range(stack_temporal_reconstructions(reconstructions).detach().cpu())
    masked_images = _to_display_range(masked_images[:num_examples].detach().cpu())
    target_images = _to_display_range(target_images[:num_examples].detach().cpu())
    reconstructions = reconstructions[:num_examples]

    if masked_images.shape[:2] != reconstructions.shape[:2]:
        raise ValueError(
            "masked_images and reconstructions must agree on batch and time dimensions, "
            f"got {tuple(masked_images.shape[:2])} vs {tuple(reconstructions.shape[:2])}."
        )

    _, total_t, _, _, _ = masked_images.shape
    num_time_steps = min(total_t, max_time_steps)
    time_idx = torch.linspace(0, total_t - 1, steps=num_time_steps).round().long()
    masked_panel = masked_images[:, time_idx]
    recon_panel = reconstructions[:, time_idx]
    target_panel = target_images[:, time_idx].unsqueeze(2)
    panel = torch.stack((masked_panel, recon_panel, target_panel), dim=1).reshape(-1, *masked_images.shape[2:])
    return make_grid(panel, nrow=num_time_steps, normalize=False, pad_value=0.5)

def _show_image_grid(grid: torch.Tensor, title: str) -> None:
    import matplotlib.pyplot as plt

    # Match torchvision.save_image rendering used in MAEmodel epoch_x_recon outputs.
    grid = grid.detach().cpu().mul(255).add_(0.5).clamp_(0, 255).to(torch.uint8)
    fig, ax = plt.subplots(figsize=(12, 4))
    if grid.shape[0] == 1 or (
        grid.shape[0] == 3
        and torch.equal(grid[0], grid[1])
        and torch.equal(grid[1], grid[2])
    ):
        ax.imshow(grid[0], cmap="gray", interpolation="nearest", vmin=0, vmax=255)
    else:
        ax.imshow(grid.permute(1, 2, 0), interpolation="nearest")
    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()


def _parse_masked_fill_arg(masked_fill: str) -> str | float:
    if masked_fill == "random":
        return masked_fill
    return float(masked_fill)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Task and model configuration
    parser.add_argument("--mode",default="generative",choices=("discriminative", "generative"),type=str,
                        help="Which COALANet readout mode to demo.",)
    
    # Data stream configuration
    parser.add_argument("--num_examples", default=1, type=int, 
                        help="Number of examples to visualize.")
    parser.add_argument("--accepted_digits", nargs="*", type=int, default=None, 
                        help="Optional subset of digits.")
    
    # Masking configuration (for input stream)
    parser.add_argument("--number_of_masks", default=50, type=int, 
                        help="Distinct masks per sample.")
    parser.add_argument("--timesteps_per_mask", default=1, type=int, 
                        help="How long each mask is reused.")
    parser.add_argument("--masked_fill", default="random", type=str, 
                        help="Masked pixel fill value or 'random'.")
    parser.add_argument("--mask_ratio", default=0.6, type=float, 
                        help="Fraction of masked patches.")
    parser.add_argument("--patch_size", default=4, type=int, 
                        help="Size of each patch.")
    parser.add_argument("--visible_corrupt", action='store_true', 
                        help="Whether to corrupt visible pixels.")

    # Visualization configuration
    parser.add_argument("--max_time_steps", default=100, type=int, 
                        help="Max timesteps shown in reconstruction grids.")
    parser.add_argument("--hide_input_grid", action="store_true", 
                        help="Skip showing the masked input sequence grid.")
    parser.add_argument("--hide_activation_maps", action="store_true", 
                        help="Skip showing temporal activation-map figures.")
    parser.add_argument("--activation_map_sample_idx", default=0, type=int, 
                        help="Which batch example to use for activation-map figures.")
    parser.add_argument("--activation_map_output_dir",default=None,type=str,
                        help="Optional directory where activation-map PNGs are saved.",)
    args = parser.parse_args()

    target_type = "image" if args.mode == "generative" else "label"
    coalanet = load_pretrained_weights(mode=args.mode)
    examples, targets = visualize_msmnist_examples(
        num_examples=args.num_examples,
        number_of_masks=args.number_of_masks//2,
        timesteps_per_mask=args.timesteps_per_mask,
        mask_ratio=args.mask_ratio,
        masked_fill=_parse_masked_fill_arg(args.masked_fill),
        visible_corrupt=args.visible_corrupt,
        accepted_digits=args.accepted_digits,
        target_type=target_type,
        show=not args.hide_input_grid,
        patch_size=args.patch_size,
    )

    examples2, targets2 = visualize_msmnist_examples(
        num_examples=args.num_examples,
        number_of_masks=args.number_of_masks,
        timesteps_per_mask=args.timesteps_per_mask,
        mask_ratio=args.mask_ratio,
        masked_fill=_parse_masked_fill_arg(args.masked_fill),
        visible_corrupt=args.visible_corrupt,
        accepted_digits=args.accepted_digits,
        target_type=target_type,
        show=not args.hide_input_grid,
        patch_size=args.patch_size,
    )

    examples = torch.cat((examples, examples2), dim=1)
    targets = torch.cat((targets.repeat_interleave(args.number_of_masks//2 * args.timesteps_per_mask, dim=1),
                         targets2.repeat_interleave(args.number_of_masks * args.timesteps_per_mask, dim=1)),
                         dim=1)

    with torch.no_grad():
        model_result = coalanet(examples, return_activation_maps=not args.hide_activation_maps)
    if isinstance(model_result, dict):
        outputs = model_result["outputs_per_timestep"]
        activation_maps = model_result["activation_maps"]
    else:
        outputs = model_result
        activation_maps = None

    import matplotlib.pyplot as plt

    if args.mode == "discriminative":
        logits = stack_temporal_logits(outputs)
        fig = create_temporal_prediction_figure(logits, targets)
    else:
        reconstructions = stack_temporal_reconstructions(outputs)
        fig = create_temporal_reconstruction_figure(reconstructions, targets)
        grid = create_temporal_reconstruction_grid(
            targets,
            examples,
            reconstructions,
            num_examples=args.num_examples,
            max_time_steps=args.max_time_steps,
        )
        _show_image_grid(grid, title="Masked inputs (top); reconstructions (middle); and unmasked targets (bottom)")

    if activation_maps is not None:
        if not (0 <= args.activation_map_sample_idx < examples.shape[0]):
            raise ValueError(
                f"activation_map_sample_idx must be in [0, {examples.shape[0] - 1}], "
                f"got {args.activation_map_sample_idx}."
            )
        save_dir = None
        if args.activation_map_output_dir is not None:
            save_dir = Path(args.activation_map_output_dir)
            save_dir.mkdir(parents=True, exist_ok=True)

        if not args.hide_input_grid or save_dir is not None:
            input_fig = create_activation_input_figure(
                examples,
                sample_idx=args.activation_map_sample_idx,
                max_time_steps=args.max_time_steps,
            )
            if save_dir is not None:
                input_fig.savefig(
                    save_dir / f"{args.mode}_inputs_sample{args.activation_map_sample_idx}.png",
                    dpi=200,
                    bbox_inches="tight",
                )

        for layer_name in activation_maps["Y"].keys():
            layer_fig = create_activation_map_figure(
                activation_maps,
                layer_name=layer_name,
                sample_idx=args.activation_map_sample_idx,
                max_time_steps=args.max_time_steps,
                mode=args.mode,
            )
            if save_dir is not None:
                layer_fig.savefig(
                    save_dir / f"{args.mode}_{layer_name}_sample{args.activation_map_sample_idx}.png",
                    dpi=200,
                    bbox_inches="tight",
                )

    plt.show()
    
