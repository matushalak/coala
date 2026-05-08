import argparse
import json
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from tqdm import tqdm

from coala import DATADIR, bioRNN_logs, dataset_log_dir
from coala.datasets import get_dataloaders
from coala.rnn.utils import EMA
from coala.autoencoder.cnn import ResidualMLP

# TODO
class BioRNN(nn.Module):
    """
    Simplest Hierarchical Recurrent Neural Network.
    Based on this paper (but doing the inpainting version):
        https://www.cell.com/current-biology/pdfExtended/S0960-9822(24)01640-3

    Input -> V1
    V1 -> V1 (recurrent)
    V1 -> V2
    V2 -> V2 (recurrent)
    V2 -> V1
    V1 -> Recon

    TODO: 
    incorporate Dale's law by constraining weights and splitting each layer into excitatory and inhibitory subpopulations.
    Each layer 80/20 excitatory/inhibitory with neurons randomly initialized as E or I, which determines their weights
        - E neurons have non-negative outgoing weights, I neurons have non-positive outgoing weights 
            - (implement via clipping or exp. paramaterization & masking)
        - E neurons receive FF inputs from E neurons of previous layer, recurrent inputs from all neurons in same layer, and FB inputs from E neurons of next layer; 
        - I neurons receive FF inputs from E neurons of previous layer, recurrent inputs from all neurons in same layer, and no FB inputs
            - (implement via masking)
    """

    def __init__(self, 
                 input_features: int = 1,
                 input_size: int = 28*28,
                 V1_features: int = 392, V2_features: int = 196, V4_features: int = 49
                 ):
        super().__init__()
        
        self.W_v1FF = nn.Linear(input_features*input_size, V1_features)
        self.W_v1rec = nn.Linear(V1_features, V1_features)
        self.W_v1FB = nn.Linear(V2_features, V1_features)
        self.W_recon = nn.Linear(V1_features, input_features*input_size)
        
        self.W_v2FF = nn.Linear(V1_features, V2_features)
        self.W_v2rec = nn.Linear(V2_features, V2_features)
        self.W_v2FB = nn.Linear(V4_features, V2_features)

        self.W_v4FF = nn.Linear(V2_features, V4_features)
        self.W_v4rec = nn.Linear(V4_features, V4_features)
        self.W_class = nn.Linear(V4_features, 10)

        
        # Alphas go into sigmoid, so effective alpha is in (0, 1);
        self.V1 = EMA((1, V1_features), alpha=0.0)
        self.V2 = EMA((1, V2_features), alpha=-1.0)
        self.V4 = EMA((1, V4_features), alpha=-2.0)

        self.act = F.relu

    def forward(
        self,
        x: torch.Tensor,
        return_activation_maps: bool = False,
        return_layer_trajectories: bool = False,
        return_final_latent: bool = False,
        return_class_logits: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor] | dict[str, torch.Tensor | dict[str, dict[str, torch.Tensor]]]:
        recon = []
        class_logits = [] if return_class_logits else None
        activation_maps = None
        layer_trajectories = None
        final_latent = None
        if return_activation_maps:
            activation_maps = {
                signal_name: {f"L{i}": [] for i in range(1,4)}
                for signal_name in ("Y", "y_FF", "y_FB")
            }
        if return_layer_trajectories:
            layer_trajectories = {f"L{i}": [] for i in range(1,4)}

        self.V1.reset_state(batch_size=x.shape[0])
        self.V2.reset_state(batch_size=x.shape[0])
        self.V4.reset_state(batch_size=x.shape[0])

        for t in range(x.shape[1]):
            x_t = x[:, t].flatten(start_dim=1)
            
            # All inputs at once
            V1_ff = self.W_v1FF(x_t)
            V1_fb = self.W_v1FB(self.V2.ema)
            V1_rec = self.W_v1rec(self.V1.ema)
            # V2
            V2_ff = self.W_v2FF(self.V1.ema)
            V2_fb = self.W_v2FB(self.V4.ema)
            V2_rec = self.W_v2rec(self.V2.ema)
            # V4
            V4_ff = self.W_v4FF(self.V2.ema)
            V4_rec = self.W_v4rec(self.V4.ema)
            
            # All activations at once
            self.V1(self.act(V1_ff + V1_fb + V1_rec))
            self.V2(self.act(V2_ff + V2_fb + V2_rec))
            self.V4(self.act(V4_ff + V4_rec))
            

            if activation_maps is not None:
                activation_maps["Y"]["L1"].append(self.V1.ema.mean(dim=1))
                activation_maps["y_FF"]["L1"].append(V1_ff.mean(dim=1))
                activation_maps["y_FB"]["L1"].append(V1_fb.mean(dim=1))
                activation_maps["Y"]["L2"].append(self.V2.ema.mean(dim=1))
                activation_maps["y_FF"]["L2"].append(V2_ff.mean(dim=1))
                activation_maps["y_FB"]["L2"].append(V2_fb.mean(dim=1))
                activation_maps["Y"]["L3"].append(self.V4.ema.mean(dim=1))
                activation_maps["y_FF"]["L3"].append(V4_ff.mean(dim=1))
                activation_maps["y_FB"]["L3"].append(V4_rec.mean(dim=1))
            if layer_trajectories is not None:
                layer_trajectories["L1"].append(self.V1.ema.flatten(start_dim=1))
                layer_trajectories["L2"].append(self.V2.ema.flatten(start_dim=1))
                layer_trajectories["L3"].append(self.V4.ema.flatten(start_dim=1))
            if return_final_latent:
                final_latent = self.V4.ema.flatten(start_dim=1)
            
            recon.append(self.W_recon(self.V1.ema).reshape(x[:, t].shape))
            if class_logits is not None:
                class_logits.append(self.W_class(self.V4.ema.squeeze(-1).squeeze(-1)))
        recon_tensor = torch.stack(recon, dim=1)
        class_logits_tensor = torch.stack(class_logits, dim=1) if class_logits is not None else None
        if activation_maps is None and layer_trajectories is None and not return_final_latent and class_logits_tensor is not None:
            return recon_tensor, class_logits_tensor
        result: dict[str, torch.Tensor | dict[str, dict[str, torch.Tensor]] | dict[str, torch.Tensor]] = {
            "recon": recon_tensor,
        }
        if class_logits_tensor is not None:
            result["class_logits"] = class_logits_tensor
        if activation_maps is not None:
            result["activation_maps"] = {
                signal_name: {
                    layer_name: torch.stack(layer_maps, dim=1)
                    for layer_name, layer_maps in per_signal_maps.items()
                }
                for signal_name, per_signal_maps in activation_maps.items()
            }
        if layer_trajectories is not None:
            result["layer_trajectories"] = {
                layer_name: torch.stack(layer_values, dim=1)
                for layer_name, layer_values in layer_trajectories.items()
            }
        if final_latent is not None:
            result["final_latent"] = final_latent
        return result


def prepare_batch(
    batch: tuple[torch.Tensor, dict[str, torch.Tensor]],
    device: torch.device,
    rollout_length: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    x, targets = batch
    x = x.to(device)
    clean_image = targets["image"].to(device)
    labels = targets["label"].to(device)
    contrastive_positive_index = None
    if isinstance(targets, dict) and "contrastive_positive_index" in targets:
        contrastive_positive_index = targets["contrastive_positive_index"].to(device)

    if rollout_length is not None:
        x = x[:, :rollout_length]
        if clean_image.dim() == 5:
            clean_image = clean_image[:, :rollout_length]
        if labels.dim() == 2:
            labels = labels[:, :rollout_length]

    return x, clean_image, labels, contrastive_positive_index


def expand_clean_targets(clean_image: torch.Tensor, n_steps: int) -> torch.Tensor:
    if clean_image.dim() == 5:
        if clean_image.shape[1] != n_steps:
            raise ValueError(
                "Per-timestep clean targets must match the reconstruction time dimension, "
                f"got {clean_image.shape[1]} vs {n_steps}."
            )
        return clean_image
    return clean_image.unsqueeze(1).expand(-1, n_steps, -1, -1, -1)


def expand_label_targets(labels: torch.Tensor, n_steps: int) -> torch.Tensor:
    if labels.dim() == 2:
        if labels.shape[1] != n_steps:
            raise ValueError(
                "Per-timestep labels must match the logits time dimension, "
                f"got {labels.shape[1]} vs {n_steps}."
            )
        return labels
    return labels.unsqueeze(1).expand(-1, n_steps)


def nt_xent_loss(
    latents: torch.Tensor,
    positive_index: torch.Tensor,
    temperature: float = 0.1,
) -> torch.Tensor:
    if latents.ndim != 3:
        raise ValueError(f"Expected latents with shape (batch, time, dim), got {tuple(latents.shape)}.")
    batch_size, _, _ = latents.shape
    if positive_index.shape != (batch_size,):
        raise ValueError(
            "positive_index must have shape (batch,), "
            f"got {tuple(positive_index.shape)} for batch size {batch_size}."
        )
    if batch_size < 2:
        raise ValueError("NT-Xent requires at least two paired samples in the batch.")

    normalized = F.normalize(latents, dim=-1)
    similarities = torch.matmul(normalized.transpose(0, 1), normalized.transpose(0, 1).transpose(1, 2))
    similarities = similarities / temperature
    eye = torch.eye(batch_size, dtype=torch.bool, device=latents.device).unsqueeze(0)
    similarities = similarities.masked_fill(eye, float("-inf"))
    targets = positive_index.unsqueeze(0).expand(latents.shape[1], -1)
    return F.cross_entropy(
        similarities.reshape(-1, batch_size),
        targets.reshape(-1),
        reduction="none",
    ).view(latents.shape[1], batch_size).mean(dim=1)


def compute_losses(
    recon: torch.Tensor,
    class_logits: torch.Tensor | None,
    clean_image: torch.Tensor,
    labels: torch.Tensor,
    t0_weight: float = 0.5,
    latents: torch.Tensor | None = None,
    contrastive_positive_index: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    weights = torch.linspace(t0_weight, 1.0, steps=recon.shape[1], device=recon.device)
    target_recon = expand_clean_targets(clean_image, recon.shape[1])
    recon_loss = F.smooth_l1_loss(recon, target_recon, reduction = 'none', beta = 0.0) # l1 if beta = 0 / huber if beta > 0
    recon_loss = (recon_loss.mean(dim=[2, 3, 4]) * weights).mean()
    
    if latents is None or contrastive_positive_index is None:
        if class_logits is None:
            raise ValueError("class_logits are required when contrastive loss is not being used.")
        target_labels = expand_label_targets(labels, class_logits.shape[1])
        class_loss = F.cross_entropy(class_logits.view(-1, class_logits.shape[-1]),target_labels.reshape(-1),
                                    reduction='none').view(class_logits.shape[0], class_logits.shape[1])
        class_loss = (class_loss * weights).mean()
    else:
        class_loss = nt_xent_loss(latents, contrastive_positive_index)
        class_loss = (class_loss * weights).mean()
        # class_loss = torch.ones_like(recon_loss) 

    return recon_loss, class_loss, recon_loss + (0.05*class_loss)


def make_reconstruction_grid(
    masked_inputs: torch.Tensor,
    recon: torch.Tensor,
    clean_image: torch.Tensor,
    num_examples: int = 4,
) -> torch.Tensor:
    num_examples = min(num_examples, masked_inputs.shape[0])
    n_steps = masked_inputs.shape[1]
    clean_sequence = expand_clean_targets(clean_image[:num_examples], n_steps)
    rows = []
    for idx in range(num_examples):
        rows.extend(
            [
                masked_inputs[idx].float(),
                recon[idx].float(),
                clean_sequence[idx].float(),
            ]
        )
    panel = torch.cat(rows, dim=0).detach().cpu()
    return make_grid(
        panel,
        nrow=n_steps,
        normalize=True,
        value_range=(-1.0, 1.0),
        pad_value=0.0,
    )


def create_run_dir(log_root: str | Path) -> Path:
    log_root = Path(log_root)
    log_root.mkdir(parents=True, exist_ok=True)
    existing_run_ids = []
    for child in log_root.iterdir():
        if child.is_dir() and child.name.isdigit():
            existing_run_ids.append(int(child.name))
    next_run_id = 1 if not existing_run_ids else max(existing_run_ids) + 1
    run_dir = log_root / str(next_run_id)
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def save_run_metadata(run_dir: Path, args: argparse.Namespace | None = None):
    if args is None:
        return
    metadata = vars(args).copy()
    metadata["run_dir"] = str(run_dir)
    metadata["created_at"] = datetime.now().isoformat(timespec="seconds")
    with open(run_dir / "args.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)


def save_checkpoint(
    checkpoint_path: Path,
    model: BioRNN,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    train_loss: float,
    val_loss: float,
    val_accuracy_percent: float,
):
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_accuracy_percent": val_accuracy_percent,
        },
        checkpoint_path,
    )


@torch.no_grad()
def evaluate(
    model: BioRNN,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    writer: SummaryWriter,
    epoch: int,
    num_examples: int = 4,
    max_batches: int | None = None,
    t0_weight: float = 0.5,
) -> tuple[float, float, float, float]:
    model.eval()
    total_loss = 0.0
    total_recon_loss = 0.0
    total_class_loss = 0.0
    total_samples = 0
    correct_by_step = None
    recon_grid = None

    for batch_idx, batch in enumerate(dataloader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        masked_inputs, clean_image, labels, contrastive_positive_index = prepare_batch(batch, device)
        model_outputs = model(
            masked_inputs,
            return_layer_trajectories=contrastive_positive_index is not None,
            return_final_latent=False,
        )
        recon = model_outputs["recon"] if isinstance(model_outputs, dict) else model_outputs[0]
        class_logits = model_outputs["class_logits"] if isinstance(model_outputs, dict) else model_outputs[1]
        latents = None if contrastive_positive_index is None else model_outputs["layer_trajectories"]["L3"]
        recon_loss, class_loss, loss = compute_losses(
            recon,
            class_logits,
            clean_image,
            labels,
            t0_weight=t0_weight,
            latents=latents,
            contrastive_positive_index=contrastive_positive_index,
        )

        batch_size = masked_inputs.shape[0]
        total_loss += loss.item() * batch_size
        total_recon_loss += recon_loss.item() * batch_size
        total_class_loss += class_loss.item() * batch_size
        total_samples += batch_size

        preds = class_logits.argmax(dim=-1)
        target_labels = expand_label_targets(labels, class_logits.shape[1])
        matches = preds.eq(target_labels)
        if correct_by_step is None:
            correct_by_step = torch.zeros(class_logits.shape[1], dtype=torch.float32)
        correct_by_step += matches.sum(dim=0).to(dtype=torch.float32).cpu()

        if batch_idx == 0:
            recon_grid = make_reconstruction_grid(masked_inputs, recon, clean_image, num_examples=num_examples)

    mean_loss = total_loss / total_samples
    mean_recon_loss = total_recon_loss / total_samples
    mean_class_loss = total_class_loss / total_samples
    step_accuracy_percent = 100.0 * correct_by_step / total_samples
    final_accuracy_percent = step_accuracy_percent[-1].item()
    log_step = epoch + 1

    writer.add_scalar("val_loss", mean_loss, log_step)
    writer.add_scalar("val_recon_loss", mean_recon_loss, log_step)
    writer.add_scalar("val_class_loss", mean_class_loss, log_step)
    writer.add_scalar("val_accuracy_percent", final_accuracy_percent, log_step)
    if recon_grid is not None:
        writer.add_image("val_reconstruction_across_steps", recon_grid, log_step)

    return mean_recon_loss, mean_class_loss, mean_loss, final_accuracy_percent


def train(
    model: BioRNN,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    n_epochs: int,
    device: torch.device,
    log_dir: str | Path,
    lr: float = 1e-3,
    max_train_batches: int | None = None,
    max_val_batches: int | None = None,
    args: argparse.Namespace | None = None,
):
    model = model.to(device)
    run_dir = create_run_dir(log_dir)
    writer = SummaryWriter(log_dir=str(run_dir))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_val_loss = float("inf")
    save_run_metadata(run_dir, args=args)

    print(f"Logging to: {run_dir}")
    for epoch in range(n_epochs):
        model.train()
        total_loss = 0.0
        total_recon_loss = 0.0
        total_class_loss = 0.0
        total_samples = 0

        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1}/{n_epochs}")):
            if max_train_batches is not None and batch_idx >= max_train_batches:
                break

            rollout_length = torch.randint(low=8, high=20, size=(1,), device=device).item()
            masked_inputs, clean_image, labels, contrastive_positive_index = prepare_batch(
                batch,
                device,
                rollout_length=rollout_length,
            )
            model_outputs = model(
                masked_inputs,
                return_layer_trajectories=contrastive_positive_index is not None,
                return_final_latent=False,
                return_class_logits=contrastive_positive_index is None,
            )
            recon = model_outputs["recon"] if isinstance(model_outputs, dict) else model_outputs[0]
            class_logits = None if contrastive_positive_index is not None else (model_outputs["class_logits"] if isinstance(model_outputs, dict) else model_outputs[1])
            latents = None if contrastive_positive_index is None else model_outputs["layer_trajectories"]["L3"]
            recon_loss, class_loss, loss = compute_losses(
                recon,
                class_logits,
                clean_image,
                labels,
                t0_weight=args.t0_weight if args is not None else 0.5,
                latents=latents,
                contrastive_positive_index=contrastive_positive_index,
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            batch_size = masked_inputs.shape[0]
            total_loss += loss.item() * batch_size
            total_recon_loss += recon_loss.item() * batch_size
            total_class_loss += class_loss.item() * batch_size
            total_samples += batch_size

        train_loss = total_loss / total_samples
        train_recon_loss = total_recon_loss / total_samples
        train_class_loss = total_class_loss / total_samples
        log_step = epoch + 1
        writer.add_scalar("train_loss", train_loss, log_step)
        writer.add_scalar("train_recon_loss", train_recon_loss, log_step)
        writer.add_scalar("train_class_loss", train_class_loss, log_step)

        val_recon_loss, val_class_loss, val_loss, val_accuracy_percent = evaluate(
            model=model,
            dataloader=val_loader,
            device=device,
            writer=writer,
            epoch=epoch,
            max_batches=max_val_batches,
            num_examples=args.num_examples if args is not None else 5,
            t0_weight=args.t0_weight if args is not None else 0.5
        )
        print(
            f"Epoch {epoch + 1}: "
            f"train_loss={train_loss:.4f}, "
            f"train_recon={train_recon_loss:.4f}, "
            f"train_class={train_class_loss:.4f}, "
            f"val_loss={val_loss:.4f}, "
            f"val_recon={val_recon_loss:.4f}, "
            f"val_class={val_class_loss:.4f}, "
            f"val_acc={val_accuracy_percent:.2f}%"
        )
        print("Alpha values -", 
              f"V1: {F.sigmoid(model.V1.alpha):.6f}, V2: {F.sigmoid(model.V2.alpha):.6f}, V4: {F.sigmoid(model.V4.alpha):.6f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                checkpoint_path=run_dir / "best_val_loss.pt",
                model=model,
                optimizer=optimizer,
                epoch=epoch + 1,
                train_loss=train_loss,
                val_loss=val_loss,
                val_accuracy_percent=val_accuracy_percent,
            )
        writer.flush()

    writer.close()

def parse_masked_fill(value: str) -> str | float:
    if value == "random":
        return value
    return float(value)

def main():
    args = build_argparser().parse_args()
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    train_loader, val_loader, _ = get_dataloaders(
        "msmnist",
        batch_size=args.batch_size,
        root=args.data_dir,
        num_workers=args.num_workers,
        patch_size=args.patch_size,
        mask_ratio=args.mask_ratio,
        mask_pattern=args.mask_pattern,
        masked_fill=parse_masked_fill(args.masked_fill),
        noise_sigma=args.noise_sigma,
        visible_corrupt=args.visible_corrupt,
        number_of_masks=args.number_of_masks,
        timesteps_per_mask=args.timesteps_per_mask,
        num_digits=args.num_digits,
        image_visibility=args.image_visibility,
        contrastive=args.contrastive,
        target_type="both",
    )
    model = BioRNN()
    train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        n_epochs=args.epochs,
        device=device,
        lr=args.lr,
        log_dir=dataset_log_dir(args.log_dir, "msmnist"),
        max_train_batches=args.max_train_batches,
        max_val_batches=args.max_val_batches,
        args=args,
    )

def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--patch_size", type=int, default=4)
    parser.add_argument("--mask_ratio", type=float, default=0.5)
    parser.add_argument("--mask_pattern", type=str, default="random", choices=("random", "structured"))
    parser.add_argument("--masked_fill", type=str, default="random")
    parser.add_argument("--noise_sigma", type=float, default=0.25)
    parser.add_argument("--number_of_masks", type=int, default=1)
    parser.add_argument("--timesteps_per_mask", type=int, default=5)
    parser.add_argument("--num_digits", type=int, default=1)
    parser.add_argument("--image_visibility", type=str, default="all")
    parser.add_argument("--visible_corrupt", action=argparse.BooleanOptionalAction)
    parser.add_argument("--contrastive", action=argparse.BooleanOptionalAction)
    parser.add_argument("--max_train_batches", type=int, default=None)
    parser.add_argument("--max_val_batches", type=int, default=None)
    parser.add_argument("--data_dir", type=str, default=str(DATADIR))
    parser.add_argument("--log_dir", type=str, default=bioRNN_logs)
    parser.add_argument("--num_examples", type=int, default=5)
    parser.add_argument("--t0_weight", type=float, default=0.5)
    return parser

if __name__ == "__main__":
    main()
