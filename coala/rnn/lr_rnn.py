import argparse
import json
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid, save_image
from tqdm import tqdm

from coala import DATADIR, dataset_log_dir, lrRNN_logs
from coala.datasets import get_dataloaders
from coala.rnn.utils import EMA

# TODO: can we get hierarchical organization (off-diagonal blocks in connectivity matrix) + convolution-like RFs to emerge
# just by nature of the task, inductive biases in the data and architecture?
class lrRNN(nn.Module):
    """
    Simplest possible RNN pool architecture, where we try to force hierarchical organization to emerge somehow
    """

    def __init__(self, input_features: int = 1, V1_features: int = 16, V2_features: int = 32, V4_features: int = 256,
                 rank:int = 4):
        super().__init__()
        self.V1_features = V1_features
        self.V2_features = V2_features
        self.V4_features = V4_features
        
        self.W_input = nn.Linear(28*28*input_features, 14*14*V1_features)
        self.W_recon = nn.Linear(14*14*V1_features, 28*28*input_features)
        self.W_class = nn.Linear(1*1*V4_features, 10)

        # Low-rank approximation of full connectivity matrix
        self.W_L = nn.Linear(14*14*V1_features+7*7*V2_features+1*1*V4_features, rank)
        self.W_R = nn.Linear(rank, 14*14*V1_features+7*7*V2_features+1*1*V4_features)

        # Alphas go into sigmoid, so effective alpha is in (0, 1);
        self.V1 = EMA((1, V1_features*14*14), alpha=1.0)
        self.V1_slice = slice(0, V1_features*14*14)
        
        self.V2 = EMA((1, V2_features*7*7), alpha=-2.5)
        self.V2_slice = slice(V1_features*14*14, 
                              V1_features*14*14 + V2_features*7*7)
        
        self.V4 = EMA((1, V4_features*1*1), alpha=-5.0)
        self.V4_slice = slice(V1_features*14*14 + V2_features*7*7, 
                              V1_features*14*14 + V2_features*7*7 + V4_features*1*1)

    @torch.no_grad()
    def full_connectivity_matrix(self) -> torch.Tensor:
        return self.W_R.weight @ self.W_L.weight

    def forward(
        self,
        x: torch.Tensor,
        return_activation_maps: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor] | dict[str, torch.Tensor | dict[str, dict[str, torch.Tensor]]]:
        recon = []
        class_logits = []
        activation_maps = None
        if return_activation_maps:
            activation_maps = {
                signal_name: {f"L{i}": [] for i in range(3)}
                for signal_name in ("Y", "y_FF", "y_FB")
            }

        self.V1.reset_state(batch_size=x.shape[0])
        self.V2.reset_state(batch_size=x.shape[0])
        self.V4.reset_state(batch_size=x.shape[0])

        network_state = torch.zeros(x.shape[0], self.V1.ema.shape[1] + self.V2.ema.shape[1] + self.V4.ema.shape[1], 
                                    device=x.device)
        for t in range(x.shape[1]):
            x_t = x[:, t]
            
            # Compute interactions
            recurrent_drive = self.W_R(self.W_L(network_state))
            network_state = recurrent_drive
            # Give input to network
            V1_ff = self.W_input(x_t.view(x_t.shape[0], -1))
            network_state[:, self.V1_slice] += V1_ff
            # Compute activations
            network_state = F.gelu(network_state)
            
            # Temporally integrate activations
            network_state[:, self.V1_slice] = self.V1(network_state[:, self.V1_slice])
            network_state[:, self.V2_slice] = self.V2(network_state[:, self.V2_slice])
            network_state[:, self.V4_slice] = self.V4(network_state[:, self.V4_slice])

            if activation_maps is not None:
                zeros_v2 = torch.zeros(
                    x_t.shape[0],
                    self.V2_features,
                    7,
                    7,
                    device=x.device,
                    dtype=x.dtype,
                )
                zeros_v4 = torch.zeros(
                    x_t.shape[0],
                    self.V4_features,
                    1,
                    1,
                    device=x.device,
                    dtype=x.dtype,
                )
                v1_y = self.V1.ema.view(x_t.shape[0], self.V1_features, 14, 14)
                v2_y = self.V2.ema.view(x_t.shape[0], self.V2_features, 7, 7)
                v4_y = self.V4.ema.view(x_t.shape[0], self.V4_features, 1, 1)
                v1_fb = recurrent_drive[:, self.V1_slice].view(x_t.shape[0], self.V1_features, 14, 14)
                v2_fb = recurrent_drive[:, self.V2_slice].view(x_t.shape[0], self.V2_features, 7, 7)
                v4_fb = recurrent_drive[:, self.V4_slice].view(x_t.shape[0], self.V4_features, 1, 1)
                v1_ff_map = V1_ff.view(x_t.shape[0], self.V1_features, 14, 14)
                activation_maps["Y"]["L0"].append(v1_y.mean(dim=1))
                activation_maps["y_FF"]["L0"].append(v1_ff_map.mean(dim=1))
                activation_maps["y_FB"]["L0"].append(v1_fb.mean(dim=1))
                activation_maps["Y"]["L1"].append(v2_y.mean(dim=1))
                activation_maps["y_FF"]["L1"].append(zeros_v2.mean(dim=1))
                activation_maps["y_FB"]["L1"].append(v2_fb.mean(dim=1))
                activation_maps["Y"]["L2"].append(v4_y.mean(dim=1))
                activation_maps["y_FF"]["L2"].append(zeros_v4.mean(dim=1))
                activation_maps["y_FB"]["L2"].append(v4_fb.mean(dim=1))

            recon.append(self.W_recon(self.V1.ema).view(x_t.shape))
            class_logits.append(self.W_class(self.V4.ema.squeeze(-1).squeeze(-1)))
        recon_tensor = torch.stack(recon, dim=1)
        class_logits_tensor = torch.stack(class_logits, dim=1)
        if activation_maps is None:
            return recon_tensor, class_logits_tensor
        return {
            "recon": recon_tensor,
            "class_logits": class_logits_tensor,
            "activation_maps": {
                signal_name: {
                    layer_name: torch.stack(layer_maps, dim=1)
                    for layer_name, layer_maps in per_signal_maps.items()
                }
                for signal_name, per_signal_maps in activation_maps.items()
            },
        }


def prepare_batch(
    batch: tuple[torch.Tensor, dict[str, torch.Tensor]],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x, targets = batch
    x = x.to(device)
    clean_image = targets["image"].to(device)
    labels = targets["label"].to(device)
    return x, clean_image, labels


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


def compute_losses(
    recon: torch.Tensor,
    class_logits: torch.Tensor,
    clean_image: torch.Tensor,
    labels: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    target_recon = expand_clean_targets(clean_image, recon.shape[1])
    target_labels = expand_label_targets(labels, class_logits.shape[1])
    recon_loss = F.mse_loss(recon, target_recon)
    class_loss = F.cross_entropy(
        class_logits.reshape(-1, class_logits.shape[-1]),
        target_labels.reshape(-1),
    )
    return recon_loss, class_loss, recon_loss + class_loss


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


@torch.no_grad()
def connectivity_matrix_to_image(connectivity: torch.Tensor) -> torch.Tensor:
    connectivity = connectivity.detach().float().cpu()
    scale = connectivity.abs().max().clamp_min(1e-8)
    normalized = (connectivity / scale).clamp_(-1.0, 1.0)
    pos = normalized.clamp_min(0.0)
    neg = (-normalized).clamp_min(0.0)
    red = 1.0 - neg
    green = 1.0 - (pos + neg)
    blue = 1.0 - pos
    return torch.stack((red, green, blue), dim=0)


@torch.no_grad()
def log_connectivity_matrix(
    model: lrRNN,
    writer: SummaryWriter,
    run_dir: Path,
    epoch: int,
) -> None:
    connectivity = model.full_connectivity_matrix()
    image = connectivity_matrix_to_image(connectivity)
    log_step = epoch + 1
    writer.add_image("connectivity/full_matrix", image, log_step)
    writer.add_scalar("connectivity/max_abs_weight", connectivity.abs().max().item(), log_step)


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
    model: lrRNN,
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
    model: lrRNN,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    writer: SummaryWriter,
    epoch: int,
    num_examples: int = 4,
    max_batches: int | None = None,
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

        masked_inputs, clean_image, labels = prepare_batch(batch, device)
        recon, class_logits = model(masked_inputs)
        recon_loss, class_loss, loss = compute_losses(recon, class_logits, clean_image, labels)

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
    model: lrRNN,
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

            masked_inputs, clean_image, labels = prepare_batch(batch, device)
            recon, class_logits = model(masked_inputs)
            recon_loss, class_loss, loss = compute_losses(recon, class_logits, clean_image, labels)

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
        # log_connectivity_matrix(model, writer, run_dir, epoch)

        val_recon_loss, val_class_loss, val_loss, val_accuracy_percent = evaluate(
            model=model,
            dataloader=val_loader,
            device=device,
            writer=writer,
            epoch=epoch,
            max_batches=max_val_batches,
            num_examples=args.num_examples if args is not None else 5,
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
        print(f"Alpha values - V1: {F.sigmoid(model.V1.alpha):.6f}, V2: {F.sigmoid(model.V2.alpha):.6f}, V4: {F.sigmoid(model.V4.alpha):.6f}")
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
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
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
        target_type="both",
    )
    model = lrRNN(rank=args.rank)
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
    parser.add_argument("--max_train_batches", type=int, default=None)
    parser.add_argument("--max_val_batches", type=int, default=None)
    parser.add_argument("--data_dir", type=str, default=str(DATADIR))
    parser.add_argument("--log_dir", type=str, default=lrRNN_logs)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--num_examples", type=int, default=5)
    return parser

if __name__ == "__main__":
    main()
