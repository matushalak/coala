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

from coala import DATADIR, RNN_logs, dataset_log_dir
from coala.datasets import get_dataloaders
from coala.rnn.utils import EMA

# TODO
# try staged approach: first only train on clean images
# then denoise
# then occlusion

class hConvRNN(nn.Module):
    """
    Simplest possible hierarchical Conv RNN architecture.
    """

    def __init__(self, input_features: int = 1, V1_features:int = 16, V2_features: int = 32, V4_features: int = 64):
        super().__init__()
        self.W_recon = nn.ConvTranspose2d(V1_features, input_features, kernel_size=5, padding=2, stride=2, output_padding=1)
        self.W_v1FF = nn.Conv2d(input_features, V1_features, kernel_size=5, padding=2, stride=2)
        self.W_v1FB = nn.ConvTranspose2d(V2_features, V1_features, kernel_size=3, padding=1, stride=2, output_padding=1)
        
        self.W_v2FF = nn.Conv2d(V1_features, V2_features, kernel_size=3, padding=1, stride=2)
        self.W_v2FB = nn.ConvTranspose2d(V4_features, V2_features, kernel_size=7, padding=0, stride=1)

        self.W_v4FF = nn.Conv2d(V2_features, V4_features, kernel_size=7, padding=0, stride=1)
        self.W_class = nn.Linear(V4_features, 10)

        self.V1 = EMA((1, V1_features, 14, 14), alpha=0.2)
        self.V2 = EMA((1, V2_features, 7, 7), alpha=0.1)
        self.V4 = EMA((1, V4_features, 1, 1), alpha=0.05)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        recon = []
        class_logits = []

        self.V1.reset_state(batch_size=x.shape[0])
        self.V2.reset_state(batch_size=x.shape[0])
        self.V4.reset_state(batch_size=x.shape[0])

        for t in range(x.shape[1]):
            x_t = x[:, t]
            
            # V1
            V1_ff = self.W_v1FF(x_t)
            V1_fb = self.W_v1FB(self.V2.ema)
            self.V1(F.gelu(V1_ff + V1_fb))
            # V2
            V2_ff = self.W_v2FF(self.V1.ema)
            V2_fb = self.W_v2FB(self.V4.ema)
            self.V2(F.gelu(V2_ff + V2_fb))
            # V4
            V4_ff = self.W_v4FF(self.V2.ema)
            self.V4(F.gelu(V4_ff))
            
            recon.append(self.W_recon(self.V1.ema))
            class_logits.append(self.W_class(self.V4.ema.squeeze(-1).squeeze(-1)))

        return torch.stack(recon, dim=1), torch.stack(class_logits, dim=1)


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
    return clean_image.unsqueeze(1).expand(-1, n_steps, -1, -1, -1)


def compute_losses(
    recon: torch.Tensor,
    class_logits: torch.Tensor,
    clean_image: torch.Tensor,
    labels: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    target_recon = expand_clean_targets(clean_image, recon.shape[1])
    target_labels = labels.unsqueeze(1).expand(-1, class_logits.shape[1])
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
    model: hConvRNN,
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
    model: hConvRNN,
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
        matches = preds.eq(labels.unsqueeze(1))
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
    model: hConvRNN,
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

        val_recon_loss, val_class_loss, val_loss, val_accuracy_percent = evaluate(
            model=model,
            dataloader=val_loader,
            device=device,
            writer=writer,
            epoch=epoch,
            max_batches=max_val_batches,
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
        visible_corrupt=args.visible_corrupt,
        number_of_masks=args.number_of_masks,
        timesteps_per_mask=args.timesteps_per_mask,
        target_type="both",
    )
    model = hConvRNN()
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
    parser.add_argument("--number_of_masks", type=int, default=1)
    parser.add_argument("--timesteps_per_mask", type=int, default=5)
    parser.add_argument("--visible_corrupt", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--max_train_batches", type=int, default=None)
    parser.add_argument("--max_val_batches", type=int, default=None)
    parser.add_argument("--data_dir", type=str, default=str(DATADIR))
    parser.add_argument("--log_dir", type=str, default=RNN_logs)
    return parser

if __name__ == "__main__":
    main()
