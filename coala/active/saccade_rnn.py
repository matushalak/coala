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

from coala import DATADIR, rCNN_logs
from coala.datasets.mnist import mnist
from coala.rnn.utils import EMA


class SigReg(nn.Module):
    def __init__(self, knots: int = 17, random_projections: int = 64, max_samples: int | None = 1024):
        super().__init__()
        if knots < 2:
            raise ValueError("knots must be >= 2.")
        if max_samples is not None and max_samples <= 0:
            raise ValueError("max_samples must be > 0 when provided.")

        self.random_projections = random_projections
        self.max_samples = max_samples
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, proj: torch.Tensor) -> torch.Tensor:
        if proj.ndim < 2:
            raise ValueError(f"Expected proj.ndim >= 2, got shape {tuple(proj.shape)}.")
        if self.max_samples is not None and proj.size(-2) > self.max_samples:
            idx = torch.randperm(proj.size(-2), device=proj.device)[: self.max_samples]
            proj = proj.index_select(-2, idx)

        t = self.t.to(device=proj.device, dtype=proj.dtype)
        phi = self.phi.to(device=proj.device, dtype=proj.dtype)
        weights = self.weights.to(device=proj.device, dtype=proj.dtype)

        A = torch.randn(proj.size(-1), self.random_projections, device=proj.device, dtype=proj.dtype)
        A = A.div_(A.norm(p=2, dim=0))
        x_t = (proj @ A).unsqueeze(-1) * t
        err = (x_t.cos().mean(dim=-3) - phi).square() + x_t.sin().mean(dim=-3).square()
        statistic = (err @ weights) * proj.size(-2)
        return statistic.mean()


class MNISTSaccadeRNN(nn.Module):
    """
    Minimal recurrent active-vision prototype for MNIST.

    The model keeps a recurrent V1 state at 7x7 and a top 1x1 state split into:
    - representation channels optimized for temporal stability
    - saccade-control channels used to predict the next fixation
    """

    def __init__(
        self,
        input_features: int = 1,
        v1_features: int = 32,
        rep_features: int = 16,
        saccade_features: int = 8,
        fovea_scale: float = 0.35,
        peripheral_scale: float = 0.9,
        blur_sigma: float = 1.5,
        fovea_sharpness: float = 12.0,
    ):
        super().__init__()
        total_top_features = rep_features + saccade_features
        if total_top_features <= 0:
            raise ValueError("rep_features + saccade_features must be > 0.")
        if not (0.0 < fovea_scale <= peripheral_scale):
            raise ValueError("Expected 0 < fovea_scale <= peripheral_scale.")

        self.input_features = input_features
        self.v1_features = v1_features
        self.rep_features = rep_features
        self.saccade_features = saccade_features
        self.total_top_features = total_top_features
        self.fovea_scale = fovea_scale
        self.peripheral_scale = peripheral_scale
        self.blur_sigma = blur_sigma
        self.fovea_sharpness = fovea_sharpness

        self.W_v1FF = nn.Conv2d(input_features, v1_features, kernel_size=4, stride=4)
        self.W_v1rec = nn.Conv2d(v1_features, v1_features, kernel_size=1)
        self.W_v1FB = nn.ConvTranspose2d(total_top_features, v1_features, kernel_size=7, stride=4)

        self.W_topFF = nn.Conv2d(v1_features, total_top_features, kernel_size=7)
        self.W_toprec = nn.Conv2d(total_top_features, total_top_features, kernel_size=1)
        self.W_saccade = nn.Linear(total_top_features, 2)

        self.V1 = EMA((1, v1_features, 7, 7), alpha=0.0)
        self.TOP = EMA((1, total_top_features, 1, 1), alpha=-1.0)

        self.act = F.relu

    @staticmethod
    def _build_base_grid(batch_size: int, height: int, width: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        ys = torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype)
        xs = torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        base_grid = torch.stack((grid_x, grid_y), dim=-1)
        return base_grid.unsqueeze(0).expand(batch_size, -1, -1, -1)

    def _gaussian_blur(self, x: torch.Tensor, sigma: float) -> torch.Tensor:
        if sigma <= 0.0:
            return x

        radius = max(1, int(round(3.0 * sigma)))
        coords = torch.arange(-radius, radius + 1, device=x.device, dtype=x.dtype)
        kernel_1d = torch.exp(-(coords.square()) / (2.0 * sigma * sigma))
        kernel_1d = kernel_1d / kernel_1d.sum()
        kernel_x = kernel_1d.view(1, 1, 1, -1).expand(x.shape[1], 1, 1, -1)
        kernel_y = kernel_1d.view(1, 1, -1, 1).expand(x.shape[1], 1, -1, 1)
        blurred = F.conv2d(x, kernel_x, padding=(0, radius), groups=x.shape[1])
        blurred = F.conv2d(blurred, kernel_y, padding=(radius, 0), groups=x.shape[1])
        return blurred

    def foveate(self, x: torch.Tensor, fixation: torch.Tensor) -> torch.Tensor:
        if fixation.ndim != 2 or fixation.shape[1] != 2:
            raise ValueError(f"Expected fixation with shape (batch, 2), got {tuple(fixation.shape)}.")

        batch_size, _, height, width = x.shape
        fixation = fixation.clamp(0.0, 1.0)
        fixation_center = fixation.mul(2.0).sub(1.0).view(batch_size, 1, 1, 2)
        base_grid = self._build_base_grid(batch_size, height, width, x.device, x.dtype)

        fovea_grid = fixation_center + self.fovea_scale * base_grid
        peripheral_grid = fixation_center + self.peripheral_scale * base_grid

        fovea = F.grid_sample(x, fovea_grid, mode="bilinear", padding_mode="border", align_corners=True)
        peripheral_source = self._gaussian_blur(x, sigma=self.blur_sigma)
        peripheral = F.grid_sample(
            peripheral_source,
            peripheral_grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )

        radial_dist = base_grid.square().sum(dim=-1, keepdim=True).sqrt()
        blend = torch.sigmoid((0.55 - radial_dist) * self.fovea_sharpness).movedim(-1, 1)
        return blend * fovea + (1.0 - blend) * peripheral

    def forward(
        self,
        imgs: torch.Tensor,
        num_steps: int,
        initial_fixation: torch.Tensor | None = None,
        return_retina_sequence: bool = False,
    ) -> dict[str, torch.Tensor]:
        if imgs.ndim != 4:
            raise ValueError(f"Expected imgs with shape (batch, channels, height, width), got {tuple(imgs.shape)}.")
        if num_steps <= 0:
            raise ValueError("num_steps must be > 0.")

        batch_size = imgs.shape[0]
        fixation = (
            torch.full((batch_size, 2), 0.5, device=imgs.device, dtype=imgs.dtype)
            if initial_fixation is None
            else initial_fixation.to(device=imgs.device, dtype=imgs.dtype).clamp(0.0, 1.0)
        )

        self.V1.reset_state(batch_size=batch_size)
        self.TOP.reset_state(batch_size=batch_size)

        retina_sequence = []
        v1_sequence = []
        representation_sequence = []
        saccade_sequence = []

        for _ in range(num_steps):
            retina = self.foveate(imgs, fixation)

            v1_drive = self.W_v1FF(retina) + self.W_v1rec(self.V1.ema) + self.W_v1FB(self.TOP.ema)
            v1_state = self.V1(self.act(v1_drive))
            top_drive = self.W_topFF(v1_state) + self.W_toprec(self.TOP.ema)
            top_state = self.TOP(self.act(top_drive))

            representation = top_state[:, : self.rep_features].flatten(start_dim=1)
            saccade_hidden = top_state[:, self.rep_features :].flatten(start_dim=1)
            fixation = torch.sigmoid(self.W_saccade(torch.cat((representation, saccade_hidden), dim=1)))

            if return_retina_sequence:
                retina_sequence.append(retina)
            v1_sequence.append(v1_state)
            representation_sequence.append(representation)
            saccade_sequence.append(fixation)

        outputs = {
            "representations": torch.stack(representation_sequence, dim=1),
            "saccades": torch.stack(saccade_sequence, dim=1),
            "final_representation": representation_sequence[-1],
            "final_fixation": saccade_sequence[-1],
            "v1_trajectory": torch.stack([state.flatten(start_dim=1) for state in v1_sequence], dim=1),
        }
        if return_retina_sequence:
            outputs["retina_sequence"] = torch.stack(retina_sequence, dim=1)
        return outputs


def stability_loss(representations: torch.Tensor) -> torch.Tensor:
    if representations.shape[1] < 2:
        return representations.new_zeros(())
    return (representations[:, 1:] - representations[:, :-1].detach()).pow(2).mean()


def saccade_variance_reward(saccades: torch.Tensor) -> torch.Tensor:
    if saccades.shape[1] < 2:
        return saccades.new_zeros(())
    per_image_variance = saccades.var(dim=1, unbiased=False).sum(dim=-1)
    return per_image_variance.mean()


def compute_ssl_losses(
    outputs: dict[str, torch.Tensor],
    sigreg: SigReg,
    sigreg_weight: float,
    saccade_var_weight: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    reps = outputs["representations"]
    saccades = outputs["saccades"]

    temporal = stability_loss(reps)
    sig = sigreg(outputs["final_representation"])
    saccade_var = saccade_variance_reward(saccades)

    total = temporal + sigreg_weight * sig - saccade_var_weight * saccade_var
    metrics = {
        "total_loss": total.detach(),
        "stability_loss": temporal.detach(),
        "sigreg_loss": sig.detach(),
        "saccade_variance": saccade_var.detach(),
    }
    return total, metrics


def create_run_dir(log_root: str | Path) -> Path:
    log_root = Path(log_root)
    log_root.mkdir(parents=True, exist_ok=True)
    existing_run_ids = [int(child.name) for child in log_root.iterdir() if child.is_dir() and child.name.isdigit()]
    run_id = 1 if not existing_run_ids else max(existing_run_ids) + 1
    run_dir = log_root / str(run_id)
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def save_run_metadata(run_dir: Path, args: argparse.Namespace | None = None) -> None:
    if args is None:
        return
    metadata = vars(args).copy()
    metadata["run_dir"] = str(run_dir)
    metadata["created_at"] = datetime.now().isoformat(timespec="seconds")
    with open(run_dir / "args.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)


def _positions_to_pixel_coordinates(saccades: torch.Tensor, image_size: int = 28) -> torch.Tensor:
    return saccades.mul(image_size - 1).round().to(dtype=torch.int64)


def make_retina_grid(
    originals: torch.Tensor,
    retina_sequence: torch.Tensor,
    saccades: torch.Tensor,
    num_examples: int = 4,
) -> torch.Tensor:
    num_examples = min(num_examples, originals.shape[0])
    sequence_len = retina_sequence.shape[1]
    originals = originals[:num_examples].float().cpu()
    retina_sequence = retina_sequence[:num_examples].float().cpu()
    points = _positions_to_pixel_coordinates(saccades[:num_examples].cpu())

    rows = []
    for sample_idx in range(num_examples):
        base = originals[sample_idx].clone()
        for point_idx in range(sequence_len):
            x_coord = int(points[sample_idx, point_idx, 0].item())
            y_coord = int(points[sample_idx, point_idx, 1].item())
            base[0, max(0, y_coord - 1) : min(28, y_coord + 2), x_coord] = 1.0
            base[0, y_coord, max(0, x_coord - 1) : min(28, x_coord + 2)] = 1.0
        rows.append(base)
        rows.extend(retina_sequence[sample_idx, time_idx] for time_idx in range(sequence_len))

    panel = torch.stack(rows, dim=0)
    return make_grid(panel, nrow=sequence_len + 1, normalize=True, value_range=(-1.0, 1.0), pad_value=0.0)


@torch.no_grad()
def evaluate(
    model: MNISTSaccadeRNN,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    sigreg: SigReg,
    writer: SummaryWriter | None,
    epoch: int,
    num_steps: int,
    sigreg_weight: float,
    saccade_var_weight: float,
    num_examples: int = 4,
    max_batches: int | None = None,
) -> dict[str, float]:
    model.eval()
    metric_sums = {"total_loss": 0.0, "stability_loss": 0.0, "sigreg_loss": 0.0, "saccade_variance": 0.0}
    total_samples = 0
    retina_grid = None

    for batch_idx, (imgs, _) in enumerate(dataloader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        imgs = imgs.to(device)
        outputs = model(imgs, num_steps=num_steps, return_retina_sequence=True)
        loss, metrics = compute_ssl_losses(outputs, sigreg, sigreg_weight=sigreg_weight, saccade_var_weight=saccade_var_weight)
        del loss

        batch_size = imgs.shape[0]
        total_samples += batch_size
        for name in metric_sums:
            metric_sums[name] += float(metrics[name].item()) * batch_size

        if retina_grid is None:
            retina_grid = make_retina_grid(
                imgs,
                outputs["retina_sequence"],
                outputs["saccades"],
                num_examples=num_examples,
            )

    averaged = {name: value / max(total_samples, 1) for name, value in metric_sums.items()}
    if writer is not None:
        step = epoch + 1
        for name, value in averaged.items():
            writer.add_scalar(f"val/{name}", value, step)
        if retina_grid is not None:
            writer.add_image("val/retina_sequence", retina_grid, step)
    return averaged


def train(
    model: MNISTSaccadeRNN,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    device: torch.device,
    n_epochs: int,
    num_steps: int,
    lr: float,
    sigreg_weight: float,
    saccade_var_weight: float,
    log_dir: str | Path,
    num_examples: int = 4,
    max_train_batches: int | None = None,
    max_val_batches: int | None = None,
    args: argparse.Namespace | None = None,
) -> Path:
    model = model.to(device)
    sigreg = SigReg().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    run_dir = create_run_dir(log_dir)
    writer = SummaryWriter(log_dir=str(run_dir))
    save_run_metadata(run_dir, args=args)

    print(f"Logging to: {run_dir}")
    for epoch in range(n_epochs):
        model.train()
        metric_sums = {"total_loss": 0.0, "stability_loss": 0.0, "sigreg_loss": 0.0, "saccade_variance": 0.0}
        total_samples = 0

        for batch_idx, (imgs, _) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1}/{n_epochs}")):
            if max_train_batches is not None and batch_idx >= max_train_batches:
                break

            imgs = imgs.to(device)
            outputs = model(imgs, num_steps=num_steps, return_retina_sequence=False)
            loss, metrics = compute_ssl_losses(
                outputs,
                sigreg,
                sigreg_weight=sigreg_weight,
                saccade_var_weight=saccade_var_weight,
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            batch_size = imgs.shape[0]
            total_samples += batch_size
            for name in metric_sums:
                metric_sums[name] += float(metrics[name].item()) * batch_size

        train_metrics = {name: value / max(total_samples, 1) for name, value in metric_sums.items()}
        step = epoch + 1
        for name, value in train_metrics.items():
            writer.add_scalar(f"train/{name}", value, step)

        val_metrics = evaluate(
            model=model,
            dataloader=val_loader,
            device=device,
            sigreg=sigreg,
            writer=writer,
            epoch=epoch,
            num_steps=num_steps,
            sigreg_weight=sigreg_weight,
            saccade_var_weight=saccade_var_weight,
            num_examples=num_examples,
            max_batches=max_val_batches,
        )
        print(
            f"Epoch {step}: "
            f"train_total={train_metrics['total_loss']:.4f}, "
            f"train_stability={train_metrics['stability_loss']:.4f}, "
            f"train_sigreg={train_metrics['sigreg_loss']:.4f}, "
            f"train_sacc_var={train_metrics['saccade_variance']:.4f}, "
            f"val_total={val_metrics['total_loss']:.4f}, "
            f"val_stability={val_metrics['stability_loss']:.4f}, "
            f"val_sigreg={val_metrics['sigreg_loss']:.4f}, "
            f"val_sacc_var={val_metrics['saccade_variance']:.4f}"
        )

    writer.close()
    return run_dir


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data_dir", type=str, default=str(DATADIR))
    parser.add_argument("--download", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--num_steps", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--v1_features", type=int, default=32)
    parser.add_argument("--rep_features", type=int, default=16)
    parser.add_argument("--saccade_features", type=int, default=8)
    parser.add_argument("--sigreg_weight", type=float, default=0.1)
    parser.add_argument("--saccade_var_weight", type=float, default=0.1)
    parser.add_argument("--fovea_scale", type=float, default=0.35)
    parser.add_argument("--peripheral_scale", type=float, default=0.9)
    parser.add_argument("--blur_sigma", type=float, default=1.5)
    parser.add_argument("--fovea_sharpness", type=float, default=12.0)
    parser.add_argument("--num_examples", type=int, default=4)
    parser.add_argument("--max_train_batches", type=int, default=None)
    parser.add_argument("--max_val_batches", type=int, default=None)
    parser.add_argument(
        "--log_dir",
        type=str,
        default=str(Path(rCNN_logs).resolve() / "saccade_ssl" / "mnist"),
    )
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, _ = mnist(
        root=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        download=args.download,
    )
    model = MNISTSaccadeRNN(
        v1_features=args.v1_features,
        rep_features=args.rep_features,
        saccade_features=args.saccade_features,
        fovea_scale=args.fovea_scale,
        peripheral_scale=args.peripheral_scale,
        blur_sigma=args.blur_sigma,
        fovea_sharpness=args.fovea_sharpness,
    )
    train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        n_epochs=args.epochs,
        num_steps=args.num_steps,
        lr=args.lr,
        sigreg_weight=args.sigreg_weight,
        saccade_var_weight=args.saccade_var_weight,
        log_dir=args.log_dir,
        num_examples=args.num_examples,
        max_train_batches=args.max_train_batches,
        max_val_batches=args.max_val_batches,
        args=args,
    )


if __name__ == "__main__":
    main()
