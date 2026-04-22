from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import importlib.util
import json
import os
from pathlib import Path
from typing import Any

import torch
from torchvision.utils import make_grid
from tqdm import tqdm

from coala import DATADIR, RNN_logs, hcRNN_logs, lrRNN_logs
from coala.datasets import get_dataloaders
from coala.rnn.hcrnn import hConvRNN, compute_losses as compute_hcrnn_losses
from coala.rnn.lr_rnn import lrRNN, compute_losses as compute_lrrnn_losses
from coala.rnn.rnn import RNN, compute_losses as compute_rnn_losses
from coala.visualize_activation_maps import create_activation_input_figure, create_activation_map_figure

__test__ = False

FAMILY_CHOICES = ("hcrnn", "rnn", "lrrnn")
DEFAULT_DATASET = "msmnist"
DEFAULT_CHECKPOINT_NAME = "best_val_loss.pt"


def _resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_name)


def _normalize_family_name(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.lower().replace("_", "")
    alias_map = {
        "hcrnn": "hcrnn",
        "hconvrnn": "hcrnn",
        "rnn": "rnn",
        "lrrnn": "lrrnn",
        "lowrankrnn": "lrrnn",
    }
    if normalized not in alias_map:
        raise ValueError(f"Unknown RNN family {value!r}. Expected one of {FAMILY_CHOICES}.")
    return alias_map[normalized]


def _family_log_root(family: str) -> Path:
    roots = {
        "hcrnn": Path(hcRNN_logs),
        "rnn": Path(RNN_logs),
        "lrrnn": Path(lrRNN_logs),
    }
    return roots[family]


def _latest_checkpoint(family: str, dataset_name: str = DEFAULT_DATASET) -> Path:
    dataset_root = _family_log_root(family) / dataset_name
    if not dataset_root.exists():
        raise FileNotFoundError(f"Missing log directory for {family}: {dataset_root}")

    candidate_runs = []
    for child in dataset_root.iterdir():
        checkpoint_path = child / DEFAULT_CHECKPOINT_NAME
        if not child.is_dir() or not checkpoint_path.exists():
            continue
        try:
            sort_key = int(child.name)
        except ValueError:
            sort_key = -1
        candidate_runs.append((sort_key, checkpoint_path))

    if not candidate_runs:
        raise FileNotFoundError(f"No {DEFAULT_CHECKPOINT_NAME} files found under {dataset_root}")
    candidate_runs.sort(key=lambda item: item[0])
    return candidate_runs[-1][1]


def _infer_family_from_checkpoint_path(checkpoint_path: Path) -> str:
    parts = [part.lower() for part in checkpoint_path.parts]
    if "hcrnn" in parts:
        return "hcrnn"
    if "lrrnn" in parts:
        return "lrrnn"
    if "rnn" in parts:
        return "rnn"
    raise ValueError(
        "Could not infer checkpoint family from path. Pass --family explicitly. "
        f"checkpoint_path={checkpoint_path}"
    )


def _load_run_args(checkpoint_path: Path) -> dict[str, Any]:
    args_path = checkpoint_path.with_name("args.json")
    if not args_path.exists():
        return {}
    with open(args_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _parse_masked_fill(value: str | float | int) -> str | float:
    if isinstance(value, str) and value == "random":
        return value
    return float(value)


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "y", "on"}:
            return True
        if lowered in {"0", "false", "no", "n", "off"}:
            return False
    return bool(value)


def _config_value(cli_value: Any, run_args: dict[str, Any], key: str, default: Any) -> Any:
    if cli_value is not None:
        return cli_value
    if key in run_args:
        return run_args[key]
    return default


def _build_model(family: str, run_args: dict[str, Any]) -> torch.nn.Module:
    if family == "hcrnn":
        return hConvRNN(hopfield=_coerce_bool(run_args.get("hopfield", False)))
    if family == "rnn":
        return RNN(conv_inout=_coerce_bool(run_args.get("conv_inout", False)))
    if family == "lrrnn":
        return lrRNN(rank=int(run_args.get("rank", 4)))
    raise ValueError(f"Unsupported family {family!r}")


def _load_model_checkpoint(
    model: torch.nn.Module,
    checkpoint_payload: dict[str, Any],
) -> tuple[list[str], list[str]]:
    state_dict = checkpoint_payload["model_state_dict"]
    model_state = model.state_dict()
    compatible_state: dict[str, torch.Tensor] = {}
    skipped_keys: list[str] = []

    for key, value in state_dict.items():
        current_value = model_state.get(key)
        if current_value is None or current_value.shape != value.shape:
            skipped_keys.append(key)
            continue
        compatible_state[key] = value

    missing_keys, unexpected_keys = model.load_state_dict(compatible_state, strict=False)
    skipped_keys.extend(unexpected_keys)
    return sorted(set(missing_keys)), sorted(set(skipped_keys))


def _show_image_grid(grid: torch.Tensor, title: str):
    import matplotlib.pyplot as plt

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
    return fig


def _stack_temporal_outputs(
    outputs: list[torch.Tensor] | torch.Tensor,
    expected_ndim: int,
) -> torch.Tensor:
    stacked = outputs if isinstance(outputs, torch.Tensor) else torch.stack(outputs, dim=1)
    if stacked.dim() != expected_ndim:
        raise ValueError(f"Expected tensor with {expected_ndim} dims, got shape {tuple(stacked.shape)}.")
    return stacked


def _stack_temporal_logits(logits: list[torch.Tensor] | torch.Tensor) -> torch.Tensor:
    return _stack_temporal_outputs(logits, expected_ndim=3)


def _stack_temporal_reconstructions(reconstructions: list[torch.Tensor] | torch.Tensor) -> torch.Tensor:
    return _stack_temporal_outputs(reconstructions, expected_ndim=5)


def _to_display_range(images: torch.Tensor) -> torch.Tensor:
    return ((images + 1.0) * 0.5).clamp_(0.0, 1.0)


def _compute_temporal_reconstruction_loss(
    reconstructions: list[torch.Tensor] | torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    reconstructions = _stack_temporal_reconstructions(reconstructions)
    if targets.dim() == 4:
        per_pixel_mse = (reconstructions - targets.unsqueeze(1)).pow(2).mean(dim=2)
    elif targets.dim() == 5:
        per_pixel_mse = (reconstructions - targets).pow(2).mean(dim=2)
    else:
        raise ValueError(
            "Expected targets with shape (B, C, H, W) or (B, T, C, H, W), "
            f"got {tuple(targets.shape)}."
        )
    return per_pixel_mse.mean(dim=(-2, -1))


def create_temporal_prediction_figure(
    logits: list[torch.Tensor] | torch.Tensor,
    labels: torch.Tensor,
    max_examples: int | None = None,
):
    import matplotlib.pyplot as plt

    logits = _stack_temporal_logits(logits).detach().cpu()
    labels = labels.detach().cpu()
    if max_examples is not None:
        logits = logits[:max_examples]
        labels = labels[:max_examples]

    probs = torch.softmax(logits, dim=-1)
    preds = probs.argmax(dim=-1)
    if labels.dim() == 1:
        label_targets = labels.unsqueeze(1).expand(-1, logits.shape[1])
    elif labels.dim() == 2 and labels.shape[1] == logits.shape[1]:
        label_targets = labels
    else:
        raise ValueError(f"Expected labels with shape (B,) or (B, T), got {tuple(labels.shape)}.")

    timesteps = torch.arange(logits.shape[1]).cpu().numpy()
    tick_step = max(1, logits.shape[1] // 12)
    fig_width = min(18.0, max(10.0, 0.35 * logits.shape[1]))
    fig, axes = plt.subplots(label_targets.shape[0], 1, figsize=(fig_width, 2.2 * label_targets.shape[0]), sharex=True)
    if label_targets.shape[0] == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        true_labels = label_targets[i]
        true_prob = probs[i].gather(1, true_labels.unsqueeze(1)).squeeze(1).numpy()
        pred_prob = probs[i].max(dim=-1).values.numpy()
        start_label = int(true_labels[0].item())
        end_label = int(true_labels[-1].item())
        label_legend = f"true={start_label}" if start_label == end_label else f"true={start_label}->{end_label}"
        label_text = f"label={start_label}" if start_label == end_label else f"label={start_label}->{end_label}"
        ax.plot(timesteps, true_prob, marker="o", label=f"P({label_legend})")
        ax.plot(timesteps, pred_prob, marker="x", linestyle="--", label="P(pred)")
        ax.set_xlim(0, logits.shape[1] - 1)
        ax.set_ylabel(f"sample {i}")
        ax.set_title(f"{label_text} | final_pred={int(preds[i, -1].item())}")
        ax.grid(alpha=0.25)
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

    losses = _compute_temporal_reconstruction_loss(reconstructions, targets).detach().cpu()
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
        ax.set_title(f"final_loss={series[-1]:.4f}")
        ax.grid(alpha=0.25)
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
    reconstructions = _to_display_range(_stack_temporal_reconstructions(reconstructions).detach().cpu())
    masked_images = _to_display_range(masked_images[:num_examples].detach().cpu())
    reconstructions = reconstructions[:num_examples]
    target_images = target_images[:num_examples].detach().cpu()
    if target_images.dim() == 4:
        target_images = _to_display_range(target_images).unsqueeze(1).expand(-1, reconstructions.shape[1], -1, -1, -1)
    elif target_images.dim() == 5:
        target_images = _to_display_range(target_images)
    else:
        raise ValueError(
            "target_images must have shape (B, C, H, W) or (B, T, C, H, W), "
            f"got {tuple(target_images.shape)}."
        )

    _, total_t, _, _, _ = masked_images.shape
    num_time_steps = min(total_t, max_time_steps)
    time_idx = torch.linspace(0, total_t - 1, steps=num_time_steps).round().long()
    masked_panel = masked_images[:, time_idx]
    recon_panel = reconstructions[:, time_idx]
    target_panel = target_images[:, time_idx]
    panel = torch.stack((masked_panel, recon_panel, target_panel), dim=1).reshape(-1, *masked_images.shape[2:])
    return make_grid(panel, nrow=num_time_steps, normalize=False, pad_value=0.5)


def _create_step_accuracy_figure(step_accuracy_percent: torch.Tensor):
    import matplotlib.pyplot as plt

    values = step_accuracy_percent.detach().cpu()
    timesteps = torch.arange(values.shape[0]).cpu().numpy()
    fig, ax = plt.subplots(figsize=(min(18.0, max(8.0, 0.4 * values.shape[0])), 3.8))
    ax.plot(timesteps, values.numpy(), marker="o")
    ax.set_xlabel("time step")
    ax.set_ylabel("accuracy (%)")
    ax.set_title(f"Step Accuracy | final={values[-1].item():.2f}%")
    ax.set_xlim(0, values.shape[0] - 1)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    return fig


def _load_scalar_history(run_dir: Path) -> dict[str, list[tuple[float, float]]]:
    try:
        from tensorboard.backend.event_processing import event_accumulator
    except ImportError:
        return {}

    event_files = sorted(run_dir.glob("events.out.tfevents.*"))
    if not event_files:
        return {}

    accumulator = event_accumulator.EventAccumulator(str(event_files[-1]), size_guidance={"scalars": 0})
    accumulator.Reload()
    histories: dict[str, list[tuple[float, float]]] = {}
    for tag in accumulator.Tags().get("scalars", []):
        histories[tag] = [(event.step, event.value) for event in accumulator.Scalars(tag)]
    return histories


def _create_training_curves_figure(run_dir: Path):
    import matplotlib.pyplot as plt

    histories = _load_scalar_history(run_dir)
    if not histories:
        return None

    plot_groups = [
        ("loss", ("train_loss", "val_loss")),
        ("reconstruction", ("train_recon_loss", "val_recon_loss")),
        ("classification", ("train_class_loss", "val_class_loss")),
        ("accuracy", ("val_accuracy_percent",)),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), squeeze=False)
    axes_flat = axes.flatten()

    for ax, (title, tags) in zip(axes_flat, plot_groups):
        plotted = False
        for tag in tags:
            series = histories.get(tag)
            if not series:
                continue
            steps, values = zip(*series)
            ax.plot(steps, values, marker="o", label=tag)
            plotted = True
        if plotted:
            ax.set_title(title)
            ax.grid(alpha=0.25)
            ax.legend()
        else:
            ax.axis("off")

    fig.suptitle(f"Training Curves | {run_dir}")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    return fig


def _compute_losses(
    family: str,
    recon: torch.Tensor,
    class_logits: torch.Tensor,
    clean_image: torch.Tensor,
    labels: torch.Tensor,
    run_args: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if family == "hcrnn":
        return compute_hcrnn_losses(
            recon,
            class_logits,
            clean_image,
            labels,
            t0_weight=float(run_args.get("t0_weight", 0.5)),
        )
    if family == "rnn":
        return compute_rnn_losses(recon, class_logits, clean_image, labels)
    if family == "lrrnn":
        return compute_lrrnn_losses(recon, class_logits, clean_image, labels)
    raise ValueError(f"Unsupported family {family!r}")


def _extract_forward_outputs(
    model_output: tuple[torch.Tensor, torch.Tensor] | dict[str, Any],
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    dict[str, dict[str, torch.Tensor]] | None,
    dict[str, torch.Tensor] | None,
]:
    if isinstance(model_output, dict):
        return (
            model_output["recon"],
            model_output["class_logits"],
            model_output.get("activation_maps"),
            model_output.get("layer_trajectories"),
        )
    recon, class_logits = model_output
    return recon, class_logits, None, None


def _expand_label_targets(labels: torch.Tensor, n_steps: int) -> torch.Tensor:
    if labels.dim() == 1:
        return labels.unsqueeze(1).expand(-1, n_steps)
    if labels.dim() == 2 and labels.shape[1] == n_steps:
        return labels
    raise ValueError(f"Expected labels with shape (B,) or (B, T), got {tuple(labels.shape)}.")


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    family: str,
    device: torch.device,
    run_args: dict[str, Any],
    max_batches: int | None = None,
    collect_activation_maps: bool = True,
    classification: bool = False,
    show_progress: bool = False,
) -> dict[str, Any]:
    model.eval()
    total_loss = 0.0
    total_recon_loss = 0.0
    total_class_loss = 0.0 if classification else None
    total_samples = 0
    correct_by_step = None if classification else None
    visual_payload = None

    iterator = dataloader
    if show_progress:
        iterator = tqdm(dataloader, desc=f"Evaluating {family}", leave=False)
    for batch_idx, batch in enumerate(iterator):
        if max_batches is not None and batch_idx >= max_batches:
            break

        masked_inputs, targets = batch
        masked_inputs = masked_inputs.to(device)
        clean_image = targets["image"].to(device)
        labels = targets["label"].to(device)
        need_maps = collect_activation_maps and visual_payload is None
        model_output = model(masked_inputs, return_activation_maps=need_maps)
        recon, class_logits, activation_maps, _ = _extract_forward_outputs(model_output)
        recon_loss, class_loss, loss = _compute_losses(
            family=family,
            recon=recon,
            class_logits=class_logits,
            clean_image=clean_image,
            labels=labels,
            run_args=run_args,
        )

        batch_size = masked_inputs.shape[0]
        tracked_loss = loss if classification else recon_loss
        total_loss += tracked_loss.item() * batch_size
        total_recon_loss += recon_loss.item() * batch_size
        if total_class_loss is not None:
            total_class_loss += class_loss.item() * batch_size
        total_samples += batch_size

        if classification:
            preds = class_logits.argmax(dim=-1)
            target_labels = _expand_label_targets(labels, class_logits.shape[1])
            matches = preds.eq(target_labels)
            if correct_by_step is None:
                correct_by_step = torch.zeros(class_logits.shape[1], dtype=torch.float32)
            correct_by_step += matches.sum(dim=0).to(dtype=torch.float32).cpu()

        if visual_payload is None:
            visual_payload = {
                "masked_inputs": masked_inputs.detach().cpu(),
                "clean_image": clean_image.detach().cpu(),
                "labels": labels.detach().cpu(),
                "recon": recon.detach().cpu(),
                "class_logits": class_logits.detach().cpu(),
                "activation_maps": None
                if activation_maps is None
                else {
                    signal_name: {
                        layer_name: layer_tensor.detach().cpu()
                        for layer_name, layer_tensor in per_signal.items()
                    }
                    for signal_name, per_signal in activation_maps.items()
                },
            }

    if total_samples == 0:
        raise RuntimeError("No evaluation batches were processed.")

    step_accuracy_percent = None if correct_by_step is None else 100.0 * correct_by_step / total_samples
    return {
        "mean_loss": total_loss / total_samples,
        "mean_recon_loss": total_recon_loss / total_samples,
        "mean_class_loss": None if total_class_loss is None else total_class_loss / total_samples,
        "step_accuracy_percent": step_accuracy_percent,
        "final_accuracy_percent": None if step_accuracy_percent is None else step_accuracy_percent[-1].item(),
        "num_samples": total_samples,
        "visual_payload": visual_payload,
    }


def _trajectory_digit(label_sequence: torch.Tensor) -> int:
    if label_sequence.dim() == 0:
        return int(label_sequence.item())
    if label_sequence.dim() == 1:
        return int(label_sequence[0].item())
    raise ValueError(f"Unsupported label sequence shape {tuple(label_sequence.shape)}")


def collect_layer_trajectories(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    n_traj_per_digit: int,
    accepted_digits: list[int] | None = None,
    max_batches: int | None = None,
    show_progress: bool = False,
) -> tuple[dict[str, list[tuple[int, torch.Tensor]]], dict[int, int]]:
    if n_traj_per_digit <= 0:
        return {}, {}

    target_digits = list(range(10)) if accepted_digits is None else sorted({int(d) for d in accepted_digits})
    per_digit_counts = {digit: 0 for digit in target_digits}
    trajectories_by_layer: dict[str, list[tuple[int, torch.Tensor]]] = defaultdict(list)

    model.eval()
    iterator = dataloader
    if show_progress:
        iterator = tqdm(dataloader, desc="Collecting corrupted trajectories", leave=False)
    for batch_idx, batch in enumerate(iterator):
        if max_batches is not None and batch_idx >= max_batches:
            break

        masked_inputs, targets = batch
        labels = targets["label"]
        selected_indices: list[int] = []
        selected_digits: list[int] = []
        selected_in_batch = {digit: 0 for digit in target_digits}
        for sample_idx in range(labels.shape[0]):
            digit = _trajectory_digit(labels[sample_idx])
            if digit not in per_digit_counts:
                continue
            if per_digit_counts[digit] + selected_in_batch[digit] >= n_traj_per_digit:
                continue
            selected_indices.append(sample_idx)
            selected_digits.append(digit)
            selected_in_batch[digit] += 1

        if not selected_indices:
            if all(count >= n_traj_per_digit for count in per_digit_counts.values()):
                break
            continue

        selected_batch = masked_inputs[selected_indices].to(device)
        model_output = model(selected_batch, return_layer_trajectories=True)
        _, _, _, layer_trajectories = _extract_forward_outputs(model_output)
        if layer_trajectories is None:
            raise RuntimeError("Model did not return layer trajectories.")

        for row_idx, digit in enumerate(selected_digits):
            for layer_name, layer_tensor in layer_trajectories.items():
                trajectories_by_layer[layer_name].append((digit, layer_tensor[row_idx].detach().cpu()))
            per_digit_counts[digit] += 1

        if all(count >= n_traj_per_digit for count in per_digit_counts.values()):
            break

    return dict(sorted(trajectories_by_layer.items())), per_digit_counts


def collect_layer_fixed_points(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    show_progress: bool = False,
) -> dict[str, torch.Tensor]:
    fixed_points_by_layer: dict[str, list[torch.Tensor]] = defaultdict(list)

    model.eval()
    iterator = dataloader
    if show_progress:
        iterator = tqdm(dataloader, desc="Collecting reference fixed points", leave=False)
    for batch in iterator:
        masked_inputs, _ = batch
        batch_inputs = masked_inputs.to(device)
        model_output = model(batch_inputs, return_layer_trajectories=True)
        _, _, _, layer_trajectories = _extract_forward_outputs(model_output)
        if layer_trajectories is None:
            raise RuntimeError("Model did not return layer trajectories.")

        for layer_name, layer_tensor in layer_trajectories.items():
            fixed_points_by_layer[layer_name].append(layer_tensor[:, -1].detach().cpu())

    return {
        layer_name: torch.cat(layer_values, dim=0)
        for layer_name, layer_values in sorted(fixed_points_by_layer.items())
        if layer_values
    }


def _project_to_principal_components(points: torch.Tensor, n_components: int) -> torch.Tensor:
    points = points.float()
    if points.ndim != 2:
        raise ValueError(f"Expected 2D points tensor, got {tuple(points.shape)}.")
    if n_components <= 0:
        raise ValueError(f"n_components must be positive, got {n_components}.")

    n_samples, n_features = points.shape
    target_dims = min(n_components, n_samples, n_features)
    if target_dims == 0:
        return torch.zeros((n_samples, n_components), dtype=torch.float32)

    if target_dims == 1:
        projected = points[:, :1] - points[:, :1].mean(dim=0, keepdim=True)
    else:
        centered = points - points.mean(dim=0, keepdim=True)
        q = min(target_dims, centered.shape[0], centered.shape[1])
        u, s, _ = torch.pca_lowrank(centered, q=q, center=False)
        projected = u[:, :target_dims] * s[:target_dims]

    if projected.shape[1] < n_components:
        padding = torch.zeros((n_samples, n_components - projected.shape[1]), dtype=projected.dtype)
        projected = torch.cat((projected, padding), dim=1)
    return projected


def _fit_single_pca_reducer(job: tuple[str, int, torch.Tensor]) -> tuple[str, dict[str, torch.Tensor]]:
    layer_name, n_components, points = job
    points = points.float()
    mean = points.mean(dim=0, keepdim=True)
    centered = points - mean
    n_samples, n_features = centered.shape
    target_dims = min(n_components, n_samples, n_features)
    if target_dims <= 0:
        components = torch.zeros((0, points.shape[1]), dtype=torch.float32)
    elif target_dims == 1:
        components = torch.zeros((1, points.shape[1]), dtype=torch.float32)
        components[0, 0] = 1.0
    else:
        q = min(target_dims, centered.shape[0], centered.shape[1])
        _, _, v = torch.pca_lowrank(centered, q=q, center=False)
        components = v[:, :target_dims].T.contiguous()
    return layer_name, {"mean": mean, "components": components}


def _fit_pca_reducers(
    reference_points_by_layer: dict[str, torch.Tensor],
    n_components: int,
    show_progress: bool = False,
) -> dict[str, dict[str, torch.Tensor]]:
    reducers: dict[str, dict[str, torch.Tensor]] = {}
    jobs = [(layer_name, n_components, points) for layer_name, points in reference_points_by_layer.items()]
    iterator = jobs
    if show_progress:
        iterator = tqdm(jobs, desc=f"Fitting PCA {n_components}D reducers", leave=False)
    for job in iterator:
        layer_name, reducer = _fit_single_pca_reducer(job)
        reducers[layer_name] = reducer
    return reducers


def _transform_with_fitted_pca(
    reducer: dict[str, torch.Tensor],
    points: torch.Tensor,
    n_components: int,
) -> torch.Tensor:
    centered = points.float() - reducer["mean"].float()
    components = reducer["components"].float()
    projected = centered @ components.T if components.numel() > 0 else torch.zeros((points.shape[0], 0), dtype=torch.float32)
    if projected.shape[1] < n_components:
        padding = torch.zeros((projected.shape[0], n_components - projected.shape[1]), dtype=projected.dtype)
        projected = torch.cat((projected, padding), dim=1)
    return projected


def _umap_available() -> bool:
    return importlib.util.find_spec("umap") is not None


def _project_with_umap(points: torch.Tensor, n_components: int) -> torch.Tensor:
    if not _umap_available():
        raise ImportError("UMAP plots require the optional 'umap-learn' package.")

    import umap

    points = points.float()
    if points.ndim != 2:
        raise ValueError(f"Expected 2D points tensor, got {tuple(points.shape)}.")
    if n_components <= 0:
        raise ValueError(f"n_components must be positive, got {n_components}.")

    reducer = umap.UMAP(n_components=n_components, random_state=42)
    embedding = reducer.fit_transform(points.numpy())
    projected = torch.from_numpy(embedding).to(dtype=torch.float32)
    if projected.shape[1] < n_components:
        padding = torch.zeros((projected.shape[0], n_components - projected.shape[1]), dtype=projected.dtype)
        projected = torch.cat((projected, padding), dim=1)
    return projected


def _fit_umap_reducers(
    reference_points_by_layer: dict[str, torch.Tensor],
    n_components: int,
) -> dict[str, Any]:
    if not _umap_available():
        return {}

    import umap

    reducers: dict[str, Any] = {}
    for layer_name, points in reference_points_by_layer.items():
        reducer = umap.UMAP(n_components=n_components, random_state=42)
        reducer.fit(points.float().numpy())
        reducers[layer_name] = reducer
    return reducers


def _fit_single_umap_reducer(job: tuple[str, int, Any]) -> tuple[str, Any]:
    import warnings

    import umap

    layer_name, n_components, points = job
    warnings.filterwarnings(
        "ignore",
        message=r"n_jobs value 1 overridden to 1 by setting random_state.*",
        category=UserWarning,
    )
    reducer = umap.UMAP(n_components=n_components, random_state=42)
    reducer.fit(points)
    return layer_name, reducer


def _fit_umap_reducers_parallel(
    reference_points_by_layer: dict[str, torch.Tensor],
    n_components: int,
    show_progress: bool = False,
) -> dict[str, Any]:
    if not _umap_available() or not reference_points_by_layer:
        return {}

    jobs = [
        (layer_name, n_components, points.float().numpy())
        for layer_name, points in reference_points_by_layer.items()
    ]
    max_workers = min(len(jobs), max(1, os.cpu_count() or 1))
    if max_workers <= 1:
        return _fit_umap_reducers(reference_points_by_layer, n_components=n_components)

    reducers: dict[str, Any] = {}
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        mapped = executor.map(_fit_single_umap_reducer, jobs)
        if show_progress:
            mapped = tqdm(
                mapped,
                total=len(jobs),
                desc=f"Fitting UMAP {n_components}D reducers",
                leave=False,
            )
        for layer_name, reducer in mapped:
            reducers[layer_name] = reducer
    return reducers


def _transform_with_fitted_umap(reducer: Any, points: torch.Tensor, n_components: int) -> torch.Tensor:
    projected = torch.from_numpy(reducer.transform(points.float().numpy())).to(dtype=torch.float32)
    if projected.shape[1] < n_components:
        padding = torch.zeros((projected.shape[0], n_components - projected.shape[1]), dtype=projected.dtype)
        projected = torch.cat((projected, padding), dim=1)
    return projected


def create_layer_trajectory_pca_2d_figure(
    layer_name: str,
    trajectories: list[tuple[int, torch.Tensor]],
    reducer: dict[str, torch.Tensor] | None = None,
):
    import matplotlib.pyplot as plt

    if not trajectories or reducer is None:
        return None

    points = torch.cat([trajectory for _, trajectory in trajectories], dim=0)
    projected = _transform_with_fitted_pca(reducer, points, n_components=2)

    cmap = plt.get_cmap("tab10")
    fig, ax = plt.subplots(figsize=(7, 6))

    cursor = 0
    legend_done: set[int] = set()
    for digit, trajectory in trajectories:
        n_steps = trajectory.shape[0]
        coords = projected[cursor: cursor + n_steps]
        cursor += n_steps
        color = cmap(digit % 10)
        label = f"digit {digit}" if digit not in legend_done else None
        legend_done.add(digit)
        ax.plot(coords[:, 0], coords[:, 1], color=color, alpha=0.85, linewidth=1.8, label=label)
        ax.scatter(coords[0, 0], coords[0, 1], color=color, marker="o", s=28)
        ax.scatter(coords[-1, 0], coords[-1, 1], color=color, marker="x", s=36)

    ax.set_title(f"{layer_name} Activation Trajectories (2D PCA)")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.grid(alpha=0.25)
    if legend_done:
        ax.legend(loc="best", ncols=2)
    fig.tight_layout()
    return fig


def create_layer_trajectory_umap_2d_figure(
    layer_name: str,
    trajectories: list[tuple[int, torch.Tensor]],
    reducer: Any | None = None,
):
    import matplotlib.pyplot as plt

    if not trajectories or reducer is None:
        return None

    points = torch.cat([trajectory for _, trajectory in trajectories], dim=0)
    projected = _transform_with_fitted_umap(reducer, points, n_components=2)

    cmap = plt.get_cmap("tab10")
    fig, ax = plt.subplots(figsize=(7, 6))

    cursor = 0
    legend_done: set[int] = set()
    for digit, trajectory in trajectories:
        n_steps = trajectory.shape[0]
        coords = projected[cursor: cursor + n_steps]
        cursor += n_steps
        color = cmap(digit % 10)
        label = f"digit {digit}" if digit not in legend_done else None
        legend_done.add(digit)
        ax.plot(coords[:, 0], coords[:, 1], color=color, alpha=0.85, linewidth=1.8, label=label)
        ax.scatter(coords[0, 0], coords[0, 1], color=color, marker="o", s=28)
        ax.scatter(coords[-1, 0], coords[-1, 1], color=color, marker="x", s=36)

    ax.set_title(f"{layer_name} Activation Trajectories (2D UMAP)")
    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")
    ax.grid(alpha=0.25)
    if legend_done:
        ax.legend(loc="best", ncols=2)
    fig.tight_layout()
    return fig


def create_layer_trajectory_pca_3d_figure(
    layer_name: str,
    trajectories: list[tuple[int, torch.Tensor]],
    reducer: dict[str, torch.Tensor] | None = None,
):
    import matplotlib.pyplot as plt

    if not trajectories or reducer is None:
        return None

    points = torch.cat([trajectory for _, trajectory in trajectories], dim=0)
    projected = _transform_with_fitted_pca(reducer, points, n_components=3)

    cmap = plt.get_cmap("tab10")
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    cursor = 0
    legend_done: set[int] = set()
    for digit, trajectory in trajectories:
        n_steps = trajectory.shape[0]
        coords = projected[cursor: cursor + n_steps]
        cursor += n_steps
        color = cmap(digit % 10)
        label = f"digit {digit}" if digit not in legend_done else None
        legend_done.add(digit)
        ax.plot(coords[:, 0], coords[:, 1], coords[:, 2], color=color, alpha=0.8, linewidth=1.8, label=label)
        ax.scatter(coords[0, 0], coords[0, 1], coords[0, 2], color=color, marker="o", s=28)
        ax.scatter(coords[-1, 0], coords[-1, 1], coords[-1, 2], color=color, marker="x", s=36)

    ax.set_title(f"{layer_name} Activation Trajectories (3D PCA)")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    if legend_done:
        ax.legend(loc="upper right", ncols=2)
    fig.tight_layout()
    return fig


def create_layer_trajectory_umap_3d_figure(
    layer_name: str,
    trajectories: list[tuple[int, torch.Tensor]],
    reducer: Any | None = None,
):
    import matplotlib.pyplot as plt

    if not trajectories or reducer is None:
        return None

    points = torch.cat([trajectory for _, trajectory in trajectories], dim=0)
    projected = _transform_with_fitted_umap(reducer, points, n_components=3)

    cmap = plt.get_cmap("tab10")
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    cursor = 0
    legend_done: set[int] = set()
    for digit, trajectory in trajectories:
        n_steps = trajectory.shape[0]
        coords = projected[cursor: cursor + n_steps]
        cursor += n_steps
        color = cmap(digit % 10)
        label = f"digit {digit}" if digit not in legend_done else None
        legend_done.add(digit)
        ax.plot(coords[:, 0], coords[:, 1], coords[:, 2], color=color, alpha=0.8, linewidth=1.8, label=label)
        ax.scatter(coords[0, 0], coords[0, 1], coords[0, 2], color=color, marker="o", s=28)
        ax.scatter(coords[-1, 0], coords[-1, 1], coords[-1, 2], color=color, marker="x", s=36)

    ax.set_title(f"{layer_name} Activation Trajectories (3D UMAP)")
    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")
    ax.set_zlabel("UMAP3")
    if legend_done:
        ax.legend(loc="upper right", ncols=2)
    fig.tight_layout()
    return fig


def _save_figure(fig, output_dir: Path | None, name: str) -> None:
    if output_dir is None:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{name}.png", dpi=200, bbox_inches="tight")


def _print_summary(
    checkpoint_path: Path,
    family: str,
    split: str,
    run_args: dict[str, Any],
    checkpoint_payload: dict[str, Any],
    results: dict[str, Any],
    classification: bool,
    trajectory_counts: dict[int, int] | None = None,
    reference_fixed_point_counts: dict[str, int] | None = None,
) -> None:
    print(f"checkpoint: {checkpoint_path}")
    print(f"family: {family}")
    print(f"split: {split}")
    print(f"evaluated_samples: {results['num_samples']}")
    print(f"checkpoint_epoch: {checkpoint_payload.get('epoch', 'n/a')}")
    if "train_loss" in checkpoint_payload:
        print(f"saved_train_loss: {float(checkpoint_payload['train_loss']):.6f}")
    if "val_loss" in checkpoint_payload:
        print(f"saved_val_loss: {float(checkpoint_payload['val_loss']):.6f}")
    if classification and "val_accuracy_percent" in checkpoint_payload:
        print(f"saved_val_accuracy_percent: {float(checkpoint_payload['val_accuracy_percent']):.2f}")
    print(f"eval_loss: {results['mean_loss']:.6f}")
    print(f"eval_recon_loss: {results['mean_recon_loss']:.6f}")
    if classification:
        assert results["mean_class_loss"] is not None
        assert results["final_accuracy_percent"] is not None
        assert results["step_accuracy_percent"] is not None
        print(f"eval_class_loss: {results['mean_class_loss']:.6f}")
        print(f"eval_final_accuracy_percent: {results['final_accuracy_percent']:.2f}")
        print(f"step_accuracy_percent: {[round(float(v), 2) for v in results['step_accuracy_percent']]}")
    if trajectory_counts:
        print(f"trajectory_counts: {trajectory_counts}")
        print(f"umap_available: {_umap_available()}")
    if reference_fixed_point_counts:
        print(f"reference_fixed_points: {reference_fixed_point_counts}")
    print("dataset_config:")
    dataset_keys = (
        "patch_size",
        "mask_ratio",
        "mask_pattern",
        "masked_fill",
        "noise_sigma",
        "visible_corrupt",
        "number_of_masks",
        "timesteps_per_mask",
        "num_digits",
        "image_visibility",
        "batch_size",
    )
    for key in dataset_keys:
        if key in run_args:
            print(f"  {key}: {run_args[key]}")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--checkpoint_path", default=None, type=str)
    parser.add_argument("--family", default=None, type=str, choices=FAMILY_CHOICES)
    parser.add_argument("--split", default="test", choices=("train", "val", "test"))
    parser.add_argument("--batch_size", default=None, type=int)
    parser.add_argument("--num_workers", default=None, type=int)
    parser.add_argument("--patch_size", default=None, type=int)
    parser.add_argument("--mask_ratio", default=None, type=float)
    parser.add_argument("--mask_pattern", default=None, type=str, choices=("random", "structured"))
    parser.add_argument("--masked_fill", default=None, type=str)
    parser.add_argument("--noise_sigma", default=None, type=float)
    parser.add_argument("--visible_corrupt", default=None, action=argparse.BooleanOptionalAction)
    parser.add_argument("--number_of_masks", default=None, type=int)
    parser.add_argument("--timesteps_per_mask", default=None, type=int)
    parser.add_argument("--num_digits", default=None, type=int)
    parser.add_argument("--image_visibility", default=None, type=str)
    parser.add_argument("--accepted_digits", nargs="*", default=None, type=int)
    parser.add_argument("--data_dir", default=None, type=str)
    parser.add_argument("--download", dest="download", action="store_true")
    parser.add_argument("--no-download", dest="download", action="store_false")
    parser.set_defaults(download=True)
    parser.add_argument("--max_batches", default=None, type=int)
    parser.add_argument("--num_examples", default=None, type=int)
    parser.add_argument("--max_time_steps", default=100, type=int)
    parser.add_argument("--classification", action="store_true")
    parser.add_argument("--n_traj_per_digit", default=3, type=int)
    parser.add_argument("--activation_map_sample_idx", default=0, type=int)
    parser.add_argument("--hide_input_grid", action="store_true")
    parser.add_argument("--hide_activation_maps", action="store_true")
    parser.add_argument("--hide_reconstruction_grid", action="store_true")
    parser.add_argument("--hide_training_curves", action="store_true")
    parser.add_argument("--no_show", action="store_true")
    parser.add_argument("--output_dir", default=None, type=str)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda", "mps"))
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_argparser()
    args = parser.parse_args(argv)

    family = _normalize_family_name(args.family) or "hcrnn"
    checkpoint_path = Path(args.checkpoint_path) if args.checkpoint_path is not None else _latest_checkpoint(family)
    if args.family is None:
        family = _infer_family_from_checkpoint_path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")

    run_args = _load_run_args(checkpoint_path)
    device = _resolve_device(args.device)
    output_dir = None if args.output_dir is None else Path(args.output_dir)

    eval_batch_size = int(_config_value(args.batch_size, run_args, "batch_size", 128))
    eval_num_workers = int(_config_value(args.num_workers, run_args, "num_workers", 0))
    num_examples = int(_config_value(args.num_examples, run_args, "num_examples", 4))
    data_dir = _config_value(args.data_dir, run_args, "data_dir", str(DATADIR))
    masked_fill_value = _config_value(args.masked_fill, run_args, "masked_fill", "random")

    dataloader_config = {
        "batch_size": eval_batch_size,
        "root": data_dir,
        "num_workers": eval_num_workers,
        "download": args.download,
        "patch_size": int(_config_value(args.patch_size, run_args, "patch_size", 4)),
        "mask_ratio": float(_config_value(args.mask_ratio, run_args, "mask_ratio", 0.5)),
        "mask_pattern": _config_value(args.mask_pattern, run_args, "mask_pattern", "random"),
        "masked_fill": _parse_masked_fill(masked_fill_value),
        "noise_sigma": float(_config_value(args.noise_sigma, run_args, "noise_sigma", 0.25)),
        "visible_corrupt": _coerce_bool(_config_value(args.visible_corrupt, run_args, "visible_corrupt", False)),
        "number_of_masks": int(_config_value(args.number_of_masks, run_args, "number_of_masks", 1)),
        "timesteps_per_mask": int(_config_value(args.timesteps_per_mask, run_args, "timesteps_per_mask", 1)),
        "num_digits": int(_config_value(args.num_digits, run_args, "num_digits", 1)),
        "image_visibility": _config_value(args.image_visibility, run_args, "image_visibility", "all"),
        "accepted_digits": args.accepted_digits,
        "target_type": "both",
    }

    train_loader, val_loader, test_loader = get_dataloaders(DEFAULT_DATASET, **dataloader_config)
    split_loaders = {"train": train_loader, "val": val_loader, "test": test_loader}
    dataloader = split_loaders[args.split]

    model = _build_model(family, run_args).to(device)
    checkpoint_payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    missing_keys, skipped_keys = _load_model_checkpoint(model, checkpoint_payload)
    if skipped_keys:
        print(f"skipped_state_keys: {skipped_keys}")
    if missing_keys:
        print(f"missing_state_keys: {missing_keys}")

    results = evaluate_model(
        model=model,
        dataloader=dataloader,
        family=family,
        device=device,
        run_args=run_args,
        max_batches=args.max_batches,
        collect_activation_maps=not args.hide_activation_maps,
        classification=args.classification,
        show_progress=True,
    )
    trajectory_payload, trajectory_counts = collect_layer_trajectories(
        model=model,
        dataloader=dataloader,
        device=device,
        n_traj_per_digit=args.n_traj_per_digit,
        accepted_digits=args.accepted_digits,
        max_batches=args.max_batches,
        show_progress=True,
    )
    pca_reducers_2d: dict[str, dict[str, torch.Tensor]] = {}
    pca_reducers_3d: dict[str, dict[str, torch.Tensor]] = {}
    umap_reducers_2d: dict[str, Any] = {}
    umap_reducers_3d: dict[str, Any] = {}
    reference_fixed_point_counts: dict[str, int] | None = None
    if trajectory_payload:
        clean_dataloader_config = dict(dataloader_config)
        clean_dataloader_config["mask_ratio"] = 0.0
        clean_dataloader_config["noise_sigma"] = 0.0
        clean_dataloader_config["visible_corrupt"] = False
        clean_dataloader_config["image_visibility"] = "first"
        clean_train_loader, _, _ = get_dataloaders(DEFAULT_DATASET, **clean_dataloader_config)
        umap_reference_points = collect_layer_fixed_points(
            model=model,
            dataloader=clean_train_loader,
            device=device,
            show_progress=True,
        )
        reference_fixed_point_counts = {
            layer_name: int(points.shape[0])
            for layer_name, points in umap_reference_points.items()
        }
        pca_reducers_2d = _fit_pca_reducers(
            umap_reference_points,
            n_components=2,
            show_progress=True,
        )
        pca_reducers_3d = _fit_pca_reducers(
            umap_reference_points,
            n_components=3,
            show_progress=True,
        )
        if _umap_available():
            umap_reducers_2d = _fit_umap_reducers_parallel(
                umap_reference_points,
                n_components=2,
                show_progress=True,
            )
            umap_reducers_3d = _fit_umap_reducers_parallel(
                umap_reference_points,
                n_components=3,
                show_progress=True,
            )
    _print_summary(
        checkpoint_path=checkpoint_path,
        family=family,
        split=args.split,
        run_args=run_args | dataloader_config,
        checkpoint_payload=checkpoint_payload,
        results=results,
        classification=args.classification,
        trajectory_counts=trajectory_counts,
        reference_fixed_point_counts=reference_fixed_point_counts,
    )

    payload = results["visual_payload"]
    if payload is None:
        return
    if not (0 <= args.activation_map_sample_idx < payload["masked_inputs"].shape[0]):
        raise ValueError(
            f"activation_map_sample_idx must be in [0, {payload['masked_inputs'].shape[0] - 1}], "
            f"got {args.activation_map_sample_idx}."
        )

    import matplotlib.pyplot as plt

    if not args.hide_input_grid:
        input_fig = create_activation_input_figure(
            payload["masked_inputs"],
            sample_idx=args.activation_map_sample_idx,
            max_time_steps=args.max_time_steps,
        )
        _save_figure(input_fig, output_dir, "masked_inputs")

    recon_loss_fig = create_temporal_reconstruction_figure(
        payload["recon"],
        payload["clean_image"],
        max_examples=num_examples,
    )
    _save_figure(recon_loss_fig, output_dir, "reconstruction_loss_over_time")

    if args.classification:
        pred_fig = create_temporal_prediction_figure(
            payload["class_logits"],
            payload["labels"],
            max_examples=num_examples,
        )
        _save_figure(pred_fig, output_dir, "prediction_confidence_over_time")

        if results["step_accuracy_percent"] is not None:
            acc_fig = _create_step_accuracy_figure(results["step_accuracy_percent"])
            _save_figure(acc_fig, output_dir, "step_accuracy")

    if not args.hide_reconstruction_grid:
        recon_grid = create_temporal_reconstruction_grid(
            payload["clean_image"],
            payload["masked_inputs"],
            payload["recon"],
            num_examples=num_examples,
            max_time_steps=args.max_time_steps,
        )
        recon_grid_fig = _show_image_grid(
            recon_grid,
            title="Masked inputs (top); reconstructions (middle); unmasked targets (bottom)",
        )
        _save_figure(recon_grid_fig, output_dir, "reconstruction_grid")

    run_dir = checkpoint_path.parent
    if not args.hide_training_curves:
        training_curves_fig = _create_training_curves_figure(run_dir)
        if training_curves_fig is not None:
            _save_figure(training_curves_fig, output_dir, "training_curves")

    activation_maps = payload["activation_maps"]
    if activation_maps is not None and not args.hide_activation_maps:
        for layer_name in activation_maps["Y"].keys():
            layer_fig = create_activation_map_figure(
                activation_maps,
                layer_name=layer_name,
                sample_idx=args.activation_map_sample_idx,
                max_time_steps=args.max_time_steps,
                mode=family,
            )
            _save_figure(layer_fig, output_dir, f"activation_{layer_name.lower()}")

    for layer_name, trajectories in trajectory_payload.items():
        dynamics_fig_2d = create_layer_trajectory_pca_2d_figure(
            layer_name,
            trajectories,
            reducer=pca_reducers_2d.get(layer_name),
        )
        if dynamics_fig_2d is not None:
            _save_figure(dynamics_fig_2d, output_dir, f"dynamics_pca2d_{layer_name.lower()}")
        dynamics_fig_3d = create_layer_trajectory_pca_3d_figure(
            layer_name,
            trajectories,
            reducer=pca_reducers_3d.get(layer_name),
        )
        if dynamics_fig_3d is not None:
            _save_figure(dynamics_fig_3d, output_dir, f"dynamics_pca3d_{layer_name.lower()}")
        dynamics_umap_fig_2d = create_layer_trajectory_umap_2d_figure(
            layer_name,
            trajectories,
            reducer=umap_reducers_2d.get(layer_name),
        )
        if dynamics_umap_fig_2d is not None:
            _save_figure(dynamics_umap_fig_2d, output_dir, f"dynamics_umap2d_{layer_name.lower()}")
        dynamics_umap_fig_3d = create_layer_trajectory_umap_3d_figure(
            layer_name,
            trajectories,
            reducer=umap_reducers_3d.get(layer_name),
        )
        if dynamics_umap_fig_3d is not None:
            _save_figure(dynamics_umap_fig_3d, output_dir, f"dynamics_umap3d_{layer_name.lower()}")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
