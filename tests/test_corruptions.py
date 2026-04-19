from __future__ import annotations

from pathlib import Path
import sys

import torch
import torchvision
from torchvision import transforms

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from coala import DATADIR
from coala.datasets.corrupted_sequential import CorruptedSequentialDataset


class _SingleExampleDataset:
    def __init__(self, image: torch.Tensor, label: int):
        self._image = image.clone()
        self.targets = torch.tensor([label], dtype=torch.long)

    def __len__(self) -> int:
        return 1

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        if idx != 0:
            raise IndexError(idx)
        return self._image.clone(), int(self.targets[0].item())


def _stack_sequence(sequence: torch.Tensor, padding: int = 2, pad_value: float = -0.35) -> torch.Tensor:
    frames = [frame for frame in sequence]
    if len(frames) == 1:
        return frames[0]
    spacer = torch.full(
        (frames[0].shape[0], frames[0].shape[1], padding),
        pad_value,
        dtype=frames[0].dtype,
    )
    pieces = []
    for frame_idx, frame in enumerate(frames):
        if frame_idx > 0:
            pieces.append(spacer)
        pieces.append(frame)
    return torch.cat(pieces, dim=-1)


def _mnist_transform():
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5,), std=(0.5,)),
        ]
    )


def _cifar_transform():
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ]
    )


def _load_reference_examples(root: str | Path, *, download: bool) -> dict[str, _SingleExampleDataset]:
    mnist = torchvision.datasets.MNIST(
        root=root,
        train=True,
        transform=_mnist_transform(),
        download=download,
    )
    cifar = torchvision.datasets.CIFAR10(
        root=root,
        train=True,
        transform=_cifar_transform(),
        download=download,
    )

    mnist_img, mnist_label = mnist[0]
    cifar_img, cifar_label = cifar[0]
    return {
        "MNIST": _SingleExampleDataset(mnist_img, int(mnist_label)),
        "CIFAR-10": _SingleExampleDataset(cifar_img, int(cifar_label)),
    }


def _sample_sequence(
    base_dataset: _SingleExampleDataset,
    *,
    corruption: tuple[str, ...],
    mask_pattern: str = "random",
    seed: int,
    severity: dict[str, float | int],
) -> torch.Tensor:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        dataset = CorruptedSequentialDataset(
            base_dataset,
            patch_size=4,
            mask_ratio=float(severity["mask_ratio"]),
            number_of_masks=4,
            timesteps_per_mask=1,
            mask_pattern=mask_pattern,
            masked_fill=0.0,
            noise_sigma=float(severity["noise_sigma"]),
            corruptions=corruption,
            mix_alpha=float(severity["mix_alpha"]),
            mix_noise_sigma=1.0,
            salt_pepper_prob=float(severity["salt_pepper_prob"]),
            blur_kernel_size=int(severity["blur_kernel_size"]),
            blur_sigma=float(severity["blur_sigma"]),
            target_type="image",
        )
        masked_imgs, _ = dataset[0]
    return masked_imgs


def _plot_panel(ax, panel: torch.Tensor) -> None:
    panel = panel.detach().cpu()
    if panel.shape[0] == 1:
        ax.imshow(panel[0], cmap="gray", interpolation="nearest", vmin=-1.0, vmax=1.0)
        return
    ax.imshow(panel.movedim(0, -1).add(1.0).mul(0.5).clamp(0.0, 1.0), interpolation="nearest")


def _render_corruption_gallery(
    save_path: str | Path,
    *,
    show: bool = False,
    root: str | Path = DATADIR,
    download: bool = False,
) -> None:
    import matplotlib

    if not show:
        matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    examples = _load_reference_examples(root, download=download)
    corruption_rows = [
        ("Structured Mask", ("mask",), "structured"),
        ("Random Mask", ("mask",), "random"),
        ("Gaussian", ("gaussian",), "random"),
        ("Mix", ("mix",), "random"),
        ("Gaussian Mix + Blur", ("mix_blur",), "random"),
        ("Salt + Pepper", ("salt_pepper",), "random"),
        ("Blur", ("blur",), "random"),
    ]
    severities = [
        ("Reference", None),
        (
            "Low",
            {
                "mask_ratio": 0.20,
                "noise_sigma": 0.15,
                "mix_alpha": 0.20,
                "salt_pepper_prob": 0.08,
                "blur_kernel_size": 3,
                "blur_sigma": 0.6,
            },
        ),
        (
            "Medium",
            {
                "mask_ratio": 0.45,
                "noise_sigma": 0.35,
                "mix_alpha": 0.45,
                "salt_pepper_prob": 0.18,
                "blur_kernel_size": 5,
                "blur_sigma": 1.0,
            },
        ),
        (
            "High",
            {
                "mask_ratio": 0.70,
                "noise_sigma": 0.60,
                "mix_alpha": 0.75,
                "salt_pepper_prob": 0.32,
                "blur_kernel_size": 9,
                "blur_sigma": 3.0,
            },
        ),
    ]
    dataset_rows = list(examples.items())
    num_rows = len(dataset_rows) * len(corruption_rows)

    fig, axes = plt.subplots(
        num_rows,
        len(severities),
        squeeze=False,
        figsize=(3.2 * len(severities), 2.1 * num_rows),
    )
    for dataset_idx, (dataset_name, base_dataset) in enumerate(dataset_rows):
        base_img, _ = base_dataset[0]
        base_sequence = base_img.unsqueeze(0).repeat(4, 1, 1, 1)
        for corruption_idx, (corruption_name, corruption, mask_pattern) in enumerate(corruption_rows):
            row_idx = dataset_idx * len(corruption_rows) + corruption_idx
            for col_idx, (severity_name, severity) in enumerate(severities):
                ax = axes[row_idx][col_idx]
                if severity is None:
                    panel = _stack_sequence(base_sequence)
                else:
                    sequence = _sample_sequence(
                        base_dataset,
                        corruption=corruption,
                        mask_pattern=mask_pattern,
                        seed=1000 * dataset_idx + 100 * corruption_idx + col_idx,
                        severity=severity,
                    )
                    panel = _stack_sequence(sequence)
                _plot_panel(ax, panel)
                ax.set_xticks([])
                ax.set_yticks([])
                if row_idx == 0:
                    ax.set_title(severity_name, fontsize=10)
                if col_idx == 0:
                    ax.set_ylabel(f"{dataset_name}\n{corruption_name}", fontsize=10)

    fig.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=160, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def test_corrupted_sequential_visualization_smoke(tmp_path):
    pytest = __import__("pytest")

    pytest.importorskip("matplotlib")
    pytest.importorskip("torchvision")
    out_path = tmp_path / "corrupted_sequential_gallery.png"
    try:
        _render_corruption_gallery(out_path, show=False, download=False)
    except RuntimeError as exc:
        if "Dataset not found" in str(exc):
            pytest.skip("MNIST/CIFAR-10 not available locally for corruption visualization smoke test.")
        raise
    assert out_path.exists() and out_path.stat().st_size > 0


if __name__ == "__main__":
    output_path = Path(__file__).with_name("corrupted_sequential_gallery.png")
    _render_corruption_gallery(output_path, show=False, download=True)
    print(f"Saved {output_path}")
