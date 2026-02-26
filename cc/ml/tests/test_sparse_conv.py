from __future__ import annotations

from pathlib import Path
import sys

import torch

# Allow direct script execution from inside this folder: `python test_sparse_conv.py`.
if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[3]))

from cc.ml.sparse_cnn_unet import SparseConv2d


def _load_mnist_digit_mask() -> torch.BoolTensor:
    import torchvision
    from torchvision import transforms

    dataset = torchvision.datasets.MNIST(
        root=".../data",
        train=False,
        download=True,
        transform=transforms.ToTensor(),
    )
    img, _ = dataset[0]  # shape: [1, 28, 28], values in [0, 1]
    return (img > 0.2).unsqueeze(0)  # [1, 1, 28, 28], True on digit foreground


def _run_sparse_vs_regular_demo(out_path: Path, steps: int = 4, seed: int = 0) -> tuple[int, int]:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    torch.manual_seed(seed)
    foreground_mask = _load_mnist_digit_mask()
    keep_mask = ~foreground_mask  # background kept, foreground masked
    # SparK-style: real background is black (known), masked foreground starts white.
    x = keep_mask.float()

    regular_conv = torch.nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)
    sparse_conv = SparseConv2d(1, 1, kernel_size=3, padding=1, bias=False)
    kernel = torch.full_like(regular_conv.weight, 1.0)
    with torch.no_grad():
        regular_conv.weight.copy_(kernel)
        sparse_conv.weight.copy_(kernel)

    regular_states = [x]
    sparse_states = [x]
    with torch.no_grad():
        for _ in range(steps):
            regular_states.append(regular_conv(regular_states[-1]))
            sparse_states.append(sparse_conv(sparse_states[-1], keep_mask))

    masked_region = foreground_mask
    sparse_nonzero = int(torch.count_nonzero(sparse_states[-1][masked_region] > 1e-8))
    regular_nonzero = int(torch.count_nonzero(regular_states[-1][masked_region] > 1e-8))

    fig, axes = plt.subplots(2, steps + 1, figsize=(1.6 * (steps + 1), 3.2), constrained_layout=True)
    for idx in range(steps + 1):
        axes[0, idx].imshow(regular_states[idx].squeeze(0).squeeze(0).cpu().clamp(0.0, 1.0), cmap="gray_r", vmin=0.0, vmax=1.0)
        axes[1, idx].imshow(sparse_states[idx].squeeze(0).squeeze(0).cpu().clamp(0.0, 1.0), cmap="gray_r", vmin=0.0, vmax=1.0)
        axes[0, idx].set_title(f"t={idx}")
        axes[0, idx].axis("off")
        axes[1, idx].axis("off")
    axes[0, 0].set_ylabel("Regular")
    axes[1, 0].set_ylabel("Sparse")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return sparse_nonzero, regular_nonzero


def test_sparse_conv_keeps_masked_pixels_zero_during_successive_application(tmp_path):
    pytest = __import__("pytest")
    pytest.importorskip("matplotlib")
    pytest.importorskip("torchvision")

    out_path = tmp_path / "regular_vs_sparse_successive_conv.png"
    sparse_nonzero, regular_nonzero = _run_sparse_vs_regular_demo(out_path=out_path, steps=4)

    assert sparse_nonzero == 0
    assert regular_nonzero > 0
    assert out_path.exists() and out_path.stat().st_size > 0


if __name__ == "__main__":
    output_path = Path(__file__).resolve().parent / "regular_vs_sparse_successive_conv.png"
    sparse_nonzero, regular_nonzero = _run_sparse_vs_regular_demo(out_path=output_path, steps=4)
    print(f"Saved visualization to: {output_path}")
    print(f"Masked-region nonzeros after 4 steps | regular: {regular_nonzero}, sparse: {sparse_nonzero}")
