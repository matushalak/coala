from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import torch

# Allow direct script execution from inside this folder: `python test_mnist_datasets.py`.
if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[3]))


class _DummyMNIST:
    def __init__(self, root, train, transform, download):
        del root, download
        self.transform = transform
        self.train = train
        self.size = 60000 if train else 10000
        self.targets = torch.arange(self.size, dtype=torch.long) % 10
        base = np.arange(28 * 28, dtype=np.uint16).reshape(28, 28)
        self._base_image = (base % 256).astype(np.uint8)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        offset = np.uint8((idx * 17) % 256)
        img = (self._base_image + offset).astype(np.uint8)
        label = int(self.targets[idx].item())
        if self.transform is not None:
            img = self.transform(img)
        return img, label


def _assert_mnist_loader_outputs_centered_range():
    from coala.datasets.mnist import mnist

    train_loader, val_loader, test_loader = mnist(batch_size=8, num_workers=0, download=False)

    assert len(train_loader.dataset) == 54000
    assert len(val_loader.dataset) == 6000
    assert len(test_loader.dataset) == 10000

    for loader in (train_loader, val_loader, test_loader):
        imgs, labels = next(iter(loader))
        assert imgs.dtype == torch.float32
        assert torch.all((-1.0 <= imgs) & (imgs <= 1.0))
        assert labels.dtype == torch.int64


def _assert_msmnist_loader_outputs_centered_range():
    from coala.datasets.msmnist import msmnist

    train_loader, val_loader, test_loader = msmnist(
        batch_size=4,
        num_workers=0,
        download=False,
        patch_size=4,
        mask_ratio=0.5,
        masked_fill='random',
        number_of_masks=2,
        timesteps_per_mask=3,
        target_type="image",
    )

    assert len(train_loader.dataset) == 54000
    assert len(val_loader.dataset) == 6000
    assert len(test_loader.dataset) == 10000

    for loader in (train_loader, val_loader, test_loader):
        masked_imgs, targets = next(iter(loader))
        assert masked_imgs.dtype == torch.float32
        assert targets.dtype == torch.float32
        assert torch.all((-1.0 <= masked_imgs) & (masked_imgs <= 1.0))
        assert torch.all((-1.0 <= targets) & (targets <= 1.0))


def test_mnist_loader_outputs_centered_range(monkeypatch):
    import torchvision

    monkeypatch.setattr(torchvision.datasets, "MNIST", _DummyMNIST)
    _assert_mnist_loader_outputs_centered_range()


def test_msmnist_loader_outputs_centered_range(monkeypatch):
    import torchvision

    monkeypatch.setattr(torchvision.datasets, "MNIST", _DummyMNIST)
    _assert_msmnist_loader_outputs_centered_range()


if __name__ == "__main__":
    import torchvision
    from unittest.mock import patch

    with patch.object(torchvision.datasets, "MNIST", _DummyMNIST):
        _assert_mnist_loader_outputs_centered_range()
        _assert_msmnist_loader_outputs_centered_range()

    print("MNIST/MSMNIST dataset tests passed.")
