from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest
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


def test_msmnist_noise_sigma_reaches_masked_sequence(monkeypatch):
    import torchvision

    from coala.datasets.msmnist import msmnist

    monkeypatch.setattr(torchvision.datasets, "MNIST", _DummyMNIST)
    train_loader, _, _ = msmnist(
        batch_size=2,
        num_workers=0,
        download=False,
        patch_size=4,
        mask_ratio=1.0,
        masked_fill="random",
        noise_sigma=0.0,
        number_of_masks=1,
        timesteps_per_mask=1,
        target_type="image",
    )
    masked_imgs, _ = next(iter(train_loader))
    assert torch.count_nonzero(masked_imgs) == 0


def test_msmnist_num_digits_and_image_visibility(monkeypatch):
    import torchvision

    from coala.datasets.msmnist import msmnist

    monkeypatch.setattr(torchvision.datasets, "MNIST", _DummyMNIST)
    torch.manual_seed(0)
    train_loader, _, _ = msmnist(
        batch_size=2,
        num_workers=0,
        download=False,
        patch_size=4,
        mask_ratio=0.5,
        masked_fill=0.0,
        noise_sigma=0.0,
        number_of_masks=2,
        timesteps_per_mask=2,
        num_digits=3,
        image_visibility="first",
        target_type="both",
    )

    masked_imgs, targets = next(iter(train_loader))
    assert masked_imgs.shape == (2, 12, 1, 28, 28)
    assert targets["image"].shape == (2, 12, 1, 28, 28)
    assert targets["label"].shape == (2, 12)

    segment_len = 4
    for digit_idx in range(3):
        start = digit_idx * segment_len
        end = start + segment_len
        label_slice = targets["label"][0, start:end]
        assert torch.all(label_slice == label_slice[0])

    assert len(set(targets["label"][0, ::segment_len].tolist())) == 3
    assert torch.all(masked_imgs[:, 1:4] == -1.0)
    assert torch.all(masked_imgs[:, 5:8] == -1.0)
    assert torch.all(masked_imgs[:, 9:12] == -1.0)


def test_msmnist_contrastive_requires_structured_mask(monkeypatch):
    import torchvision

    from coala.datasets.msmnist import msmnist

    monkeypatch.setattr(torchvision.datasets, "MNIST", _DummyMNIST)
    with pytest.raises(ValueError, match="structured"):
        msmnist(
            batch_size=2,
            num_workers=0,
            download=False,
            patch_size=4,
            mask_ratio=0.5,
            mask_pattern="random",
            masked_fill=0.0,
            noise_sigma=0.0,
            number_of_masks=2,
            timesteps_per_mask=1,
            target_type="both",
            contrastive=True,
        )


def test_msmnist_contrastive_interleaves_positive_pairs(monkeypatch):
    import torchvision

    from coala.datasets.msmnist import msmnist

    monkeypatch.setattr(torchvision.datasets, "MNIST", _DummyMNIST)
    masked_fill = 2.0

    def _load_batch(mask_ratio: float):
        torch.manual_seed(0)
        train_loader, _, _ = msmnist(
            batch_size=2,
            num_workers=0,
            download=False,
            patch_size=4,
            mask_ratio=mask_ratio,
            mask_pattern="structured",
            masked_fill=masked_fill,
            noise_sigma=0.0,
            number_of_masks=2,
            timesteps_per_mask=1,
            target_type="both",
            contrastive=True,
        )
        return next(iter(train_loader))

    masked_imgs_mid, targets_mid = _load_batch(mask_ratio=0.5)
    masked_imgs_high, _ = _load_batch(mask_ratio=0.8)
    masked_imgs_zero, targets_zero = _load_batch(mask_ratio=0.0)

    assert masked_imgs_mid.shape == (4, 2, 1, 28, 28)
    assert targets_mid["image"].shape == (4, 2, 1, 28, 28)
    assert targets_mid["label"].shape == (4, 2)
    assert targets_mid["contrastive_group"].tolist() == [0, 0, 1, 1]
    assert targets_mid["contrastive_view"].tolist() == [0, 1, 0, 1]
    assert targets_mid["contrastive_positive_index"].tolist() == [1, 0, 3, 2]
    assert torch.equal(targets_mid["image"][0], targets_mid["image"][1])
    assert torch.equal(targets_mid["image"][2], targets_mid["image"][3])
    assert torch.equal(targets_mid["label"][0], targets_mid["label"][1])
    assert torch.equal(targets_mid["label"][2], targets_mid["label"][3])

    mid_view0_masked = (masked_imgs_mid[0] == masked_fill).sum().item()
    mid_view1_masked = (masked_imgs_mid[1] == masked_fill).sum().item()
    high_view0_masked = (masked_imgs_high[0] == masked_fill).sum().item()
    high_view1_masked = (masked_imgs_high[1] == masked_fill).sum().item()

    assert mid_view1_masked > mid_view0_masked
    assert high_view0_masked > mid_view0_masked
    assert high_view1_masked == mid_view1_masked
    assert torch.equal(masked_imgs_zero[1], targets_zero["image"][1])


if __name__ == "__main__":
    import torchvision
    from unittest.mock import patch

    from coala.datasets.msmnist import msmnist

    with patch.object(torchvision.datasets, "MNIST", _DummyMNIST):
        _assert_mnist_loader_outputs_centered_range()
        _assert_msmnist_loader_outputs_centered_range()

        train_loader, _, _ = msmnist(
            batch_size=2,
            num_workers=0,
            download=False,
            patch_size=4,
            mask_ratio=1.0,
            masked_fill="random",
            noise_sigma=0.0,
            number_of_masks=1,
            timesteps_per_mask=1,
            target_type="image",
        )
        masked_imgs, _ = next(iter(train_loader))
        assert torch.count_nonzero(masked_imgs) == 0

        torch.manual_seed(0)
        train_loader, _, _ = msmnist(
            batch_size=2,
            num_workers=0,
            download=False,
            patch_size=4,
            mask_ratio=0.5,
            masked_fill=0.0,
            noise_sigma=0.0,
            number_of_masks=2,
            timesteps_per_mask=2,
            num_digits=3,
            image_visibility="first",
            target_type="both",
        )
        masked_imgs, targets = next(iter(train_loader))
        assert masked_imgs.shape == (2, 12, 1, 28, 28)
        assert targets["image"].shape == (2, 12, 1, 28, 28)
        assert targets["label"].shape == (2, 12)
        assert len(set(targets["label"][0, ::4].tolist())) == 3
        assert torch.all(masked_imgs[:, 1:4] == -1.0)
        assert torch.all(masked_imgs[:, 5:8] == -1.0)
        assert torch.all(masked_imgs[:, 9:12] == -1.0)

        with pytest.raises(ValueError, match="structured"):
            msmnist(
                batch_size=2,
                num_workers=0,
                download=False,
                patch_size=4,
                mask_ratio=0.5,
                mask_pattern="random",
                masked_fill=0.0,
                noise_sigma=0.0,
                number_of_masks=2,
                timesteps_per_mask=1,
                target_type="both",
                contrastive=True,
            )

        masked_fill = 2.0

        def _load_batch(mask_ratio: float):
            torch.manual_seed(0)
            train_loader, _, _ = msmnist(
                batch_size=2,
                num_workers=0,
                download=False,
                patch_size=4,
                mask_ratio=mask_ratio,
                mask_pattern="structured",
                masked_fill=masked_fill,
                noise_sigma=0.0,
                number_of_masks=2,
                timesteps_per_mask=1,
                target_type="both",
                contrastive=True,
            )
            return next(iter(train_loader))

        masked_imgs_mid, targets_mid = _load_batch(mask_ratio=0.5)
        masked_imgs_high, _ = _load_batch(mask_ratio=0.8)
        masked_imgs_zero, targets_zero = _load_batch(mask_ratio=0.0)
        assert masked_imgs_mid.shape == (4, 2, 1, 28, 28)
        assert targets_mid["contrastive_positive_index"].tolist() == [1, 0, 3, 2]
        assert (masked_imgs_mid[1] == masked_fill).sum().item() > (masked_imgs_mid[0] == masked_fill).sum().item()
        assert (masked_imgs_high[0] == masked_fill).sum().item() > (masked_imgs_mid[0] == masked_fill).sum().item()
        assert (masked_imgs_high[1] == masked_fill).sum().item() == (masked_imgs_mid[1] == masked_fill).sum().item()
        assert torch.equal(masked_imgs_zero[1], targets_zero["image"][1])

    print("MNIST/MSMNIST dataset tests passed.")
