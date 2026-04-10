from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.utils.data as data

from coala import DATADIR
from coala.datasets.common import DatasetBundle, build_dataloaders, dataset_root, download_url

MOVING_MNIST_URL = "http://www.cs.toronto.edu/~nitish/unsupervised_video/mnist_test_seq.npy"


class MovingMNISTDataset(data.Dataset):
    def __init__(
        self,
        file_path: str | Path,
        split: str = "train",
        sequence_length: int | None = None,
        split_fractions: Sequence[float] = (0.8, 0.1, 0.1),
    ):
        self.file_path = Path(file_path)
        if not self.file_path.exists():
            raise FileNotFoundError(f"MovingMNIST file not found: {self.file_path}")

        self.data = np.load(self.file_path, mmap_mode="r")
        if self.data.ndim != 4:
            raise ValueError(
                f"Expected MovingMNIST array with 4 dimensions, got {self.data.shape}"
            )

        self.sequence_length = sequence_length or int(self.data.shape[0])
        self.sequence_length = min(self.sequence_length, int(self.data.shape[0]))

        num_sequences = int(self.data.shape[1])
        train_fraction, val_fraction, test_fraction = split_fractions
        if abs((train_fraction + val_fraction + test_fraction) - 1.0) > 1e-6:
            raise ValueError("split_fractions must sum to 1.")

        train_end = int(num_sequences * train_fraction)
        val_end = train_end + int(num_sequences * val_fraction)

        split_to_slice = {
            "train": slice(0, train_end),
            "val": slice(train_end, val_end),
            "test": slice(val_end, num_sequences),
        }
        if split not in split_to_slice:
            raise ValueError(f"Unknown split '{split}'. Expected one of {tuple(split_to_slice)}")

        selected = split_to_slice[split]
        self.indices = list(range(num_sequences))[selected]
        if not self.indices:
            raise ValueError(f"Split '{split}' produced no MovingMNIST sequences.")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        sample_idx = self.indices[idx]
        sequence = np.asarray(
            self.data[: self.sequence_length, sample_idx],
            dtype=np.float32,
        )
        sequence = torch.from_numpy(sequence).unsqueeze(1) / 255.0
        sequence = sequence.mul(2.0).sub(1.0)
        return sequence, torch.tensor(-1, dtype=torch.long)


def build_moving_mnist_datasets(
    root: str | Path = DATADIR,
    download: bool = False,
    filename: str = "mnist_test_seq.npy",
    sequence_length: int | None = None,
) -> DatasetBundle:
    data_root = dataset_root("moving_mnist", root=root)
    file_path = data_root / filename

    if download and not file_path.exists():
        download_moving_mnist(root=root, filename=filename)

    if not file_path.exists():
        raise FileNotFoundError(
            f"MovingMNIST file is missing at {file_path}. Run download_moving_mnist(...) first."
        )

    train_dataset = MovingMNISTDataset(file_path, split="train", sequence_length=sequence_length)
    val_dataset = MovingMNISTDataset(file_path, split="val", sequence_length=sequence_length)
    test_dataset = MovingMNISTDataset(file_path, split="test", sequence_length=sequence_length)

    return DatasetBundle(
        name="moving_mnist",
        root=data_root,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        metadata={"sequence_length": sequence_length},
    )


def moving_mnist(
    root: str | Path = DATADIR,
    batch_size: int = 64,
    num_workers: int = 4,
    download: bool = False,
    filename: str = "mnist_test_seq.npy",
    sequence_length: int | None = None,
):
    bundle = build_moving_mnist_datasets(
        root=root,
        download=download,
        filename=filename,
        sequence_length=sequence_length,
    )
    return build_dataloaders(bundle, batch_size=batch_size, num_workers=num_workers)


def download_moving_mnist(
    root: str | Path = DATADIR,
    filename: str = "mnist_test_seq.npy",
) -> Path:
    data_root = dataset_root("moving_mnist", root=root)
    return download_url(MOVING_MNIST_URL, destination_dir=data_root, filename=filename)
