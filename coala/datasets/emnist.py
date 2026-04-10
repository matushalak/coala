from __future__ import annotations

from pathlib import Path
from typing import Sequence

from coala import DATADIR
from coala.datasets.common import (
    DEFAULT_GRAY_MEAN,
    DEFAULT_GRAY_STD,
    DatasetBundle,
    build_dataloaders,
    build_grayscale_transform,
    dataset_root,
    split_dataset,
)


def build_emnist_datasets(
    root: str | Path = DATADIR,
    split: str = "balanced",
    download: bool = False,
    image_size: int | tuple[int, int] | None = 28,
    val_fraction: float = 0.1,
    seed: int = 42,
    mean: Sequence[float] = DEFAULT_GRAY_MEAN,
    std: Sequence[float] = DEFAULT_GRAY_STD,
) -> DatasetBundle:
    import torchvision

    data_root = dataset_root("emnist", root=root)
    transform = build_grayscale_transform(image_size=image_size, mean=mean, std=std)

    train_full = torchvision.datasets.EMNIST(
        root=data_root,
        split=split,
        train=True,
        transform=transform,
        download=download,
    )
    test_dataset = torchvision.datasets.EMNIST(
        root=data_root,
        split=split,
        train=False,
        transform=transform,
        download=download,
    )
    train_dataset, val_dataset = split_dataset(train_full, val_fraction=val_fraction, seed=seed)

    return DatasetBundle(
        name="emnist",
        root=data_root,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        metadata={"split": split},
    )


def emnist(
    root: str | Path = DATADIR,
    batch_size: int = 128,
    num_workers: int = 4,
    split: str = "balanced",
    download: bool = False,
    image_size: int | tuple[int, int] | None = 28,
    val_fraction: float = 0.1,
    seed: int = 42,
    mean: Sequence[float] = DEFAULT_GRAY_MEAN,
    std: Sequence[float] = DEFAULT_GRAY_STD,
):
    bundle = build_emnist_datasets(
        root=root,
        split=split,
        download=download,
        image_size=image_size,
        val_fraction=val_fraction,
        seed=seed,
        mean=mean,
        std=std,
    )
    return build_dataloaders(bundle, batch_size=batch_size, num_workers=num_workers)


def download_emnist(root: str | Path = DATADIR, split: str = "balanced") -> Path:
    import torchvision

    data_root = dataset_root("emnist", root=root)
    torchvision.datasets.EMNIST(root=data_root, split=split, train=True, download=True)
    torchvision.datasets.EMNIST(root=data_root, split=split, train=False, download=True)
    return data_root
