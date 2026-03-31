from __future__ import annotations

from pathlib import Path
from typing import Sequence

from torch.utils.data import ConcatDataset

from cc import DATADIR
from cc.datasets.common import (
    DEFAULT_RGB_MEAN,
    DEFAULT_RGB_STD,
    DatasetBundle,
    build_dataloaders,
    build_rgb_transform,
    dataset_root,
    split_dataset,
)


def build_stl10_datasets(
    root: str | Path = DATADIR,
    download: bool = False,
    include_unlabeled: bool = True,
    image_size: int | tuple[int, int] | None = 96,
    val_fraction: float = 0.1,
    seed: int = 42,
    mean: Sequence[float] = DEFAULT_RGB_MEAN,
    std: Sequence[float] = DEFAULT_RGB_STD,
) -> DatasetBundle:
    import torchvision

    data_root = dataset_root("stl10", root=root)
    transform = build_rgb_transform(image_size=image_size, mean=mean, std=std)

    labeled_train = torchvision.datasets.STL10(
        root=data_root,
        split="train",
        transform=transform,
        download=download,
    )
    test_dataset = torchvision.datasets.STL10(
        root=data_root,
        split="test",
        transform=transform,
        download=download,
    )

    train_dataset, val_dataset = split_dataset(labeled_train, val_fraction=val_fraction, seed=seed)
    metadata = {"include_unlabeled": include_unlabeled}

    if include_unlabeled:
        unlabeled_dataset = torchvision.datasets.STL10(
            root=data_root,
            split="unlabeled",
            transform=transform,
            download=download,
        )
        train_dataset = ConcatDataset([train_dataset, unlabeled_dataset])
        metadata["unlabeled_examples"] = len(unlabeled_dataset)

    return DatasetBundle(
        name="stl10",
        root=data_root,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        metadata=metadata,
    )


def stl10(
    root: str | Path = DATADIR,
    batch_size: int = 128,
    num_workers: int = 4,
    download: bool = False,
    include_unlabeled: bool = True,
    image_size: int | tuple[int, int] | None = 96,
    val_fraction: float = 0.1,
    seed: int = 42,
    mean: Sequence[float] = DEFAULT_RGB_MEAN,
    std: Sequence[float] = DEFAULT_RGB_STD,
):
    bundle = build_stl10_datasets(
        root=root,
        download=download,
        include_unlabeled=include_unlabeled,
        image_size=image_size,
        val_fraction=val_fraction,
        seed=seed,
        mean=mean,
        std=std,
    )
    return build_dataloaders(bundle, batch_size=batch_size, num_workers=num_workers)


def download_stl10(root: str | Path = DATADIR, include_unlabeled: bool = True) -> Path:
    import torchvision

    data_root = dataset_root("stl10", root=root)
    splits = ["train", "test"]
    if include_unlabeled:
        splits.append("unlabeled")

    for split in splits:
        torchvision.datasets.STL10(root=data_root, split=split, download=True)
    return data_root
