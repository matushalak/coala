from __future__ import annotations

from pathlib import Path
from typing import Sequence

from cc import DATADIR
from cc.datasets.common import DEFAULT_RGB_MEAN, DEFAULT_RGB_STD, build_dataloaders, build_rgb_transform
from cc.datasets.common import dataset_root
from cc.datasets.folder_based import recursive_image_bundle


def download_openimages_v7(root: str | Path = DATADIR) -> Path:
    data_root = dataset_root("openimages_v7", root=root)
    raise NotImplementedError(
        f"Automatic OpenImages V7 download is not implemented. "
        f"Place the extracted train/validation/test image folders under {data_root}."
    )


def build_openimages_v7_datasets(
    root: str | Path = DATADIR,
    download: bool = False,
    image_size: int | tuple[int, int] | None = 224,
    val_fraction: float = 0.1,
    test_fraction: float = 0.1,
    seed: int = 42,
    mean: Sequence[float] = DEFAULT_RGB_MEAN,
    std: Sequence[float] = DEFAULT_RGB_STD,
):
    transform = build_rgb_transform(image_size=image_size, mean=mean, std=std)
    return recursive_image_bundle(
        name="openimages_v7",
        root=root,
        transform=transform,
        train_candidates=("train", "images/train"),
        test_candidates=("validation", "val", "images/validation", "test"),
        val_fraction=val_fraction,
        test_fraction=test_fraction,
        seed=seed,
        download=download,
        download_fn=download_openimages_v7,
    )


def openimages_v7(
    root: str | Path = DATADIR,
    batch_size: int = 64,
    num_workers: int = 4,
    download: bool = False,
    image_size: int | tuple[int, int] | None = 224,
    val_fraction: float = 0.1,
    test_fraction: float = 0.1,
    seed: int = 42,
    mean: Sequence[float] = DEFAULT_RGB_MEAN,
    std: Sequence[float] = DEFAULT_RGB_STD,
):
    bundle = build_openimages_v7_datasets(
        root=root,
        download=download,
        image_size=image_size,
        val_fraction=val_fraction,
        test_fraction=test_fraction,
        seed=seed,
        mean=mean,
        std=std,
    )
    return build_dataloaders(bundle, batch_size=batch_size, num_workers=num_workers)
