from __future__ import annotations

from pathlib import Path
from typing import Sequence

from coala import DATADIR
from coala.datasets.common import DEFAULT_RGB_MEAN, DEFAULT_RGB_STD, build_dataloaders, build_rgb_transform
from coala.datasets.common import download_and_extract, dataset_root
from coala.datasets.folder_based import imagefolder_bundle

IMAGENETTE_URL = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2.tgz"


def build_imagenet1k_datasets(
    root: str | Path = DATADIR,
    download: bool = False,
    image_size: int | tuple[int, int] | None = 224,
    val_fraction: float = 0.1,
    seed: int = 42,
    mean: Sequence[float] = DEFAULT_RGB_MEAN,
    std: Sequence[float] = DEFAULT_RGB_STD,
):
    del download
    transform = build_rgb_transform(image_size=image_size, mean=mean, std=std)
    return imagefolder_bundle(
        name="imagenet1k",
        root=root,
        transform=transform,
        train_candidates=("train",),
        test_candidates=("val", "validation"),
        val_fraction=val_fraction,
        seed=seed,
    )


def imagenet1k(
    root: str | Path = DATADIR,
    batch_size: int = 128,
    num_workers: int = 4,
    download: bool = False,
    image_size: int | tuple[int, int] | None = 224,
    val_fraction: float = 0.1,
    seed: int = 42,
    mean: Sequence[float] = DEFAULT_RGB_MEAN,
    std: Sequence[float] = DEFAULT_RGB_STD,
):
    bundle = build_imagenet1k_datasets(
        root=root,
        download=download,
        image_size=image_size,
        val_fraction=val_fraction,
        seed=seed,
        mean=mean,
        std=std,
    )
    return build_dataloaders(bundle, batch_size=batch_size, num_workers=num_workers)


def download_imagenet1k(root: str | Path = DATADIR) -> Path:
    data_root = dataset_root("imagenet1k", root=root)
    raise NotImplementedError(
        f"Automatic ImageNet-1k download is intentionally not implemented. "
        f"Place the extracted 'train/' and 'val/' folders under {data_root}."
    )


def build_imagenet100_datasets(
    root: str | Path = DATADIR,
    download: bool = False,
    image_size: int | tuple[int, int] | None = 224,
    val_fraction: float = 0.1,
    seed: int = 42,
    mean: Sequence[float] = DEFAULT_RGB_MEAN,
    std: Sequence[float] = DEFAULT_RGB_STD,
):
    del download
    transform = build_rgb_transform(image_size=image_size, mean=mean, std=std)
    return imagefolder_bundle(
        name="imagenet100",
        root=root,
        transform=transform,
        train_candidates=("train",),
        test_candidates=("val", "validation"),
        val_fraction=val_fraction,
        seed=seed,
    )


def imagenet100(
    root: str | Path = DATADIR,
    batch_size: int = 128,
    num_workers: int = 4,
    download: bool = False,
    image_size: int | tuple[int, int] | None = 224,
    val_fraction: float = 0.1,
    seed: int = 42,
    mean: Sequence[float] = DEFAULT_RGB_MEAN,
    std: Sequence[float] = DEFAULT_RGB_STD,
):
    bundle = build_imagenet100_datasets(
        root=root,
        download=download,
        image_size=image_size,
        val_fraction=val_fraction,
        seed=seed,
        mean=mean,
        std=std,
    )
    return build_dataloaders(bundle, batch_size=batch_size, num_workers=num_workers)


def download_imagenet100(root: str | Path = DATADIR) -> Path:
    data_root = dataset_root("imagenet100", root=root)
    raise NotImplementedError(
        f"Automatic ImageNet-100 download is intentionally not implemented. "
        f"Place the extracted 'train/' and 'val/' folders under {data_root}."
    )


def download_imagenette(root: str | Path = DATADIR, remove_archive: bool = False) -> Path:
    data_root = dataset_root("imagenette", root=root)
    download_and_extract(IMAGENETTE_URL, destination_dir=data_root, remove_archive=remove_archive)
    return data_root


def build_imagenette_datasets(
    root: str | Path = DATADIR,
    download: bool = False,
    image_size: int | tuple[int, int] | None = 224,
    val_fraction: float = 0.1,
    seed: int = 42,
    mean: Sequence[float] = DEFAULT_RGB_MEAN,
    std: Sequence[float] = DEFAULT_RGB_STD,
):
    transform = build_rgb_transform(image_size=image_size, mean=mean, std=std)
    return imagefolder_bundle(
        name="imagenette",
        root=root,
        transform=transform,
        train_candidates=("train",),
        test_candidates=("val",),
        val_fraction=val_fraction,
        seed=seed,
        download=download,
        download_fn=download_imagenette,
        dataset_subdirs=("imagenette2", "imagenette2-320", "imagenette2-160"),
    )


def imagenette(
    root: str | Path = DATADIR,
    batch_size: int = 128,
    num_workers: int = 4,
    download: bool = False,
    image_size: int | tuple[int, int] | None = 224,
    val_fraction: float = 0.1,
    seed: int = 42,
    mean: Sequence[float] = DEFAULT_RGB_MEAN,
    std: Sequence[float] = DEFAULT_RGB_STD,
):
    bundle = build_imagenette_datasets(
        root=root,
        download=download,
        image_size=image_size,
        val_fraction=val_fraction,
        seed=seed,
        mean=mean,
        std=std,
    )
    return build_dataloaders(bundle, batch_size=batch_size, num_workers=num_workers)
