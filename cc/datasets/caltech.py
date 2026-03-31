from __future__ import annotations

from pathlib import Path
from typing import Sequence

from cc import DATADIR
from cc.datasets.common import (
    DEFAULT_RGB_MEAN,
    DEFAULT_RGB_STD,
    DatasetBundle,
    build_dataloaders,
    build_rgb_transform,
    dataset_root,
    three_way_split,
)


def _build_caltech_datasets(
    dataset_cls_name: str,
    dataset_name: str,
    root: str | Path = DATADIR,
    download: bool = False,
    image_size: int | tuple[int, int] | None = 224,
    val_fraction: float = 0.1,
    test_fraction: float = 0.1,
    seed: int = 42,
    mean: Sequence[float] = DEFAULT_RGB_MEAN,
    std: Sequence[float] = DEFAULT_RGB_STD,
) -> DatasetBundle:
    import torchvision

    data_root = dataset_root(dataset_name, root=root)
    transform = build_rgb_transform(image_size=image_size, mean=mean, std=std)

    dataset_cls = getattr(torchvision.datasets, dataset_cls_name)
    dataset_kwargs = {
        "root": data_root,
        "transform": transform,
        "download": download,
    }
    if dataset_cls_name == "Caltech101":
        dataset_kwargs["target_type"] = "category"

    dataset = dataset_cls(**dataset_kwargs)
    train_dataset, val_dataset, test_dataset = three_way_split(
        dataset,
        val_fraction=val_fraction,
        test_fraction=test_fraction,
        seed=seed,
    )

    return DatasetBundle(
        name=dataset_name,
        root=data_root,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
    )


def _download_caltech(dataset_cls_name: str, dataset_name: str, root: str | Path = DATADIR) -> Path:
    import torchvision

    data_root = dataset_root(dataset_name, root=root)
    dataset_cls = getattr(torchvision.datasets, dataset_cls_name)
    dataset_kwargs = {"root": data_root, "download": True}
    if dataset_cls_name == "Caltech101":
        dataset_kwargs["target_type"] = "category"
    dataset_cls(**dataset_kwargs)
    return data_root


def build_caltech101_datasets(
    root: str | Path = DATADIR,
    download: bool = False,
    image_size: int | tuple[int, int] | None = 224,
    val_fraction: float = 0.1,
    test_fraction: float = 0.1,
    seed: int = 42,
    mean: Sequence[float] = DEFAULT_RGB_MEAN,
    std: Sequence[float] = DEFAULT_RGB_STD,
) -> DatasetBundle:
    return _build_caltech_datasets(
        dataset_cls_name="Caltech101",
        dataset_name="caltech101",
        root=root,
        download=download,
        image_size=image_size,
        val_fraction=val_fraction,
        test_fraction=test_fraction,
        seed=seed,
        mean=mean,
        std=std,
    )


def caltech101(
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
    bundle = build_caltech101_datasets(
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


def download_caltech101(root: str | Path = DATADIR) -> Path:
    return _download_caltech("Caltech101", "caltech101", root=root)


def build_caltech256_datasets(
    root: str | Path = DATADIR,
    download: bool = False,
    image_size: int | tuple[int, int] | None = 224,
    val_fraction: float = 0.1,
    test_fraction: float = 0.1,
    seed: int = 42,
    mean: Sequence[float] = DEFAULT_RGB_MEAN,
    std: Sequence[float] = DEFAULT_RGB_STD,
) -> DatasetBundle:
    return _build_caltech_datasets(
        dataset_cls_name="Caltech256",
        dataset_name="caltech256",
        root=root,
        download=download,
        image_size=image_size,
        val_fraction=val_fraction,
        test_fraction=test_fraction,
        seed=seed,
        mean=mean,
        std=std,
    )


def caltech256(
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
    bundle = build_caltech256_datasets(
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


def download_caltech256(root: str | Path = DATADIR) -> Path:
    return _download_caltech("Caltech256", "caltech256", root=root)
