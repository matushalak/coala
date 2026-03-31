from __future__ import annotations

from pathlib import Path
from typing import Sequence
import shutil

from cc import DATADIR
from cc.datasets.common import (
    DEFAULT_RGB_MEAN,
    DEFAULT_RGB_STD,
    DatasetBundle,
    build_dataloaders,
    build_rgb_transform,
    dataset_root,
    download_and_extract,
    split_dataset,
)

TINY_IMAGENET_URL = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"


def _tiny_imagenet_root(root: str | Path = DATADIR) -> tuple[Path, Path]:
    data_root = dataset_root("tiny_imagenet", root=root)
    return data_root, data_root / "tiny-imagenet-200"


def prepare_tiny_imagenet(root: str | Path = DATADIR, overwrite: bool = False) -> Path:
    data_root, dataset_dir = _tiny_imagenet_root(root=root)
    del data_root

    val_dir = dataset_dir / "val"
    images_dir = val_dir / "images"
    annotations_path = val_dir / "val_annotations.txt"
    prepared_dir = dataset_dir / "val_by_class"

    if prepared_dir.exists() and not overwrite and any(prepared_dir.iterdir()):
        return prepared_dir
    if overwrite and prepared_dir.exists():
        shutil.rmtree(prepared_dir)

    if not images_dir.exists() or not annotations_path.exists():
        raise FileNotFoundError(
            f"TinyImageNet validation files are missing under {val_dir}. "
            "Download and extract tiny-imagenet-200 first."
        )

    prepared_dir.mkdir(parents=True, exist_ok=True)
    with annotations_path.open("r", encoding="utf-8") as annotations_file:
        for line in annotations_file:
            image_name, class_name, *_ = line.strip().split("\t")
            class_dir = prepared_dir / class_name
            class_dir.mkdir(parents=True, exist_ok=True)
            source = images_dir / image_name
            destination = class_dir / image_name
            if not destination.exists():
                shutil.copy2(source, destination)

    return prepared_dir


def download_tiny_imagenet(root: str | Path = DATADIR, remove_archive: bool = False) -> Path:
    data_root, dataset_dir = _tiny_imagenet_root(root=root)
    if not dataset_dir.exists():
        download_and_extract(
            url=TINY_IMAGENET_URL,
            destination_dir=data_root,
            remove_archive=remove_archive,
        )
    prepare_tiny_imagenet(root=root)
    return dataset_dir


def build_tiny_imagenet_datasets(
    root: str | Path = DATADIR,
    download: bool = False,
    image_size: int | tuple[int, int] | None = 64,
    val_fraction: float = 0.1,
    seed: int = 42,
    mean: Sequence[float] = DEFAULT_RGB_MEAN,
    std: Sequence[float] = DEFAULT_RGB_STD,
) -> DatasetBundle:
    from torchvision.datasets import ImageFolder

    _, dataset_dir = _tiny_imagenet_root(root=root)
    if download and not dataset_dir.exists():
        download_tiny_imagenet(root=root)

    if not dataset_dir.exists():
        raise FileNotFoundError(
            f"TinyImageNet is missing under {dataset_dir}. Run download_tiny_imagenet(...) first."
        )

    transform = build_rgb_transform(image_size=image_size, mean=mean, std=std)
    prepared_val_dir = prepare_tiny_imagenet(root=root)

    train_full = ImageFolder(dataset_dir / "train", transform=transform)
    train_dataset, val_dataset = split_dataset(train_full, val_fraction=val_fraction, seed=seed)
    test_dataset = ImageFolder(prepared_val_dir, transform=transform)

    return DatasetBundle(
        name="tiny_imagenet",
        root=dataset_dir,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        metadata={
            "classes": len(train_full.classes),
            "class_names": tuple(train_full.classes),
            "prepared_val_dir": str(prepared_val_dir),
        },
    )


def tiny_imagenet(
    root: str | Path = DATADIR,
    batch_size: int = 128,
    num_workers: int = 4,
    download: bool = False,
    image_size: int | tuple[int, int] | None = 64,
    val_fraction: float = 0.1,
    seed: int = 42,
    mean: Sequence[float] = DEFAULT_RGB_MEAN,
    std: Sequence[float] = DEFAULT_RGB_STD,
):
    bundle = build_tiny_imagenet_datasets(
        root=root,
        download=download,
        image_size=image_size,
        val_fraction=val_fraction,
        seed=seed,
        mean=mean,
        std=std,
    )
    return build_dataloaders(bundle, batch_size=batch_size, num_workers=num_workers)
