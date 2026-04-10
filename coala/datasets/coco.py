from __future__ import annotations

from pathlib import Path
from typing import Sequence

from coala import DATADIR
from coala.datasets.common import DEFAULT_RGB_MEAN, DEFAULT_RGB_STD, build_dataloaders, build_rgb_transform
from coala.datasets.common import dataset_root, download_and_extract
from coala.datasets.folder_based import recursive_image_bundle

COCO_URLS = {
    "train2017": "http://images.cocodataset.org/zips/train2017.zip",
    "val2017": "http://images.cocodataset.org/zips/val2017.zip",
    "test2017": "http://images.cocodataset.org/zips/test2017.zip",
    "annotations": "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
}


def download_coco2017(
    root: str | Path = DATADIR,
    include_test: bool = False,
    include_annotations: bool = True,
    remove_archive: bool = False,
) -> Path:
    data_root = dataset_root("coco2017", root=root)
    for key in ("train2017", "val2017"):
        download_and_extract(COCO_URLS[key], destination_dir=data_root, remove_archive=remove_archive)
    if include_test:
        download_and_extract(COCO_URLS["test2017"], destination_dir=data_root, remove_archive=remove_archive)
    if include_annotations:
        download_and_extract(COCO_URLS["annotations"], destination_dir=data_root, remove_archive=remove_archive)
    return data_root


def build_coco2017_datasets(
    root: str | Path = DATADIR,
    download: bool = False,
    image_size: int | tuple[int, int] | None = 224,
    val_fraction: float = 0.1,
    seed: int = 42,
    mean: Sequence[float] = DEFAULT_RGB_MEAN,
    std: Sequence[float] = DEFAULT_RGB_STD,
):
    transform = build_rgb_transform(image_size=image_size, mean=mean, std=std)
    return recursive_image_bundle(
        name="coco2017",
        root=root,
        transform=transform,
        train_candidates=("train2017",),
        test_candidates=("val2017", "test2017"),
        val_fraction=val_fraction,
        seed=seed,
        download=download,
        download_fn=download_coco2017,
        metadata={"annotations_dir": str(dataset_root("coco2017", root=root) / "annotations")},
    )


def coco2017(
    root: str | Path = DATADIR,
    batch_size: int = 64,
    num_workers: int = 4,
    download: bool = False,
    image_size: int | tuple[int, int] | None = 224,
    val_fraction: float = 0.1,
    seed: int = 42,
    mean: Sequence[float] = DEFAULT_RGB_MEAN,
    std: Sequence[float] = DEFAULT_RGB_STD,
):
    bundle = build_coco2017_datasets(
        root=root,
        download=download,
        image_size=image_size,
        val_fraction=val_fraction,
        seed=seed,
        mean=mean,
        std=std,
    )
    return build_dataloaders(bundle, batch_size=batch_size, num_workers=num_workers)
