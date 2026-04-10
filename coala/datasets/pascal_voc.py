from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch.utils.data as data
from PIL import Image

from coala import DATADIR
from coala.datasets.common import (
    DEFAULT_RGB_MEAN,
    DEFAULT_RGB_STD,
    DatasetBundle,
    build_dataloaders,
    build_rgb_transform,
    dataset_root,
    split_dataset,
)


class VOCImageDataset(data.Dataset):
    def __init__(
        self,
        root: str | Path,
        split: str,
        year: str = "2012",
        transform=None,
    ):
        self.dataset_dir = Path(root) / "VOCdevkit" / f"VOC{year}"
        self.transform = transform
        if not self.dataset_dir.exists():
            raise FileNotFoundError(f"PASCAL VOC root not found: {self.dataset_dir}")

        split_candidates = (
            self.dataset_dir / "ImageSets" / "Main" / f"{split}.txt",
            self.dataset_dir / "ImageSets" / "Segmentation" / f"{split}.txt",
            self.dataset_dir / "ImageSets" / "Layout" / f"{split}.txt",
        )
        split_path = next((path for path in split_candidates if path.exists()), None)
        if split_path is None:
            raise FileNotFoundError(f"Could not find a split file for '{split}' in {self.dataset_dir}")

        self.image_ids = []
        with split_path.open("r", encoding="utf-8") as split_file:
            for line in split_file:
                image_id = line.strip().split()[0]
                if image_id:
                    self.image_ids.append(image_id)
        if not self.image_ids:
            raise ValueError(f"No image ids found in {split_path}")

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, idx: int):
        image_id = self.image_ids[idx]
        image_path = self.dataset_dir / "JPEGImages" / f"{image_id}.jpg"
        if not image_path.exists():
            image_path = self.dataset_dir / "JPEGImages" / f"{image_id}.jpeg"
        with Image.open(image_path) as image:
            image = image.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, image_id


def download_pascal_voc(root: str | Path = DATADIR, year: str = "2012") -> Path:
    import torchvision

    data_root = dataset_root("pascal_voc", root=root)
    torchvision.datasets.VOCSegmentation(
        root=data_root,
        year=year,
        image_set="train",
        download=True,
    )
    return data_root


def build_pascal_voc_datasets(
    root: str | Path = DATADIR,
    download: bool = False,
    year: str = "2012",
    image_size: int | tuple[int, int] | None = 224,
    val_fraction: float = 0.1,
    seed: int = 42,
    mean: Sequence[float] = DEFAULT_RGB_MEAN,
    std: Sequence[float] = DEFAULT_RGB_STD,
) -> DatasetBundle:
    data_root = dataset_root("pascal_voc", root=root)
    if download:
        download_pascal_voc(root=root, year=year)

    transform = build_rgb_transform(image_size=image_size, mean=mean, std=std)
    train_full = VOCImageDataset(root=data_root, split="train", year=year, transform=transform)
    train_dataset, val_dataset = split_dataset(train_full, val_fraction=val_fraction, seed=seed)
    try:
        test_dataset = VOCImageDataset(root=data_root, split="val", year=year, transform=transform)
    except FileNotFoundError:
        test_dataset = val_dataset

    return DatasetBundle(
        name="pascal_voc",
        root=data_root,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        metadata={"year": year},
    )


def pascal_voc(
    root: str | Path = DATADIR,
    batch_size: int = 64,
    num_workers: int = 4,
    download: bool = False,
    year: str = "2012",
    image_size: int | tuple[int, int] | None = 224,
    val_fraction: float = 0.1,
    seed: int = 42,
    mean: Sequence[float] = DEFAULT_RGB_MEAN,
    std: Sequence[float] = DEFAULT_RGB_STD,
):
    bundle = build_pascal_voc_datasets(
        root=root,
        download=download,
        year=year,
        image_size=image_size,
        val_fraction=val_fraction,
        seed=seed,
        mean=mean,
        std=std,
    )
    return build_dataloaders(bundle, batch_size=batch_size, num_workers=num_workers)
