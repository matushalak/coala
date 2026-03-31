from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence
import re
import shutil
import tarfile
import urllib.request
import zipfile

import torch
import torch.utils.data as data
from PIL import Image

from cc import DATADIR

DEFAULT_RGB_MEAN = (0.5, 0.5, 0.5)
DEFAULT_RGB_STD = (0.5, 0.5, 0.5)
DEFAULT_GRAY_MEAN = (0.5,)
DEFAULT_GRAY_STD = (0.5,)
IMAGE_EXTENSIONS = {
    ".bmp",
    ".jpeg",
    ".jpg",
    ".png",
    ".ppm",
    ".tif",
    ".tiff",
    ".webp",
}


@dataclass(frozen=True)
class DatasetBundle:
    name: str
    root: Path
    train_dataset: data.Dataset
    val_dataset: data.Dataset
    test_dataset: data.Dataset
    metadata: dict[str, Any] = field(default_factory=dict)


def dataset_root(name: str, root: str | Path = DATADIR) -> Path:
    slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    path = Path(root) / slug
    path.mkdir(parents=True, exist_ok=True)
    return path


def split_dataset(
    dataset: data.Dataset,
    val_fraction: float | int = 0.1,
    seed: int = 42,
) -> tuple[data.Dataset, data.Dataset]:
    dataset_size = len(dataset)
    if dataset_size < 2:
        raise ValueError("Need at least two samples to create train/val splits.")

    if isinstance(val_fraction, int):
        val_len = val_fraction
    else:
        if not 0.0 < val_fraction < 1.0:
            raise ValueError("val_fraction must be between 0 and 1.")
        val_len = max(1, int(round(dataset_size * val_fraction)))

    val_len = min(val_len, dataset_size - 1)
    train_len = dataset_size - val_len
    return data.random_split(
        dataset,
        lengths=[train_len, val_len],
        generator=torch.Generator().manual_seed(seed),
    )


def three_way_split(
    dataset: data.Dataset,
    val_fraction: float = 0.1,
    test_fraction: float = 0.1,
    seed: int = 42,
) -> tuple[data.Dataset, data.Dataset, data.Dataset]:
    dataset_size = len(dataset)
    if dataset_size < 3:
        raise ValueError("Need at least three samples to create train/val/test splits.")
    if val_fraction <= 0.0 or test_fraction <= 0.0:
        raise ValueError("val_fraction and test_fraction must be positive.")
    if val_fraction + test_fraction >= 1.0:
        raise ValueError("val_fraction + test_fraction must be smaller than 1.")

    val_len = max(1, int(round(dataset_size * val_fraction)))
    test_len = max(1, int(round(dataset_size * test_fraction)))
    if val_len + test_len >= dataset_size:
        raise ValueError("Split fractions leave no samples for training.")

    train_len = dataset_size - val_len - test_len
    return data.random_split(
        dataset,
        lengths=[train_len, val_len, test_len],
        generator=torch.Generator().manual_seed(seed),
    )


def build_rgb_transform(
    image_size: int | tuple[int, int] | None = None,
    mean: Sequence[float] = DEFAULT_RGB_MEAN,
    std: Sequence[float] = DEFAULT_RGB_STD,
):
    from torchvision import transforms

    ops = []
    if image_size is not None:
        resize_size = (image_size, image_size) if isinstance(image_size, int) else tuple(image_size)
        ops.append(transforms.Resize(resize_size))
    ops.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=tuple(mean), std=tuple(std)),
        ]
    )
    return transforms.Compose(ops)


def build_grayscale_transform(
    image_size: int | tuple[int, int] | None = None,
    mean: Sequence[float] = DEFAULT_GRAY_MEAN,
    std: Sequence[float] = DEFAULT_GRAY_STD,
):
    from torchvision import transforms

    ops = []
    if image_size is not None:
        resize_size = (image_size, image_size) if isinstance(image_size, int) else tuple(image_size)
        ops.append(transforms.Resize(resize_size))
    ops.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=tuple(mean), std=tuple(std)),
        ]
    )
    return transforms.Compose(ops)


def build_dataloaders(
    bundle: DatasetBundle,
    batch_size: int = 128,
    num_workers: int = 4,
    pin_memory: bool = True,
    eval_num_workers: int = 0,
) -> tuple[data.DataLoader, data.DataLoader, data.DataLoader]:
    train_loader = data.DataLoader(
        bundle.train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
    )
    val_loader = data.DataLoader(
        bundle.val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=eval_num_workers,
        drop_last=False,
    )
    test_loader = data.DataLoader(
        bundle.test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=eval_num_workers,
        drop_last=False,
    )
    return train_loader, val_loader, test_loader


def download_url(url: str, destination_dir: str | Path, filename: str | None = None) -> Path:
    destination_dir = Path(destination_dir)
    destination_dir.mkdir(parents=True, exist_ok=True)
    archive_name = filename or Path(url).name
    archive_path = destination_dir / archive_name

    with urllib.request.urlopen(url) as response, archive_path.open("wb") as output_file:
        shutil.copyfileobj(response, output_file)
    return archive_path


def extract_archive(archive_path: str | Path, destination_dir: str | Path) -> Path:
    archive_path = Path(archive_path)
    destination_dir = Path(destination_dir)
    destination_dir.mkdir(parents=True, exist_ok=True)

    if zipfile.is_zipfile(archive_path):
        with zipfile.ZipFile(archive_path) as archive:
            archive.extractall(destination_dir)
        return destination_dir

    if tarfile.is_tarfile(archive_path):
        with tarfile.open(archive_path) as archive:
            archive.extractall(destination_dir)
        return destination_dir

    raise ValueError(f"Unsupported archive format: {archive_path}")


def download_and_extract(
    url: str,
    destination_dir: str | Path,
    filename: str | None = None,
    remove_archive: bool = False,
) -> Path:
    archive_path = download_url(url=url, destination_dir=destination_dir, filename=filename)
    extract_archive(archive_path=archive_path, destination_dir=destination_dir)
    if remove_archive and archive_path.exists():
        archive_path.unlink()
    return Path(destination_dir)


class RecursiveImageDataset(data.Dataset):
    def __init__(self, root: str | Path, transform=None):
        self.root = Path(root)
        if not self.root.exists():
            raise FileNotFoundError(f"Image directory does not exist: {self.root}")

        self.transform = transform
        self.samples = sorted(
            path
            for path in self.root.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not self.samples:
            raise FileNotFoundError(f"No image files found under {self.root}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        image_path = self.samples[idx]
        with Image.open(image_path) as image:
            image = image.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        target = str(image_path.relative_to(self.root))
        return image, target
