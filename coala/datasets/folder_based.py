from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Sequence

from coala import DATADIR
from coala.datasets.common import (
    DatasetBundle,
    RecursiveImageDataset,
    dataset_root,
    split_dataset,
    three_way_split,
)


def _resolve_existing_path(base_dir: Path, candidates: Sequence[str | Path]) -> Path | None:
    for candidate in candidates:
        candidate_path = Path(candidate)
        if not candidate_path.is_absolute():
            candidate_path = base_dir / candidate_path
        if candidate_path.exists():
            return candidate_path
    return None


def _resolve_dataset_dir(dataset_home: Path, dataset_subdirs: Sequence[str | Path] | None) -> Path:
    if not dataset_subdirs:
        return dataset_home

    dataset_dir = _resolve_existing_path(dataset_home, dataset_subdirs)
    if dataset_dir is None:
        candidate_str = ", ".join(str(candidate) for candidate in dataset_subdirs)
        raise FileNotFoundError(f"Could not find any dataset directory under {dataset_home}: {candidate_str}")
    return dataset_dir


def imagefolder_bundle(
    name: str,
    root: str | Path = DATADIR,
    transform=None,
    train_candidates: Sequence[str | Path] = ("train",),
    test_candidates: Sequence[str | Path] = ("val", "validation"),
    val_fraction: float = 0.1,
    seed: int = 42,
    download: bool = False,
    download_fn: Callable[..., Path] | None = None,
    dataset_subdirs: Sequence[str | Path] | None = None,
    metadata: dict[str, Any] | None = None,
) -> DatasetBundle:
    from torchvision.datasets import ImageFolder

    dataset_home = dataset_root(name=name, root=root)
    if download and download_fn is not None:
        download_fn(root=root)

    dataset_dir = _resolve_dataset_dir(dataset_home=dataset_home, dataset_subdirs=dataset_subdirs)
    train_dir = _resolve_existing_path(dataset_dir, train_candidates)
    if train_dir is None:
        candidate_str = ", ".join(str(candidate) for candidate in train_candidates)
        raise FileNotFoundError(f"Could not find any train split under {dataset_dir}: {candidate_str}")

    train_full = ImageFolder(train_dir, transform=transform)
    train_dataset, val_dataset = split_dataset(train_full, val_fraction=val_fraction, seed=seed)

    test_dir = _resolve_existing_path(dataset_dir, test_candidates)
    if test_dir is None:
        test_dataset = val_dataset
    else:
        test_dataset = ImageFolder(test_dir, transform=transform)

    bundle_metadata = {
        "classes": len(train_full.classes),
        "class_names": tuple(train_full.classes),
    }
    if metadata is not None:
        bundle_metadata.update(metadata)

    return DatasetBundle(
        name=name,
        root=dataset_dir,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        metadata=bundle_metadata,
    )


def recursive_image_bundle(
    name: str,
    root: str | Path = DATADIR,
    transform=None,
    train_candidates: Sequence[str | Path] = ("train",),
    test_candidates: Sequence[str | Path] = ("val", "validation", "test"),
    val_fraction: float = 0.1,
    test_fraction: float = 0.1,
    seed: int = 42,
    download: bool = False,
    download_fn: Callable[..., Path] | None = None,
    dataset_subdirs: Sequence[str | Path] | None = None,
    fallback_to_dataset_root: bool = False,
    metadata: dict[str, Any] | None = None,
) -> DatasetBundle:
    dataset_home = dataset_root(name=name, root=root)
    if download and download_fn is not None:
        download_fn(root=root)

    dataset_dir = _resolve_dataset_dir(dataset_home=dataset_home, dataset_subdirs=dataset_subdirs)
    train_dir = _resolve_existing_path(dataset_dir, train_candidates)
    if train_dir is None:
        if fallback_to_dataset_root:
            train_dir = dataset_dir
        else:
            candidate_str = ", ".join(str(candidate) for candidate in train_candidates)
            raise FileNotFoundError(f"Could not find any train split under {dataset_dir}: {candidate_str}")

    test_dir = _resolve_existing_path(dataset_dir, test_candidates)
    train_full = RecursiveImageDataset(train_dir, transform=transform)

    if test_dir is None:
        train_dataset, val_dataset, test_dataset = three_way_split(
            train_full,
            val_fraction=val_fraction,
            test_fraction=test_fraction,
            seed=seed,
        )
    else:
        train_dataset, val_dataset = split_dataset(train_full, val_fraction=val_fraction, seed=seed)
        test_dataset = RecursiveImageDataset(test_dir, transform=transform)

    bundle_metadata = {"return_type": "image_path"}
    if metadata is not None:
        bundle_metadata.update(metadata)

    return DatasetBundle(
        name=name,
        root=dataset_dir,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        metadata=bundle_metadata,
    )
