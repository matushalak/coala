from __future__ import annotations

from dataclasses import dataclass
import importlib
import re


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    loaders_path: str
    bundle_path: str | None = None
    download_path: str | None = None
    download_mode: str = "none"
    aliases: tuple[str, ...] = ()
    description: str = ""


def _load_callable(callable_path: str):
    module_path, attr_name = callable_path.split(":")
    module = importlib.import_module(module_path)
    return getattr(module, attr_name)


def _normalize_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


DATASET_SPECS: dict[str, DatasetSpec] = {
    "mnist": DatasetSpec(
        name="mnist",
        loaders_path="cc.datasets.mnist:mnist",
        aliases=("digit_mnist",),
        description="Standard handwritten digits from torchvision.",
    ),
    "fashion_mnist": DatasetSpec(
        name="fashion_mnist",
        loaders_path="cc.datasets.fashion_mnist:fashion_mnist",
        aliases=("fashion-mnist",),
        description="Fashion-MNIST from torchvision.",
    ),
    "cifar10": DatasetSpec(
        name="cifar10",
        loaders_path="cc.datasets.cifar:cifar10",
        aliases=("cifar-10",),
        description="CIFAR-10 from torchvision.",
    ),
    "svhn": DatasetSpec(
        name="svhn",
        loaders_path="cc.datasets.svhn:svhn",
        description="Street View House Numbers from torchvision.",
    ),
    "msmnist": DatasetSpec(
        name="msmnist",
        loaders_path="cc.datasets.msmnist:msmnist",
        description="Masked sequential MNIST used in the current project.",
    ),
    "emnist": DatasetSpec(
        name="emnist",
        loaders_path="cc.datasets.emnist:emnist",
        bundle_path="cc.datasets.emnist:build_emnist_datasets",
        download_path="cc.datasets.emnist:download_emnist",
        download_mode="auto",
        description="Extended handwritten characters from torchvision.",
    ),
    "kmnist": DatasetSpec(
        name="kmnist",
        loaders_path="cc.datasets.kmnist:kmnist",
        bundle_path="cc.datasets.kmnist:build_kmnist_datasets",
        download_path="cc.datasets.kmnist:download_kmnist",
        download_mode="auto",
        description="Kuzushiji-MNIST from torchvision.",
    ),
    "moving_mnist": DatasetSpec(
        name="moving_mnist",
        loaders_path="cc.datasets.moving_mnist:moving_mnist",
        bundle_path="cc.datasets.moving_mnist:build_moving_mnist_datasets",
        download_path="cc.datasets.moving_mnist:download_moving_mnist",
        download_mode="auto",
        aliases=("movingmnist",),
        description="Sequence dataset backed by the classic MovingMNIST .npy file.",
    ),
    "stl10": DatasetSpec(
        name="stl10",
        loaders_path="cc.datasets.stl10:stl10",
        bundle_path="cc.datasets.stl10:build_stl10_datasets",
        download_path="cc.datasets.stl10:download_stl10",
        download_mode="auto",
        aliases=("stl-10",),
        description="STL-10 with optional unlabeled split merged into training.",
    ),
    "tiny_imagenet": DatasetSpec(
        name="tiny_imagenet",
        loaders_path="cc.datasets.tiny_imagenet:tiny_imagenet",
        bundle_path="cc.datasets.tiny_imagenet:build_tiny_imagenet_datasets",
        download_path="cc.datasets.tiny_imagenet:download_tiny_imagenet",
        download_mode="auto",
        aliases=("tinyimagenet", "tiny-imagenet"),
        description="TinyImageNet with prepared validation folders for ImageFolder access.",
    ),
    "caltech101": DatasetSpec(
        name="caltech101",
        loaders_path="cc.datasets.caltech:caltech101",
        bundle_path="cc.datasets.caltech:build_caltech101_datasets",
        download_path="cc.datasets.caltech:download_caltech101",
        download_mode="auto",
        aliases=("caltech-101",),
        description="Caltech-101 with random train/val/test splits.",
    ),
    "caltech256": DatasetSpec(
        name="caltech256",
        loaders_path="cc.datasets.caltech:caltech256",
        bundle_path="cc.datasets.caltech:build_caltech256_datasets",
        download_path="cc.datasets.caltech:download_caltech256",
        download_mode="auto",
        aliases=("caltech-256",),
        description="Caltech-256 with random train/val/test splits.",
    ),
    "coco2017": DatasetSpec(
        name="coco2017",
        loaders_path="cc.datasets.coco:coco2017",
        bundle_path="cc.datasets.coco:build_coco2017_datasets",
        download_path="cc.datasets.coco:download_coco2017",
        download_mode="auto",
        aliases=("coco-2017", "coco"),
        description="COCO 2017 images wired for SSL-style image access.",
    ),
    "pascal_voc": DatasetSpec(
        name="pascal_voc",
        loaders_path="cc.datasets.pascal_voc:pascal_voc",
        bundle_path="cc.datasets.pascal_voc:build_pascal_voc_datasets",
        download_path="cc.datasets.pascal_voc:download_pascal_voc",
        download_mode="auto",
        aliases=("pascal-voc", "voc", "voc2012"),
        description="PASCAL VOC image loaders using official split files.",
    ),
    "imagenet1k": DatasetSpec(
        name="imagenet1k",
        loaders_path="cc.datasets.imagenet:imagenet1k",
        bundle_path="cc.datasets.imagenet:build_imagenet1k_datasets",
        download_path="cc.datasets.imagenet:download_imagenet1k",
        download_mode="manual",
        aliases=("imagenet-1k", "imagenet"),
        description="ImageNet-1k from local train/val folders.",
    ),
    "imagenet100": DatasetSpec(
        name="imagenet100",
        loaders_path="cc.datasets.imagenet:imagenet100",
        bundle_path="cc.datasets.imagenet:build_imagenet100_datasets",
        download_path="cc.datasets.imagenet:download_imagenet100",
        download_mode="manual",
        aliases=("imagenet-100",),
        description="ImageNet-100 from local train/val folders.",
    ),
    "imagenette": DatasetSpec(
        name="imagenette",
        loaders_path="cc.datasets.imagenet:imagenette",
        bundle_path="cc.datasets.imagenet:build_imagenette_datasets",
        download_path="cc.datasets.imagenet:download_imagenette",
        download_mode="auto",
        description="Imagenette with an included public download helper.",
    ),
    "openimages_v7": DatasetSpec(
        name="openimages_v7",
        loaders_path="cc.datasets.openimages:openimages_v7",
        bundle_path="cc.datasets.openimages:build_openimages_v7_datasets",
        download_path="cc.datasets.openimages:download_openimages_v7",
        download_mode="manual",
        aliases=("open-images-v7", "openimages", "openimages-v7"),
        description="OpenImages V7 from extracted image folders.",
    ),
    "pass": DatasetSpec(
        name="pass",
        loaders_path="cc.datasets.pass_dataset:pass_dataset",
        bundle_path="cc.datasets.pass_dataset:build_pass_datasets",
        download_path="cc.datasets.pass_dataset:download_pass",
        download_mode="manual",
        aliases=("pass_dataset",),
        description="PASS from extracted image folders.",
    ),
}

_ALIASES = {
    _normalize_name(alias): spec.name
    for spec in DATASET_SPECS.values()
    for alias in (spec.name, *spec.aliases)
}


def resolve_dataset_name(name: str) -> str:
    normalized = _normalize_name(name)
    if normalized not in _ALIASES:
        available = ", ".join(sorted(DATASET_SPECS))
        raise KeyError(f"Unknown dataset '{name}'. Available datasets: {available}")
    return _ALIASES[normalized]


def get_dataset_spec(name: str) -> DatasetSpec:
    return DATASET_SPECS[resolve_dataset_name(name)]


def list_datasets() -> tuple[DatasetSpec, ...]:
    return tuple(DATASET_SPECS[name] for name in sorted(DATASET_SPECS))


def get_dataloaders(name: str, **kwargs):
    spec = get_dataset_spec(name)
    builder = _load_callable(spec.loaders_path)
    return builder(**kwargs)


def get_dataset_bundle(name: str, **kwargs):
    spec = get_dataset_spec(name)
    if spec.bundle_path is None:
        raise NotImplementedError(f"{spec.name} does not expose a dataset bundle builder.")
    builder = _load_callable(spec.bundle_path)
    return builder(**kwargs)


def download_dataset(name: str, **kwargs):
    spec = get_dataset_spec(name)
    if spec.download_path is None:
        raise NotImplementedError(f"{spec.name} does not expose a downloader.")
    downloader = _load_callable(spec.download_path)
    return downloader(**kwargs)
