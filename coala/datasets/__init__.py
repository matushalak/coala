from __future__ import annotations

import importlib

_EXPORTS = {
    "mnist": "coala.datasets.mnist:mnist",
    "cifar10": "coala.datasets.cifar:cifar10",
    "svhn": "coala.datasets.svhn:svhn",
    "fashion_mnist": "coala.datasets.fashion_mnist:fashion_mnist",
    "msmnist": "coala.datasets.msmnist:msmnist",
    "emnist": "coala.datasets.emnist:emnist",
    "kmnist": "coala.datasets.kmnist:kmnist",
    "moving_mnist": "coala.datasets.moving_mnist:moving_mnist",
    "stl10": "coala.datasets.stl10:stl10",
    "tiny_imagenet": "coala.datasets.tiny_imagenet:tiny_imagenet",
    "prepare_tiny_imagenet": "coala.datasets.tiny_imagenet:prepare_tiny_imagenet",
    "caltech101": "coala.datasets.caltech:caltech101",
    "caltech256": "coala.datasets.caltech:caltech256",
    "coco2017": "coala.datasets.coco:coco2017",
    "pascal_voc": "coala.datasets.pascal_voc:pascal_voc",
    "imagenet1k": "coala.datasets.imagenet:imagenet1k",
    "imagenet100": "coala.datasets.imagenet:imagenet100",
    "imagenette": "coala.datasets.imagenet:imagenette",
    "openimages_v7": "coala.datasets.openimages:openimages_v7",
    "pass_dataset": "coala.datasets.pass_dataset:pass_dataset",
    "list_datasets": "coala.datasets.registry:list_datasets",
    "get_dataloaders": "coala.datasets.registry:get_dataloaders",
    "get_dataset_bundle": "coala.datasets.registry:get_dataset_bundle",
    "download_dataset": "coala.datasets.registry:download_dataset",
}

__all__ = list(_EXPORTS)


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_path, attr_name = _EXPORTS[name].split(":")
    module = importlib.import_module(module_path)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
