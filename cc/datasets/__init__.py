from __future__ import annotations

import importlib

_EXPORTS = {
    "mnist": "cc.datasets.mnist:mnist",
    "cifar10": "cc.datasets.cifar:cifar10",
    "svhn": "cc.datasets.svhn:svhn",
    "fashion_mnist": "cc.datasets.fashion_mnist:fashion_mnist",
    "msmnist": "cc.datasets.msmnist:msmnist",
    "emnist": "cc.datasets.emnist:emnist",
    "kmnist": "cc.datasets.kmnist:kmnist",
    "moving_mnist": "cc.datasets.moving_mnist:moving_mnist",
    "stl10": "cc.datasets.stl10:stl10",
    "tiny_imagenet": "cc.datasets.tiny_imagenet:tiny_imagenet",
    "prepare_tiny_imagenet": "cc.datasets.tiny_imagenet:prepare_tiny_imagenet",
    "caltech101": "cc.datasets.caltech:caltech101",
    "caltech256": "cc.datasets.caltech:caltech256",
    "coco2017": "cc.datasets.coco:coco2017",
    "pascal_voc": "cc.datasets.pascal_voc:pascal_voc",
    "imagenet1k": "cc.datasets.imagenet:imagenet1k",
    "imagenet100": "cc.datasets.imagenet:imagenet100",
    "imagenette": "cc.datasets.imagenet:imagenette",
    "openimages_v7": "cc.datasets.openimages:openimages_v7",
    "pass_dataset": "cc.datasets.pass_dataset:pass_dataset",
    "list_datasets": "cc.datasets.registry:list_datasets",
    "get_dataloaders": "cc.datasets.registry:get_dataloaders",
    "get_dataset_bundle": "cc.datasets.registry:get_dataset_bundle",
    "download_dataset": "cc.datasets.registry:download_dataset",
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
