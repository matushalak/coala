from __future__ import annotations

from pathlib import Path
import sys

# Allow direct script execution from inside this folder: `python test_dataset_registry.py`.
if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[3]))


def test_registry_aliases_and_listing():
    from cc.datasets.registry import list_datasets, resolve_dataset_name

    dataset_names = {spec.name for spec in list_datasets()}

    assert "stl10" in dataset_names
    assert "tiny_imagenet" in dataset_names
    assert "coco2017" in dataset_names

    assert resolve_dataset_name("stl-10") == "stl10"
    assert resolve_dataset_name("tinyimagenet") == "tiny_imagenet"
    assert resolve_dataset_name("voc") == "pascal_voc"


if __name__ == "__main__":
    test_registry_aliases_and_listing()
    print("Dataset registry tests passed.")
