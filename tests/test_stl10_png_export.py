from __future__ import annotations

import csv
from pathlib import Path
import sys

import numpy as np
from PIL import Image

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[1]))


def _sample(offset: int) -> np.ndarray:
    base = np.arange(3 * 96 * 96, dtype=np.uint32).reshape(3, 96, 96)
    return ((base + offset) % 256).astype(np.uint8)


class _DummySTL10:
    def __init__(self, root, split, transform=None, download=False):
        del root, download
        self.split = split
        self.transform = transform

        if split == "train":
            self.data = np.stack([_sample(0), _sample(17)], axis=0)
            self.labels = np.array([0, 9], dtype=np.int64)
        elif split == "test":
            self.data = np.stack([_sample(33)], axis=0)
            self.labels = np.array([3], dtype=np.int64)
        elif split == "unlabeled":
            self.data = np.stack([_sample(51)], axis=0)
            self.labels = None
        else:
            raise ValueError(split)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = Image.fromarray(self.data[idx].transpose(1, 2, 0))
        label = -1 if self.labels is None else int(self.labels[idx])
        if self.transform is not None:
            image = self.transform(image)
        return image, label


def test_export_stl10_png_bank_writes_pngs_manifest_and_viewer(monkeypatch, tmp_path):
    import torchvision

    from coala.datasets.stl10 import export_stl10_png_bank

    monkeypatch.setattr(torchvision.datasets, "STL10", _DummySTL10)

    export_dir = tmp_path / "stl10_browser"
    result = export_stl10_png_bank(
        root=tmp_path,
        output_dir=export_dir,
        include_unlabeled=True,
        write_viewer=True,
    )

    assert result.export_dir == export_dir
    assert result.total_images == 4
    assert result.split_counts == {"train": 2, "test": 1, "unlabeled": 1}
    assert result.metadata_path.exists()
    assert result.manifest_path.exists()
    assert result.viewer_path is not None
    assert result.viewer_path.exists()

    with result.metadata_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 4
    assert {row["split"] for row in rows} == {"train", "test", "unlabeled"}
    assert {row["label_name"] for row in rows} == {"airplane", "truck", "cat", "unlabeled"}

    for row in rows:
        image_path = export_dir / row["relative_path"]
        assert image_path.exists()
        with Image.open(image_path) as image:
            assert image.size == (96, 96)

    manifest_text = result.manifest_path.read_text(encoding="utf-8")
    viewer_text = result.viewer_path.read_text(encoding="utf-8")
    assert "window.STL10_VIEWER_MANIFEST" in manifest_text
    assert "STL-10 PNG Browser" in viewer_text
    assert "Cards per page" in viewer_text


def test_export_stl10_png_bank_limit_per_split(monkeypatch, tmp_path):
    import torchvision

    from coala.datasets.stl10 import export_stl10_png_bank

    monkeypatch.setattr(torchvision.datasets, "STL10", _DummySTL10)

    result = export_stl10_png_bank(
        root=tmp_path,
        output_dir=tmp_path / "limited",
        include_unlabeled=False,
        limit_per_split=1,
        write_viewer=False,
    )

    assert result.total_images == 2
    assert result.split_counts == {"train": 1, "test": 1}
    assert result.viewer_path is None
