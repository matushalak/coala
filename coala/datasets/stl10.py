from __future__ import annotations

from dataclasses import dataclass
import csv
import json
from pathlib import Path
import shutil
from typing import Sequence

from PIL import Image
from torch.utils.data import ConcatDataset

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

STL10_CLASS_NAMES = (
    "airplane",
    "bird",
    "car",
    "cat",
    "deer",
    "dog",
    "horse",
    "monkey",
    "ship",
    "truck",
)


@dataclass(frozen=True)
class Stl10PngBankResult:
    export_dir: Path
    images_dir: Path
    metadata_path: Path
    manifest_path: Path
    viewer_path: Path | None
    split_counts: dict[str, int]
    total_images: int


def _stl10_splits(include_unlabeled: bool) -> tuple[str, ...]:
    if include_unlabeled:
        return ("train", "test", "unlabeled")
    return ("train", "test")


def _stl10_label_name(label: int) -> str:
    if 0 <= label < len(STL10_CLASS_NAMES):
        return STL10_CLASS_NAMES[label]
    return "unlabeled"


def _stl10_export_dir(root: str | Path = DATADIR, output_dir: str | Path | None = None) -> Path:
    if output_dir is not None:
        return Path(output_dir)
    return dataset_root("stl10", root=root) / "png_bank"


def _stl10_png_relative_path(split: str, label_name: str, index: int) -> Path:
    shard = f"chunk_{index // 1000:03d}"
    filename = f"{split}_{index:06d}.png"
    return Path("images") / split / label_name / shard / filename


def _dataset_label(dataset, index: int) -> int:
    labels = getattr(dataset, "labels", None)
    if labels is None:
        return -1
    return int(labels[index])


def _dataset_image(dataset, index: int) -> Image.Image:
    raw_data = getattr(dataset, "data", None)
    if raw_data is not None:
        sample = raw_data[index]
        if sample.ndim == 3 and sample.shape[0] in (1, 3):
            sample = sample.transpose(1, 2, 0)
        if sample.ndim == 3 and sample.shape[-1] == 1:
            sample = sample[..., 0]
        return Image.fromarray(sample)

    image, _ = dataset[index]
    if isinstance(image, Image.Image):
        return image
    return Image.fromarray(image)


def _write_stl10_metadata(metadata_path: Path, rows: list[dict[str, object]]) -> None:
    with metadata_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("split", "index", "label", "label_name", "relative_path"),
        )
        writer.writeheader()
        writer.writerows(rows)


def _write_stl10_manifest(manifest_path: Path, rows: list[dict[str, object]], split_counts: dict[str, int]) -> None:
    manifest = {
        "title": "STL-10 PNG Browser",
        "class_names": list(STL10_CLASS_NAMES),
        "split_counts": split_counts,
        "total_images": sum(split_counts.values()),
        "images": rows,
    }
    manifest_path.write_text(
        "window.STL10_VIEWER_MANIFEST = " + json.dumps(manifest, indent=2) + ";\n",
        encoding="utf-8",
    )


def _write_stl10_viewer(viewer_path: Path, manifest_name: str = "manifest.js") -> None:
    viewer_path.write_text(
        f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>STL-10 PNG Browser</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f5efe5;
      --panel: rgba(255, 252, 247, 0.92);
      --panel-border: rgba(60, 44, 24, 0.12);
      --text: #1e1c18;
      --muted: #6b655a;
      --accent: #a64527;
      --accent-soft: #f2d5c8;
      --shadow: 0 20px 50px rgba(72, 48, 20, 0.12);
      --radius: 20px;
    }}

    * {{
      box-sizing: border-box;
    }}

    body {{
      margin: 0;
      font-family: "Avenir Next", "Segoe UI", sans-serif;
      background:
        radial-gradient(circle at top left, rgba(255, 216, 179, 0.8), transparent 28%),
        radial-gradient(circle at top right, rgba(205, 232, 255, 0.75), transparent 24%),
        linear-gradient(180deg, #fcf7ef 0%, var(--bg) 100%);
      color: var(--text);
      min-height: 100vh;
    }}

    .shell {{
      width: min(1440px, calc(100vw - 32px));
      margin: 0 auto;
      padding: 28px 0 40px;
    }}

    .hero {{
      display: grid;
      gap: 18px;
      margin-bottom: 24px;
    }}

    .hero-card,
    .toolbar,
    .card {{
      background: var(--panel);
      border: 1px solid var(--panel-border);
      border-radius: var(--radius);
      box-shadow: var(--shadow);
      backdrop-filter: blur(16px);
    }}

    .hero-card {{
      padding: 28px;
    }}

    h1 {{
      margin: 0 0 8px;
      font-size: clamp(2rem, 4vw, 3.5rem);
      line-height: 0.95;
      letter-spacing: -0.04em;
    }}

    .subtitle {{
      margin: 0;
      max-width: 70ch;
      color: var(--muted);
      line-height: 1.55;
    }}

    .stats {{
      display: flex;
      flex-wrap: wrap;
      gap: 12px;
      margin-top: 20px;
    }}

    .stat {{
      min-width: 140px;
      padding: 14px 16px;
      border-radius: 16px;
      background: rgba(255, 255, 255, 0.7);
      border: 1px solid rgba(60, 44, 24, 0.08);
    }}

    .stat-value {{
      display: block;
      font-size: 1.35rem;
      font-weight: 700;
    }}

    .stat-label {{
      color: var(--muted);
      font-size: 0.9rem;
    }}

    .toolbar {{
      position: sticky;
      top: 14px;
      z-index: 5;
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 14px;
      padding: 18px;
      margin-bottom: 22px;
    }}

    .control {{
      display: grid;
      gap: 6px;
    }}

    label {{
      font-size: 0.82rem;
      font-weight: 700;
      color: var(--muted);
      letter-spacing: 0.04em;
      text-transform: uppercase;
    }}

    input,
    select,
    button {{
      width: 100%;
      border: 1px solid rgba(60, 44, 24, 0.14);
      border-radius: 14px;
      padding: 12px 14px;
      font: inherit;
      color: var(--text);
      background: rgba(255, 255, 255, 0.95);
    }}

    button {{
      cursor: pointer;
      background: var(--accent);
      border-color: var(--accent);
      color: #fff8f2;
      font-weight: 700;
      transition: transform 0.18s ease, box-shadow 0.18s ease;
    }}

    button:hover {{
      transform: translateY(-1px);
      box-shadow: 0 14px 24px rgba(166, 69, 39, 0.22);
    }}

    button:disabled {{
      cursor: default;
      opacity: 0.45;
      transform: none;
      box-shadow: none;
    }}

    .results-bar {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 12px;
      margin-bottom: 16px;
    }}

    .results-text {{
      color: var(--muted);
    }}

    .pager {{
      display: flex;
      align-items: center;
      gap: 8px;
    }}

    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
      gap: 16px;
    }}

    .card {{
      overflow: hidden;
    }}

    .thumb {{
      aspect-ratio: 1 / 1;
      background:
        linear-gradient(135deg, rgba(166, 69, 39, 0.16), rgba(205, 232, 255, 0.4));
      display: grid;
      place-items: center;
    }}

    .thumb img {{
      width: 100%;
      height: 100%;
      object-fit: cover;
      display: block;
    }}

    .meta {{
      padding: 14px;
      display: grid;
      gap: 6px;
    }}

    .badge-row {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
    }}

    .badge {{
      padding: 5px 9px;
      border-radius: 999px;
      font-size: 0.74rem;
      font-weight: 700;
      background: var(--accent-soft);
      color: var(--accent);
    }}

    .path {{
      color: var(--muted);
      font-size: 0.82rem;
      overflow-wrap: anywhere;
    }}

    .empty {{
      padding: 30px;
      text-align: center;
      color: var(--muted);
      background: rgba(255, 255, 255, 0.7);
      border-radius: var(--radius);
      border: 1px dashed rgba(60, 44, 24, 0.18);
    }}

    @media (max-width: 920px) {{
      .toolbar {{
        grid-template-columns: 1fr 1fr;
      }}
    }}

    @media (max-width: 640px) {{
      .shell {{
        width: min(100vw - 18px, 1440px);
        padding-top: 18px;
      }}

      .hero-card,
      .toolbar {{
        padding: 16px;
      }}

      .toolbar {{
        position: static;
        grid-template-columns: 1fr;
      }}

      .results-bar {{
        align-items: flex-start;
        flex-direction: column;
      }}
    }}
  </style>
</head>
<body>
  <div class="shell">
    <section class="hero">
      <div class="hero-card">
        <h1>STL-10 PNG Browser</h1>
        <p class="subtitle">
          Browse the exported STL-10 image bank without touching the original binary blobs.
          Filter by split, label, and filename, then page through the dataset in manageable slices.
        </p>
        <div class="stats" id="stats"></div>
      </div>
    </section>

    <section class="toolbar">
      <div class="control">
        <label for="search">Search</label>
        <input id="search" type="search" placeholder="Search split, label, or file name">
      </div>
      <div class="control">
        <label for="splitFilter">Split</label>
        <select id="splitFilter"></select>
      </div>
      <div class="control">
        <label for="labelFilter">Label</label>
        <select id="labelFilter"></select>
      </div>
      <div class="control">
        <label for="pageSize">Cards per page</label>
        <select id="pageSize">
          <option value="48">48</option>
          <option value="96" selected>96</option>
          <option value="192">192</option>
          <option value="384">384</option>
        </select>
      </div>
    </section>

    <div class="results-bar">
      <div class="results-text" id="resultsText"></div>
      <div class="pager">
        <button id="prevBtn" type="button">Previous</button>
        <span id="pageText"></span>
        <button id="nextBtn" type="button">Next</button>
      </div>
    </div>

    <div class="grid" id="grid"></div>
    <div class="empty" id="emptyState" hidden>No images match the current filters.</div>
  </div>

  <script src="{manifest_name}"></script>
  <script>
    const manifest = window.STL10_VIEWER_MANIFEST;
    const searchInput = document.getElementById("search");
    const splitFilter = document.getElementById("splitFilter");
    const labelFilter = document.getElementById("labelFilter");
    const pageSizeSelect = document.getElementById("pageSize");
    const grid = document.getElementById("grid");
    const resultsText = document.getElementById("resultsText");
    const pageText = document.getElementById("pageText");
    const prevBtn = document.getElementById("prevBtn");
    const nextBtn = document.getElementById("nextBtn");
    const stats = document.getElementById("stats");
    const emptyState = document.getElementById("emptyState");

    let filtered = manifest.images.slice();
    let page = 0;

    function addOption(select, value, label) {{
      const option = document.createElement("option");
      option.value = value;
      option.textContent = label;
      select.appendChild(option);
    }}

    function populateFilters() {{
      addOption(splitFilter, "all", "All splits");
      Object.entries(manifest.split_counts).forEach(([name, count]) => {{
        addOption(splitFilter, name, `${{name}} (${{count.toLocaleString()}})`);
      }});

      addOption(labelFilter, "all", "All labels");
      ["unlabeled", ...manifest.class_names].forEach((label) => {{
        const count = manifest.images.filter((item) => item.label_name === label).length;
        if (count > 0) {{
          addOption(labelFilter, label, `${{label}} (${{count.toLocaleString()}})`);
        }}
      }});
    }}

    function renderStats() {{
      const cards = [
        ["Total images", manifest.total_images.toLocaleString()],
        ["Train", (manifest.split_counts.train || 0).toLocaleString()],
        ["Test", (manifest.split_counts.test || 0).toLocaleString()],
        ["Unlabeled", (manifest.split_counts.unlabeled || 0).toLocaleString()],
      ];
      stats.innerHTML = "";
      cards.forEach(([label, value]) => {{
        const card = document.createElement("div");
        card.className = "stat";
        card.innerHTML = `<span class="stat-value">${{value}}</span><span class="stat-label">${{label}}</span>`;
        stats.appendChild(card);
      }});
    }}

    function applyFilters() {{
      const query = searchInput.value.trim().toLowerCase();
      const splitValue = splitFilter.value;
      const labelValue = labelFilter.value;

      filtered = manifest.images.filter((item) => {{
        if (splitValue !== "all" && item.split !== splitValue) {{
          return false;
        }}
        if (labelValue !== "all" && item.label_name !== labelValue) {{
          return false;
        }}
        if (!query) {{
          return true;
        }}
        const haystack = `${{item.split}} ${{item.label_name}} ${{item.relative_path}}`.toLowerCase();
        return haystack.includes(query);
      }});
      page = 0;
      renderPage();
    }}

    function renderPage() {{
      const pageSize = Number(pageSizeSelect.value);
      const pageCount = Math.max(1, Math.ceil(filtered.length / pageSize));
      page = Math.min(page, pageCount - 1);
      const start = page * pageSize;
      const end = start + pageSize;
      const visible = filtered.slice(start, end);

      grid.innerHTML = "";
      emptyState.hidden = visible.length !== 0;
      grid.hidden = visible.length === 0;

      visible.forEach((item) => {{
        const article = document.createElement("article");
        article.className = "card";
        article.innerHTML = `
          <div class="thumb">
            <img src="${{item.relative_path}}" alt="${{item.label_name}} image ${{item.index}}" loading="lazy">
          </div>
          <div class="meta">
            <div class="badge-row">
              <span class="badge">${{item.split}}</span>
              <span class="badge">${{item.label_name}}</span>
              <span class="badge">#${{item.index}}</span>
            </div>
            <div class="path">${{item.relative_path}}</div>
          </div>
        `;
        grid.appendChild(article);
      }});

      const shownEnd = visible.length === 0 ? 0 : end;
      resultsText.textContent =
        `Showing ${{visible.length === 0 ? 0 : start + 1}}-${{Math.min(shownEnd, filtered.length)}} of ${{filtered.length.toLocaleString()}} filtered images`;
      pageText.textContent = `Page ${{page + 1}} / ${{pageCount}}`;
      prevBtn.disabled = page === 0;
      nextBtn.disabled = page >= pageCount - 1;
    }}

    searchInput.addEventListener("input", applyFilters);
    splitFilter.addEventListener("change", applyFilters);
    labelFilter.addEventListener("change", applyFilters);
    pageSizeSelect.addEventListener("change", renderPage);
    prevBtn.addEventListener("click", () => {{
      if (page > 0) {{
        page -= 1;
        renderPage();
      }}
    }});
    nextBtn.addEventListener("click", () => {{
      const pageSize = Number(pageSizeSelect.value);
      const pageCount = Math.max(1, Math.ceil(filtered.length / pageSize));
      if (page < pageCount - 1) {{
        page += 1;
        renderPage();
      }}
    }});

    populateFilters();
    renderStats();
    renderPage();
  </script>
</body>
</html>
""",
        encoding="utf-8",
    )


def build_stl10_datasets(
    root: str | Path = DATADIR,
    download: bool = False,
    include_unlabeled: bool = True,
    image_size: int | tuple[int, int] | None = 96,
    val_fraction: float = 0.1,
    seed: int = 42,
    mean: Sequence[float] = DEFAULT_RGB_MEAN,
    std: Sequence[float] = DEFAULT_RGB_STD,
) -> DatasetBundle:
    import torchvision

    data_root = dataset_root("stl10", root=root)
    transform = build_rgb_transform(image_size=image_size, mean=mean, std=std)

    labeled_train = torchvision.datasets.STL10(
        root=data_root,
        split="train",
        transform=transform,
        download=download,
    )
    test_dataset = torchvision.datasets.STL10(
        root=data_root,
        split="test",
        transform=transform,
        download=download,
    )

    train_dataset, val_dataset = split_dataset(labeled_train, val_fraction=val_fraction, seed=seed)
    metadata = {"include_unlabeled": include_unlabeled}

    if include_unlabeled:
        unlabeled_dataset = torchvision.datasets.STL10(
            root=data_root,
            split="unlabeled",
            transform=transform,
            download=download,
        )
        train_dataset = ConcatDataset([train_dataset, unlabeled_dataset])
        metadata["unlabeled_examples"] = len(unlabeled_dataset)

    return DatasetBundle(
        name="stl10",
        root=data_root,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        metadata=metadata,
    )


def stl10(
    root: str | Path = DATADIR,
    batch_size: int = 128,
    num_workers: int = 4,
    download: bool = False,
    include_unlabeled: bool = True,
    image_size: int | tuple[int, int] | None = 96,
    val_fraction: float = 0.1,
    seed: int = 42,
    mean: Sequence[float] = DEFAULT_RGB_MEAN,
    std: Sequence[float] = DEFAULT_RGB_STD,
):
    bundle = build_stl10_datasets(
        root=root,
        download=download,
        include_unlabeled=include_unlabeled,
        image_size=image_size,
        val_fraction=val_fraction,
        seed=seed,
        mean=mean,
        std=std,
    )
    return build_dataloaders(bundle, batch_size=batch_size, num_workers=num_workers)


def download_stl10(root: str | Path = DATADIR, include_unlabeled: bool = True) -> Path:
    import torchvision

    data_root = dataset_root("stl10", root=root)
    for split in _stl10_splits(include_unlabeled):
        torchvision.datasets.STL10(root=data_root, split=split, download=True)
    return data_root


def export_stl10_png_bank(
    root: str | Path = DATADIR,
    output_dir: str | Path | None = None,
    download: bool = False,
    include_unlabeled: bool = True,
    overwrite: bool = False,
    limit_per_split: int | None = None,
    write_viewer: bool = True,
) -> Stl10PngBankResult:
    import torchvision

    data_root = dataset_root("stl10", root=root)
    export_dir = _stl10_export_dir(root=root, output_dir=output_dir)
    images_dir = export_dir / "images"
    metadata_path = export_dir / "metadata.csv"
    manifest_path = export_dir / "manifest.js"
    viewer_path = export_dir / "index.html" if write_viewer else None

    if overwrite and export_dir.exists():
        shutil.rmtree(export_dir)

    images_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    split_counts: dict[str, int] = {}

    for split in _stl10_splits(include_unlabeled):
        dataset = torchvision.datasets.STL10(
            root=data_root,
            split=split,
            transform=None,
            download=download,
        )
        count = len(dataset) if limit_per_split is None else min(len(dataset), limit_per_split)
        split_counts[split] = count

        for index in range(count):
            label = _dataset_label(dataset, index)
            label_name = _stl10_label_name(label)
            relative_path = _stl10_png_relative_path(split=split, label_name=label_name, index=index)
            destination = export_dir / relative_path
            destination.parent.mkdir(parents=True, exist_ok=True)
            if not destination.exists():
                image = _dataset_image(dataset, index).convert("RGB")
                image.save(destination, format="PNG")
            rows.append(
                {
                    "split": split,
                    "index": index,
                    "label": label,
                    "label_name": label_name,
                    "relative_path": str(relative_path).replace("\\", "/"),
                }
            )

    _write_stl10_metadata(metadata_path=metadata_path, rows=rows)
    _write_stl10_manifest(manifest_path=manifest_path, rows=rows, split_counts=split_counts)
    if viewer_path is not None:
        _write_stl10_viewer(viewer_path=viewer_path, manifest_name=manifest_path.name)

    return Stl10PngBankResult(
        export_dir=export_dir,
        images_dir=images_dir,
        metadata_path=metadata_path,
        manifest_path=manifest_path,
        viewer_path=viewer_path,
        split_counts=split_counts,
        total_images=sum(split_counts.values()),
    )


if __name__ == "__main__":
    download_stl10()
