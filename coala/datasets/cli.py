from __future__ import annotations

import argparse
import ast

from coala.datasets.registry import download_dataset, get_dataloaders, list_datasets
from coala.datasets.stl10 import export_stl10_png_bank


def _coerce_value(raw_value: str):
    lowered = raw_value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered == "none":
        return None
    try:
        return ast.literal_eval(raw_value)
    except (ValueError, SyntaxError):
        return raw_value


def _parse_key_values(items: list[str]) -> dict[str, object]:
    parsed: dict[str, object] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Expected key=value, got '{item}'")
        key, raw_value = item.split("=", maxsplit=1)
        parsed[key.replace("-", "_")] = _coerce_value(raw_value)
    return parsed


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Dataset registry CLI for coala.datasets.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("list", help="List registered datasets.")

    download_parser = subparsers.add_parser("download", help="Run a dataset download helper.")
    download_parser.add_argument("dataset", help="Canonical dataset name or alias.")
    download_parser.add_argument("--arg", action="append", default=[], help="Extra key=value args.")

    inspect_parser = subparsers.add_parser("inspect", help="Construct loaders and print basic info.")
    inspect_parser.add_argument("dataset", help="Canonical dataset name or alias.")
    inspect_parser.add_argument("--batch-size", type=int, default=8)
    inspect_parser.add_argument("--num-workers", type=int, default=0)
    inspect_parser.add_argument("--download", action="store_true")
    inspect_parser.add_argument("--arg", action="append", default=[], help="Extra key=value args.")

    export_stl10_parser = subparsers.add_parser(
        "export-stl10-pngs",
        help="Export STL-10 images as PNGs and generate a static HTML browser.",
    )
    export_stl10_parser.add_argument("--root", default=None, help="Dataset root. Defaults to coala.DATADIR.")
    export_stl10_parser.add_argument(
        "--output-dir",
        default=None,
        help="Export directory. Defaults to <root>/stl10/png_bank.",
    )
    export_stl10_parser.add_argument("--download", action="store_true")
    export_stl10_parser.add_argument(
        "--include-unlabeled",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include the 100k unlabeled STL-10 split in the export.",
    )
    export_stl10_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete and recreate the export directory before writing files.",
    )
    export_stl10_parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on the number of exported images per split.",
    )
    export_stl10_parser.add_argument(
        "--viewer",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write index.html alongside the PNG bank.",
    )

    return parser


def _format_batch_shape(batch) -> str:
    if isinstance(batch, (tuple, list)) and batch:
        first_item = batch[0]
        if hasattr(first_item, "shape"):
            return str(tuple(first_item.shape))
    return type(batch).__name__


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "list":
        for spec in list_datasets():
            aliases = ", ".join(spec.aliases) if spec.aliases else "-"
            print(
                f"{spec.name:16} aliases={aliases:24} "
                f"download={spec.download_mode:6}  {spec.description}"
            )
        return 0

    if args.command == "download":
        extra_kwargs = _parse_key_values(args.arg)
        download_result = download_dataset(args.dataset, **extra_kwargs)
        print(download_result)
        return 0

    if args.command == "export-stl10-pngs":
        result = export_stl10_png_bank(
            root=args.root,
            output_dir=args.output_dir,
            download=args.download,
            include_unlabeled=args.include_unlabeled,
            overwrite=args.overwrite,
            limit_per_split=args.limit,
            write_viewer=args.viewer,
        )
        print(f"export_dir={result.export_dir}")
        print(f"total_images={result.total_images}")
        for split_name, count in result.split_counts.items():
            print(f"{split_name}={count}")
        print(f"metadata={result.metadata_path}")
        print(f"manifest={result.manifest_path}")
        if result.viewer_path is not None:
            print(f"viewer={result.viewer_path}")
        return 0

    extra_kwargs = _parse_key_values(args.arg)
    loaders = get_dataloaders(
        args.dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        download=args.download,
        **extra_kwargs,
    )
    split_names = ("train", "val", "test")
    for split_name, loader in zip(split_names, loaders, strict=True):
        print(f"{split_name}: {len(loader.dataset)} samples")
        try:
            batch = next(iter(loader))
        except Exception as exc:
            print(f"  batch: unavailable ({exc})")
            continue
        print(f"  batch[0] shape: {_format_batch_shape(batch)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
