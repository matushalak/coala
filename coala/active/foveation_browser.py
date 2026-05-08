from __future__ import annotations

import argparse
import base64
import io
import json
import random
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import torch
from torchvision import transforms
from torchvision.datasets import MNIST, STL10

from coala import DATADIR
from coala.active.foveation import overlay_fixation_marker, render_foveation_bundle, tensor_to_pil_image
from coala.datasets.stl10 import STL10_CLASS_NAMES


APP_DIR = Path(__file__).resolve().parent
INDEX_HTML_PATH = APP_DIR / "foveation_browser.html"


class DatasetRepository:
    def __init__(self, root: str | Path = DATADIR, download: bool = True):
        self.root = Path(root)
        self.download = download
        self.to_tensor = transforms.ToTensor()
        self.mnist_root = self.root
        self.stl10_root = self.root / "stl10"
        self.stl10_root.mkdir(parents=True, exist_ok=True)

        self.mnist_train = MNIST(self.mnist_root, train=True, download=download)
        self.mnist_test = MNIST(self.mnist_root, train=False, download=download)
        self.stl10_train = STL10(self.stl10_root, split="train", download=download)
        self.stl10_test = STL10(self.stl10_root, split="test", download=download)

    def manifest(self) -> dict[str, Any]:
        return {
            "datasets": {
                "mnist": {
                    "display_name": "MNIST",
                    "splits": {
                        "train": len(self.mnist_train),
                        "test": len(self.mnist_test),
                    },
                    "labels": [str(idx) for idx in range(10)],
                    "image_size": [28, 28],
                },
                "stl10": {
                    "display_name": "STL-10",
                    "splits": {
                        "train": len(self.stl10_train),
                        "test": len(self.stl10_test),
                    },
                    "labels": list(STL10_CLASS_NAMES),
                    "image_size": [96, 96],
                },
            }
        }

    def sample(self, dataset_name: str, split: str, index: int | None = None) -> dict[str, Any]:
        dataset = self._get_dataset(dataset_name, split)
        if index is None:
            index = random.randrange(len(dataset))
        if not (0 <= index < len(dataset)):
            raise IndexError(f"Index {index} out of range for {dataset_name}:{split}.")

        image, label = dataset[index]
        image_tensor = self.to_tensor(image)
        return {
            "dataset": dataset_name,
            "split": split,
            "index": index,
            "label": int(label),
            "label_name": self._label_name(dataset_name, int(label)),
            "image": image_tensor,
        }

    def _get_dataset(self, dataset_name: str, split: str):
        if dataset_name == "mnist":
            if split == "train":
                return self.mnist_train
            if split == "test":
                return self.mnist_test
        if dataset_name == "stl10":
            if split == "train":
                return self.stl10_train
            if split == "test":
                return self.stl10_test
        raise KeyError(f"Unsupported dataset/split combination: {dataset_name}:{split}")

    @staticmethod
    def _label_name(dataset_name: str, label: int) -> str:
        if dataset_name == "mnist":
            return str(label)
        if 0 <= label < len(STL10_CLASS_NAMES):
            return STL10_CLASS_NAMES[label]
        return "unknown"


def _parse_float(query: dict[str, list[str]], name: str, default: float) -> float:
    value = query.get(name, [str(default)])[0]
    return float(value)


def _parse_int(query: dict[str, list[str]], name: str, default: int | None = None) -> int | None:
    raw = query.get(name, ["" if default is None else str(default)])[0]
    if raw == "":
        return default
    return int(raw)


def _image_to_data_url(image: torch.Tensor) -> str:
    pil_image = tensor_to_pil_image(image)
    buffer = io.BytesIO()
    pil_image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _render_payload(sample: dict[str, Any], query: dict[str, list[str]]) -> dict[str, Any]:
    fixation_x = _parse_float(query, "fixation_x", 0.5)
    fixation_y = _parse_float(query, "fixation_y", 0.5)
    fovea_radius_ratio = _parse_float(query, "fovea_radius_ratio", 0.12)
    kerw_coef = _parse_float(query, "kerw_coef", 0.06)
    ring_spacing = _parse_float(query, "ring_spacing", 0.2)
    parafoveal_noise_std = _parse_float(query, "parafoveal_noise_std", 0.0)
    magnif_fov_ratio = _parse_float(query, "magnif_fov_ratio", 0.22)
    magnif_k_ratio = _parse_float(query, "magnif_k_ratio", 0.22)
    cover_ratio = _parse_float(query, "cover_ratio", 1.0)
    marker_radius = int(round(_parse_float(query, "marker_radius", 3.0)))

    rendered = render_foveation_bundle(
        sample["image"],
        fixation_x,
        fixation_y,
        kerw_coef=kerw_coef,
        fovea_radius_ratio=fovea_radius_ratio,
        ring_spacing=ring_spacing,
        parafoveal_noise_std=parafoveal_noise_std,
        magnif_fov_ratio=magnif_fov_ratio,
        magnif_k_ratio=magnif_k_ratio,
        cover_ratio=cover_ratio,
    )
    overlay = overlay_fixation_marker(rendered["original"], fixation_x, fixation_y, marker_radius=marker_radius)

    return {
        "dataset": sample["dataset"],
        "split": sample["split"],
        "index": sample["index"],
        "label": sample["label"],
        "label_name": sample["label_name"],
        "image_size": [int(sample["image"].shape[-1]), int(sample["image"].shape[-2])],
        "fixation": {"x": fixation_x, "y": fixation_y},
        "params": {
            "fovea_radius_ratio": fovea_radius_ratio,
            "kerw_coef": kerw_coef,
            "ring_spacing": ring_spacing,
            "parafoveal_noise_std": parafoveal_noise_std,
            "magnif_fov_ratio": magnif_fov_ratio,
            "magnif_k_ratio": magnif_k_ratio,
            "cover_ratio": cover_ratio,
            "marker_radius": marker_radius,
        },
        "images": {
            "original": _image_to_data_url(rendered["original"]),
            "overlay": _image_to_data_url(overlay),
            "perceptual": _image_to_data_url(rendered["perceptual"]),
            "combined": _image_to_data_url(rendered["combined"]),
        },
    }


def build_handler(repository: DatasetRepository):
    class FoveationBrowserHandler(BaseHTTPRequestHandler):
        def _send_json(self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_text(self, body: str, status: HTTPStatus = HTTPStatus.OK, content_type: str = "text/html; charset=utf-8") -> None:
            encoded = body.encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            query = parse_qs(parsed.query)

            if parsed.path == "/":
                self._send_text(INDEX_HTML_PATH.read_text(encoding="utf-8"))
                return
            if parsed.path == "/api/manifest":
                self._send_json(repository.manifest())
                return
            if parsed.path == "/api/render":
                try:
                    dataset_name = query.get("dataset", ["mnist"])[0]
                    split = query.get("split", ["train"])[0]
                    index = _parse_int(query, "index", None)
                    sample = repository.sample(dataset_name, split, index)
                    self._send_json(_render_payload(sample, query))
                except Exception as exc:  # noqa: BLE001
                    self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                return

            self._send_text("Not found", status=HTTPStatus.NOT_FOUND, content_type="text/plain; charset=utf-8")

        def log_message(self, format: str, *args) -> None:
            return

    return FoveationBrowserHandler


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--data_dir", type=str, default=str(DATADIR))
    parser.add_argument("--download", action=argparse.BooleanOptionalAction, default=True)
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    repository = DatasetRepository(root=args.data_dir, download=args.download)
    server = ThreadingHTTPServer((args.host, args.port), build_handler(repository))
    print(f"Serving foveation browser at http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
