# author: Matus Halak (@matushalak)
# Color-opponent MNIST dataset construction and utilities
from __future__ import annotations

import re
from collections import OrderedDict
from typing import Iterable, Mapping, Sequence, Literal

import torch
import torch.utils.data as data


def _get_targets(dataset):
    if isinstance(dataset, data.Subset):
        base_targets = _get_targets(dataset.dataset)
        if base_targets is None:
            return None
        return base_targets[torch.as_tensor(dataset.indices, dtype=torch.long)]

    targets = getattr(dataset, "targets", None)
    if targets is None:
        return None
    return torch.as_tensor(targets, dtype=torch.long)


def _parse_digits(digits):
    if digits is None:
        return None

    if isinstance(digits, str):
        stripped = digits.strip()
        if stripped.lower() in {"all", "*"}:
            return None
        if stripped == "":
            return []
        parts = [part.strip() for part in stripped.split(",") if part.strip() != ""]
        parsed = sorted({int(part) for part in parts})
    else:
        parsed = sorted({int(digit) for digit in digits})

    if any(digit < 0 or digit > 9 for digit in parsed):
        raise ValueError(f"Digits must be in [0, 9]. Got: {parsed}")
    return parsed


def _coerce_rgb(color: Sequence[float], num_values: int) -> torch.Tensor:
    tensor = torch.as_tensor(color, dtype=torch.float32)
    if tensor.numel() != 3:
        raise ValueError(f"RGB endpoints must have 3 values. Got: {color}")

    max_value = float(num_values - 1)
    if torch.all((tensor >= 0.0) & (tensor <= 1.0)):
        tensor = tensor * max_value

    if torch.any(tensor < 0.0) or torch.any(tensor > max_value):
        raise ValueError(
            f"RGB values must be within [0, {max_value}] or [0, 1]. Got: {color}"
        )
    return tensor


def _default_axis_definitions(num_values: int) -> OrderedDict[str, tuple[tuple[float, float, float], tuple[float, float, float]]]:
    ''''
    Default to four 3d diagonal axes in RGB cube, with endpoints at the corners of the cube. 
    Values are scaled to num_values.
    '''
    max_value = float(num_values - 1)
    return OrderedDict(
        [
            ("axis1", ((max_value, 0.0, 0.0), (0.0, max_value, max_value))),  # red <-> cyan
            ("axis2", ((max_value, max_value, 0.0), (0.0, 0.0, max_value))),  # yellow <-> blue
            ("axis3", ((0.0, max_value, 0.0), (max_value, 0.0, max_value))),  # green <-> magenta
            ("axis4", ((0.0, 0.0, 0.0), (max_value, max_value, max_value))),  # black <-> white
        ]
    )


def _normalize_axis_definitions(
    axis_definitions: Mapping[str, Sequence[Sequence[float]]] | None,
    num_values: int,
) -> OrderedDict[str, tuple[torch.Tensor, torch.Tensor]]:
    if axis_definitions is None:
        axis_definitions = _default_axis_definitions(num_values)

    normalized = OrderedDict()
    for axis_name, endpoints in axis_definitions.items():
        if len(endpoints) != 2:
            raise ValueError(
                f"Axis '{axis_name}' must have two RGB endpoints. Got: {endpoints}"
            )
        start = _coerce_rgb(endpoints[0], num_values)
        end = _coerce_rgb(endpoints[1], num_values)
        normalized[str(axis_name)] = (start, end)

    if not normalized:
        raise ValueError("At least one axis definition is required.")
    return normalized


def _parse_level_token(level_token: str, n_levels_per_axis: int) -> int:
    raw = level_token.strip().lower().replace("_", "")
    if raw.startswith("level"):
        suffix = raw[5:]
        if suffix == "":
            raise ValueError(f"Invalid level token '{level_token}'")
        number = int(suffix)
        level_idx = 0 if number == 0 else number - 1
    else:
        level_idx = int(level_token.strip())

    if level_idx < 0 or level_idx >= n_levels_per_axis:
        raise ValueError(
            f"Level index {level_idx} out of range for n_levels_per_axis={n_levels_per_axis}. "
            f"Valid levels: 0..{n_levels_per_axis - 1}"
        )
    return level_idx


def _parse_axis_level_digits_spec(
    spec: str,
    axis_names: Sequence[str],
    n_levels_per_axis: int,
) -> tuple[str, dict[int, list[int] | None]]:
    if ":" not in spec:
        raise ValueError(
            f"Invalid axis-level spec '{spec}'. "
            "Expected format axis:(level:d1,d2),(level:d3,...)"
        )

    axis_name, assignments = spec.split(":", 1)
    axis_name = axis_name.strip()
    if axis_name not in axis_names:
        raise ValueError(
            f"Unknown axis '{axis_name}' in spec '{spec}'. Valid axes: {list(axis_names)}"
        )

    assignments = assignments.strip()
    groups = re.findall(r"\(([^()]*)\)", assignments)
    if groups:
        remainder = re.sub(r"\([^()]*\)", "", assignments).strip(" ,;")
        if remainder:
            raise ValueError(
                f"Mixed assignment formats in spec '{spec}'. "
                "Use either '(level:d1,d2),(...)' or 'level:d1,d2;...'."
            )
        entries = groups
    else:
        entries = [entry.strip() for entry in assignments.split(";") if entry.strip() != ""]

    if not entries:
        raise ValueError(
            f"No level assignments found in spec '{spec}'. "
            "Expected at least one assignment like level1:1,2"
        )

    level_map: dict[int, list[int] | None] = {}
    for entry in entries:
        if ":" not in entry:
            raise ValueError(
                f"Invalid assignment '{entry}' in spec '{spec}'. "
                "Expected format level:d1,d2"
            )
        level_token, digits_token = entry.split(":", 1)
        level_idx = _parse_level_token(level_token, n_levels_per_axis)
        level_map[level_idx] = _parse_digits(digits_token)
    return axis_name, level_map


def parse_axis_level_digits_specs(
    specs: Sequence[str],
    axis_names: Sequence[str],
    n_levels_per_axis: int,
) -> dict[str, dict[int, list[int] | None]]:
    parsed: dict[str, dict[int, list[int] | None]] = {}
    for spec in specs:
        axis_name, level_map = _parse_axis_level_digits_spec(
            spec=spec,
            axis_names=axis_names,
            n_levels_per_axis=n_levels_per_axis,
        )
        axis_map = parsed.setdefault(axis_name, {})
        axis_map.update(level_map)
    return parsed


def _normalize_axis_level_digits(
    axis_names: Sequence[str],
    n_levels_per_axis: int,
    axis_level_digits: Mapping[str, Mapping[int, Iterable[int] | str | None]] | None,
    axis_level_digit_specs: Sequence[str] | None,
    kwargs,
) -> dict[str, dict[int, list[int] | None]]:
    normalized = {
        axis_name: {level_idx: None for level_idx in range(n_levels_per_axis)}
        for axis_name in axis_names
    }

    if axis_level_digits is not None:
        for axis_name, level_map in axis_level_digits.items():
            if axis_name not in normalized:
                raise ValueError(
                    f"Unknown axis '{axis_name}' in axis_level_digits. "
                    f"Valid axes: {list(axis_names)}"
                )
            for level_idx, digits in level_map.items():
                level_idx = int(level_idx)
                if level_idx < 0 or level_idx >= n_levels_per_axis:
                    raise ValueError(
                        f"Level {level_idx} out of range for axis '{axis_name}'. "
                        f"Valid levels: 0..{n_levels_per_axis - 1}"
                    )
                normalized[axis_name][level_idx] = _parse_digits(digits)

    if axis_level_digit_specs:
        parsed = parse_axis_level_digits_specs(
            specs=axis_level_digit_specs,
            axis_names=axis_names,
            n_levels_per_axis=n_levels_per_axis,
        )
        for axis_name, level_map in parsed.items():
            for level_idx, digits in level_map.items():
                normalized[axis_name][level_idx] = digits

    key_pattern = re.compile(r"^(?P<axis>[A-Za-z0-9_-]+)_level_(?P<level>\d+)_digits$")
    unknown_keys = []
    for key, digits in kwargs.items():
        match = key_pattern.match(key)
        if not match:
            unknown_keys.append(key)
            continue
        axis_name = match.group("axis")
        level_idx = int(match.group("level"))
        if axis_name not in normalized:
            raise ValueError(
                f"Unknown axis '{axis_name}' in kwarg '{key}'. Valid axes: {list(axis_names)}"
            )
        if level_idx < 0 or level_idx >= n_levels_per_axis:
            raise ValueError(
                f"Level {level_idx} out of range in kwarg '{key}'. "
                f"Valid levels: 0..{n_levels_per_axis - 1}"
            )
        normalized[axis_name][level_idx] = _parse_digits(digits)

    if unknown_keys:
        raise ValueError(
            "Unknown kwargs: "
            f"{sorted(unknown_keys)}. Expected '<axis>_level_<idx>_digits'."
        )

    return normalized


def _build_axis_level_colors(
    start_color: torch.Tensor,
    end_color: torch.Tensor,
    level_idx: int,
    n_levels_per_axis: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if n_levels_per_axis <= 1:
        alpha = 0.0
    else:
        alpha = float(level_idx) / float(n_levels_per_axis - 1)

    foreground_color = (1.0 - alpha) * start_color + alpha * end_color
    background_color = (1.0 - alpha) * end_color + alpha * start_color
    return foreground_color, background_color


def _build_color_lut(
    foreground_color: torch.Tensor,
    background_color: torch.Tensor,
    num_values: int,
) -> torch.Tensor:
    if num_values < 2:
        raise ValueError("num_values must be >= 2.")
    denom = float(num_values - 1)
    pixel_values = (torch.arange(num_values, dtype=torch.float32) / denom).unsqueeze(1)
    rgb = (1.0 - pixel_values) * background_color.unsqueeze(0) + pixel_values * foreground_color.unsqueeze(0)
    return torch.round(rgb).long().clamp_(min=0, max=num_values - 1)


class _DigitAxisLevelColorDataset(data.Dataset):
    def __init__(
        self,
        dataset,
        axis_label: int,
        level_label: int,
        foreground_color: torch.Tensor,
        background_color: torch.Tensor,
        num_values: int,
        include_digits: Iterable[int] | None,
    ):
        self.dataset = dataset
        self.axis_label = int(axis_label)
        self.level_label = int(level_label)
        self.color_lut = _build_color_lut(
            foreground_color=foreground_color,
            background_color=background_color,
            num_values=num_values,
        )
        self.filtered_indices = None

        if include_digits is not None:
            targets = _get_targets(dataset)
            if targets is None:
                raise ValueError("Could not extract targets for digit filtering.")
            include_digits = torch.as_tensor(sorted(set(include_digits)), dtype=torch.long)
            mask = torch.isin(targets, include_digits)
            self.filtered_indices = torch.nonzero(mask, as_tuple=False).squeeze(1).tolist()

    def __len__(self):
        if self.filtered_indices is None:
            return len(self.dataset)
        return len(self.filtered_indices)

    def __getitem__(self, idx):
        if self.filtered_indices is not None:
            idx = self.filtered_indices[idx]

        img, digit_label = self.dataset[idx]
        img = img.long()
        if img.dim() == 3:
            if img.shape[0] != 1:
                raise ValueError(
                    f"Expected single-channel MNIST input. Got shape {tuple(img.shape)}"
                )
            img = img.squeeze(0)
        elif img.dim() != 2:
            raise ValueError(f"Expected image shape [H,W] or [1,H,W]. Got {tuple(img.shape)}")

        rgb_img = self.color_lut[img]
        rgb_img = rgb_img.permute(2, 0, 1).contiguous()
        return rgb_img, (int(digit_label), self.axis_label, self.level_label)


def combine_color_opponent_mnist(
    mnist_loader,
    n_levels_per_axis: int,
    batch_size: int | None = None,
    shuffle: bool = True,
    num_workers: int | None = None,
    drop_last: bool = False,
    num_values: int = 16,
    axis_definitions: Mapping[str, Sequence[Sequence[float]]] | None = None,
    axis_level_digits: Mapping[str, Mapping[int, Iterable[int] | str | None]] | None = None,
    axis_level_digit_specs: Sequence[str] | None = None,
    **kwargs,
):
    """
    Combine multiple color-opponent variants of quantized MNIST into one data loader.

    Output labels are tuples: (digit_class, axis_index, level_index).

    Digit filters can be provided with either:
    - axis_level_digits mapping:
        {"axis1": {0: [1,2], 1: [3,4]}, "axis2": {0: [0]}}
    - axis_level_digit_specs list (CLI-friendly):
        ["axis1:(level1:1,2),(level2:3,4)", "axis2:(0:0,9)"]
      Notes:
      - level tokens like "level1" are interpreted as 1-based.
      - numeric level tokens like "0" are interpreted as 0-based.
    - kwargs pattern:
        axis1_level_0_digits=[...], axis2_level_1_digits=[...], ...

    Any unspecified axis-level defaults to all digits.
    """
    n_levels_per_axis = int(n_levels_per_axis)
    if n_levels_per_axis < 1:
        raise ValueError("n_levels_per_axis must be >= 1.")

    axes = _normalize_axis_definitions(
        axis_definitions=axis_definitions,
        num_values=num_values,
    )
    axis_names = list(axes.keys())

    digit_filters = _normalize_axis_level_digits(
        axis_names=axis_names,
        n_levels_per_axis=n_levels_per_axis,
        axis_level_digits=axis_level_digits,
        axis_level_digit_specs=axis_level_digit_specs,
        kwargs=kwargs,
    )

    axis_level_datasets = []
    for axis_idx, axis_name in enumerate(axis_names):
        start_color, end_color = axes[axis_name]
        for level_idx in range(n_levels_per_axis):
            foreground_color, background_color = _build_axis_level_colors(
                start_color=start_color,
                end_color=end_color,
                level_idx=level_idx,
                n_levels_per_axis=n_levels_per_axis,
            )
            axis_level_dataset = _DigitAxisLevelColorDataset(
                dataset=mnist_loader.dataset,
                axis_label=axis_idx,
                level_label=level_idx,
                foreground_color=foreground_color,
                background_color=background_color,
                num_values=num_values,
                include_digits=digit_filters[axis_name][level_idx],
            )
            axis_level_datasets.append(axis_level_dataset)

    combined_dataset = data.ConcatDataset(axis_level_datasets)

    if batch_size is None:
        batch_size = mnist_loader.batch_size
    if num_workers is None:
        num_workers = mnist_loader.num_workers

    pin_memory = bool(getattr(mnist_loader, "pin_memory", False))
    persistent_workers = bool(getattr(mnist_loader, "persistent_workers", False)) and num_workers > 0

    combined_loader = data.DataLoader(
        combined_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )

    combined_loader.axis_names = axis_names
    combined_loader.n_levels_per_axis = n_levels_per_axis
    return combined_loader

# Separate function for converting CoMNIST batches to different color space
# TODO:
def convert_comnist_batch(batch:torch.Tensor, 
                          from_color_space:Literal['rgb', 'hsv', 'ycbcr']='rgb',
                          to_color_space:Literal['rgb', 'hsv', 'ycbcr']='hsv'
                          )->torch.Tensor:
    pass

if __name__ == '__main__':
    from dataset_utils import visualize_dataset
    from mnist import mnist
    loader = combine_color_opponent_mnist(
        mnist_loader=mnist(batch_size=64, num_workers=2, num_values=16)[0],
        n_levels_per_axis=4,
        num_values=16,
        axis_level_digits = {'axis1':{0:[0], 1:[0], 2:[3], 3:[0]},
                             'axis2':{0:[3], 1:[0], 2:[3], 3:[3]},
                             'axis3':{0:[8], 1:[5], 2:[5], 3:[5]},
                             'axis4':{0:[8], 1:[8], 2:[8], 3:[5]}
                             })
    print(f"Combined dataset size: {len(loader.dataset)}")
    batch = next(iter(loader))
    print(f"Batch image shape: {batch[0].shape}, Batch label shape: {len(batch[1])},{batch[1][0].shape}")
    grid = visualize_dataset(loader, n_examples=40)
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 8))
    plt.imshow(grid.permute(1, 2, 0))
    plt.axis("off")
    plt.title("Sample from Combined Color-Opponent MNIST")
    plt.show()