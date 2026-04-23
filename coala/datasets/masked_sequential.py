import re

import torch
import torch.utils.data as data
from torch.utils.data._utils.collate import default_collate


def _extract_dataset_targets(dataset: data.Dataset) -> list[int]:
    if isinstance(dataset, data.Subset):
        base_targets = _extract_dataset_targets(dataset.dataset)
        return [int(base_targets[index]) for index in dataset.indices]
    for attr_name in ("targets", "labels"):
        if hasattr(dataset, attr_name):
            targets = getattr(dataset, attr_name)
            if isinstance(targets, torch.Tensor):
                return [int(value) for value in targets.tolist()]
            return [int(value) for value in targets]
    raise AttributeError(f"Dataset of type {type(dataset)!r} does not expose targets/labels.")


class MaskedSequentialDataset(data.Dataset):
    CONTRASTIVE_VISIBLE_RATIO = 0.3
    """
    Masked Sequential dataset where 
        image is divided into non-overlapping patches 
        and each patch is masked with a random binary mask.

    Args:
        dataset: Which dataset to use (eg. MNIST)
        patch_size: Size of each patch
        mask_ratio: Mask ratio (eg. 50% of patches masked)
        number_of_masks: Number of distinct sampled masks
        timesteps_per_mask: How many consecutive timesteps to reuse each mask
    
    
    Each retrieved batch is of the shape: (batch_size, num_timeframes, channels, height, width)
    Each __getitem__ is:
        masked_imgs: (num_timeframes, channels, height, width)
        label: (1, *label_dims)
    """

    def __init__(
        self,
        dataset: data.Dataset,
        patch_size: int,
        mask_ratio: float,
        number_of_masks: int = 100,
        timesteps_per_mask: int = 1,
        mask_pattern: str = "random",
        masked_fill: str | float = 0.0,
        noise_sigma: float = 0.25,
        visible_corrupt: bool = False,
        num_digits: int = 1,
        image_visibility: str = "all",
        target_type: str = "label",
        contrastive: bool = False,
    ):
        super().__init__()
        if patch_size <= 0:
            raise ValueError("patch_size must be > 0.")
        if not (0.0 <= mask_ratio <= 1.0):
            raise ValueError("mask_ratio must be in [0, 1].")
        if number_of_masks <= 0:
            raise ValueError("number_of_masks must be > 0.")
        if timesteps_per_mask <= 0:
            raise ValueError("timesteps_per_mask must be > 0.")
        if num_digits <= 0:
            raise ValueError("num_digits must be > 0.")
        if mask_pattern not in ("random", "structured"):
            raise ValueError("mask_pattern must be one of: 'random', 'structured'.")
        if contrastive and mask_pattern != "structured":
            raise ValueError("contrastive=True is only supported with mask_pattern='structured'.")
        if isinstance(masked_fill, str):
            if masked_fill != "random":
                raise ValueError("masked_fill must be 'random' or a float.")
            self.masked_fill_mode = "random"
            self.masked_fill_value = 0.0
        else:
            try:
                self.masked_fill_value = float(masked_fill)
            except (TypeError, ValueError):
                raise ValueError("masked_fill must be 'random' or a float.") from None
            self.masked_fill_mode = "constant"
        if target_type not in ("label", "image", "both"):
            raise ValueError("target_type must be one of: 'label', 'image', 'both'.")

        self.dataset = dataset
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio
        self.number_of_masks = number_of_masks
        self.timesteps_per_mask = timesteps_per_mask
        self.num_digits = num_digits
        self.timeframes_per_digit = number_of_masks * timesteps_per_mask
        self.num_timeframes = self.timeframes_per_digit * self.num_digits
        self.mask_pattern = mask_pattern
        self.target_type = target_type
        self.visible_corrupt = visible_corrupt
        self.noise_sigma = noise_sigma
        self.image_visibility = image_visibility
        self.contrastive = contrastive
        self.image_visibility_mode, self.image_visibility_stride = self._parse_image_visibility(image_visibility)

        sample_img, _ = self.dataset[0]
        if sample_img.dim() != 3:
            raise ValueError("Expected images in (channels, height, width) format.")
        _, self.height, self.width = sample_img.shape
        self.blank_stimulus_value = float(sample_img.min().item())
        if self.height % patch_size != 0 or self.width % patch_size != 0:
            raise ValueError(
                f"Image size ({self.height}, {self.width}) must be divisible by patch_size={patch_size}."
            )

        self.patches_h = self.height // patch_size
        self.patches_w = self.width // patch_size
        self.num_patches = self.patches_h * self.patches_w
        self.num_mask = int(round(mask_ratio * self.num_patches))
        self.num_mask = max(0, min(self.num_patches, self.num_mask))
        self.num_keep = self.num_patches - self.num_mask

        coords_h = torch.arange(self.patches_h, dtype=torch.float32)
        coords_w = torch.arange(self.patches_w, dtype=torch.float32)
        grid_h, grid_w = torch.meshgrid(coords_h, coords_w, indexing="ij")
        self.patch_coords = torch.stack((grid_h.reshape(-1), grid_w.reshape(-1)), dim=1)

        self.dataset_targets: list[int] | None = None
        self.label_to_indices: dict[int, list[int]] = {}
        if self.num_digits > 1:
            self.dataset_targets = _extract_dataset_targets(self.dataset)
            for dataset_idx, target in enumerate(self.dataset_targets):
                self.label_to_indices.setdefault(int(target), []).append(dataset_idx)
            if self.num_digits > len(self.label_to_indices):
                raise ValueError(
                    f"num_digits={self.num_digits} requires at least that many distinct labels, "
                    f"but only found {len(self.label_to_indices)}."
                )

    def __len__(self) -> int:
        return len(self.dataset)

    @staticmethod
    def _parse_image_visibility(value: str) -> tuple[str, int | None]:
        if value in ("all", "first"):
            return value, None
        match = re.fullmatch(r"every-(\d+)", value)
        if match is None:
            raise ValueError("image_visibility must be 'all', 'first', or 'every-N' with N > 0.")
        stride = int(match.group(1))
        if stride <= 0:
            raise ValueError("image_visibility stride must be > 0.")
        return "every", stride

    def _sample_keep_mask(self) -> torch.BoolTensor:
        m = self.number_of_masks
        t = self.timeframes_per_digit
        p = self.num_patches

        if self.num_keep == 0:
            return torch.zeros(t, p, dtype=torch.bool)
        if self.num_keep == p:
            return torch.ones(t, p, dtype=torch.bool)

        if self.mask_pattern == "random":
            scores = torch.rand(m, p)
            keep_idx = scores.topk(self.num_keep, dim=1, largest=True, sorted=False).indices
            keep = torch.zeros(m, p, dtype=torch.bool)
            keep.scatter_(1, keep_idx, True)
            if self.timesteps_per_mask > 1:
                keep = keep.repeat_interleave(self.timesteps_per_mask, dim=0)
            return keep

        return self._sample_structured_occluder_keep_mask(self.num_mask)

    def _sample_structured_occluder_keep_mask(self, num_mask: int) -> torch.BoolTensor:
        m = self.number_of_masks
        p = self.num_patches
        num_mask = max(0, min(p, int(num_mask)))
        if num_mask == 0:
            return torch.ones(self.timeframes_per_digit, p, dtype=torch.bool)
        if num_mask == p:
            return torch.zeros(self.timeframes_per_digit, p, dtype=torch.bool)

        centers = torch.stack(
            (
                torch.rand(m) * (self.patches_h - 1),
                torch.rand(m) * (self.patches_w - 1),
            ),
            dim=1,
        )
        diffs = self.patch_coords.unsqueeze(0) - centers.unsqueeze(1)
        dists = diffs.square().sum(dim=-1)
        dists = dists + 1e-4 * torch.rand_like(dists)
        mask_idx = dists.topk(num_mask, dim=1, largest=False, sorted=False).indices
        keep = torch.ones(m, p, dtype=torch.bool)
        keep.scatter_(1, mask_idx, False)
        if self.timesteps_per_mask > 1:
            keep = keep.repeat_interleave(self.timesteps_per_mask, dim=0)
        return keep

    def _sample_structured_visible_keep_mask(self, num_keep: int) -> torch.BoolTensor:
        m = self.number_of_masks
        p = self.num_patches
        num_keep = max(0, min(p, int(num_keep)))
        if num_keep == 0:
            return torch.zeros(self.timeframes_per_digit, p, dtype=torch.bool)
        if num_keep == p:
            return torch.ones(self.timeframes_per_digit, p, dtype=torch.bool)

        centers = torch.stack(
            (
                torch.rand(m) * (self.patches_h - 1),
                torch.rand(m) * (self.patches_w - 1),
            ),
            dim=1,
        )
        diffs = self.patch_coords.unsqueeze(0) - centers.unsqueeze(1)
        dists = diffs.square().sum(dim=-1)
        dists = dists + 1e-4 * torch.rand_like(dists)
        keep_idx = dists.topk(num_keep, dim=1, largest=False, sorted=False).indices
        keep = torch.zeros(m, p, dtype=torch.bool)
        keep.scatter_(1, keep_idx, True)
        if self.timesteps_per_mask > 1:
            keep = keep.repeat_interleave(self.timesteps_per_mask, dim=0)
        return keep

    def _sample_distinct_keep_mask(self, reference_keep: torch.BoolTensor) -> torch.BoolTensor:
        if self.num_keep in (0, self.num_patches):
            return reference_keep.clone()
        candidate = self._sample_keep_mask()
        for _ in range(8):
            if not torch.equal(candidate, reference_keep):
                return candidate
            candidate = self._sample_keep_mask()
        return candidate

    def _sample_contrastive_visible_keep_mask(self, reference_keep: torch.BoolTensor) -> torch.BoolTensor:
        if self.mask_ratio == 0.0:
            return torch.ones_like(reference_keep)

        visible_ratio = self.CONTRASTIVE_VISIBLE_RATIO
        visible_keep = int(round(visible_ratio * self.num_patches))
        visible_keep = max(0, min(self.num_patches, visible_keep))
        if visible_keep in (0, self.num_patches):
            return torch.full_like(reference_keep, visible_keep == self.num_patches)

        candidate = self._sample_structured_visible_keep_mask(visible_keep)
        for _ in range(8):
            if not torch.equal(candidate, reference_keep):
                return candidate
            candidate = self._sample_structured_visible_keep_mask(visible_keep)
        return candidate

    def _expand_keep_mask(self, keep: torch.BoolTensor) -> torch.BoolTensor:
        keep = keep.view(self.timeframes_per_digit, self.patches_h, self.patches_w)
        keep = keep.repeat_interleave(self.patch_size, dim=1).repeat_interleave(self.patch_size, dim=2)
        return keep.unsqueeze(1)

    def _masked_fill_tensor(self, img_t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        if self.masked_fill_mode == "random":
            return noise
        return torch.full_like(img_t, self.masked_fill_value)

    def _compose_visible_pixels(
        self,
        img_t: torch.Tensor,
        noise: torch.Tensor,
        *,
        force_noise: bool = False,
    ) -> torch.Tensor:
        if force_noise or self.visible_corrupt:
            return (img_t + noise).clamp_(-1.0, 1.0)
        return img_t

    def _build_masked_view(
        self,
        img_t: torch.Tensor,
        reveal_mask: torch.BoolTensor,
        *,
        force_visible_noise: bool = False,
    ) -> torch.Tensor:
        noise = (self.noise_sigma * torch.randn_like(img_t)).clamp_(-1.0, 1.0)
        visible = self._compose_visible_pixels(img_t, noise, force_noise=force_visible_noise)
        masked_imgs = torch.where(reveal_mask, visible, self._masked_fill_tensor(img_t, noise))
        return self._apply_image_visibility(masked_imgs)

    def _apply_image_visibility(self, masked_imgs: torch.Tensor) -> torch.Tensor:
        if self.image_visibility_mode == "all":
            return masked_imgs

        visible_frames = torch.zeros(self.timeframes_per_digit, dtype=torch.bool, device=masked_imgs.device)
        if self.image_visibility_mode == "first":
            visible_frames[0] = True
        else:
            visible_frames[:: self.image_visibility_stride] = True

        blank = torch.full_like(masked_imgs, self.blank_stimulus_value)
        return torch.where(visible_frames.view(-1, 1, 1, 1), masked_imgs, blank)

    def _sample_component_indices(self, idx: int, base_label: int) -> list[int]:
        if self.num_digits == 1:
            return [idx]

        other_labels = sorted(label for label in self.label_to_indices if label != base_label)
        needed = self.num_digits - 1
        if len(other_labels) < needed:
            raise ValueError(
                f"Cannot sample {self.num_digits} distinct digits when only {len(other_labels) + 1} are available."
            )

        sampled_order = torch.randperm(len(other_labels))[:needed].tolist()
        component_indices = [idx]
        for sampled_idx in sampled_order:
            label = other_labels[sampled_idx]
            label_indices = self.label_to_indices[label]
            component_indices.append(label_indices[torch.randint(len(label_indices), (1,)).item()])
        return component_indices

    def _build_component_sequences(
        self,
        img: torch.Tensor,
        label: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.LongTensor]:
        img_t = img.unsqueeze(0).expand(self.timeframes_per_digit, -1, -1, -1)
        keep = self._sample_keep_mask()
        keep_pixels = self._expand_keep_mask(keep)

        clean_sequence = img_t.clone()
        label_sequence = torch.full((self.timeframes_per_digit,), int(label), dtype=torch.long)

        if not self.contrastive:
            masked_imgs = self._build_masked_view(img_t, keep_pixels, force_visible_noise=False)
            return masked_imgs, clean_sequence, label_sequence

        contrastive_visible_keep = self._sample_contrastive_visible_keep_mask(keep)
        contrastive_keep_pixels = self._expand_keep_mask(contrastive_visible_keep)
        primary_view = self._build_masked_view(img_t, keep_pixels, force_visible_noise=True)
        inverse_view = self._build_masked_view(img_t, contrastive_keep_pixels, force_visible_noise=True)
        masked_imgs = torch.stack((primary_view, inverse_view), dim=0)
        clean_sequence = clean_sequence.unsqueeze(0).expand(2, -1, -1, -1, -1).clone()
        label_sequence = label_sequence.unsqueeze(0).expand(2, -1).clone()
        return masked_imgs, clean_sequence, label_sequence

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int | torch.Tensor | dict[str, torch.Tensor]]:
        img, label = self.dataset[idx]
        label_value = int(torch.as_tensor(label).item())
        component_indices = self._sample_component_indices(idx, label_value)

        if len(component_indices) == 1:
            masked_imgs, clean_sequence, label_sequence = self._build_component_sequences(img, label_value)
            clean_target = img
            label_target: int | torch.Tensor = label_value
        else:
            masked_parts = []
            clean_parts = []
            label_parts = []
            for component_idx in component_indices:
                component_img, component_label = self.dataset[component_idx]
                component_label_value = int(torch.as_tensor(component_label).item())
                masked_part, clean_part, label_part = self._build_component_sequences(component_img, component_label_value)
                masked_parts.append(masked_part)
                clean_parts.append(clean_part)
                label_parts.append(label_part)
            cat_dim = 1 if self.contrastive else 0
            masked_imgs = torch.cat(masked_parts, dim=cat_dim)
            clean_sequence = torch.cat(clean_parts, dim=cat_dim)
            label_sequence = torch.cat(label_parts, dim=cat_dim)
            clean_target = clean_sequence
            label_target = label_sequence

        if self.contrastive and len(component_indices) == 1:
            clean_target = clean_sequence
            label_target = label_sequence

        if self.target_type == "label":
            target = label_target
        elif self.target_type == "image":
            target = clean_target
        else:
            target = {
                "image": clean_target,
                "label": torch.as_tensor(label_target, dtype=torch.long),
            }
        return masked_imgs, target


def _flatten_contrastive_tensor(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim < 2 or tensor.shape[1] != 2:
        return tensor
    return tensor.reshape(tensor.shape[0] * tensor.shape[1], *tensor.shape[2:])


def _flatten_contrastive_targets(targets):
    if isinstance(targets, dict):
        flattened = {key: _flatten_contrastive_targets(value) for key, value in targets.items()}
        batch_size = next(iter(flattened.values())).shape[0] if flattened else 0
        flattened["contrastive_group"] = torch.arange(batch_size // 2, dtype=torch.long).repeat_interleave(2)
        flattened["contrastive_view"] = torch.arange(2, dtype=torch.long).repeat(batch_size // 2)
        flattened["contrastive_positive_index"] = torch.arange(batch_size, dtype=torch.long).view(-1, 2).flip(1).reshape(-1)
        return flattened
    if torch.is_tensor(targets):
        return _flatten_contrastive_tensor(targets)
    return targets


def masked_sequential_collate(batch):
    masked_imgs, targets = default_collate(batch)
    if masked_imgs.ndim != 6 or masked_imgs.shape[1] != 2:
        return masked_imgs, targets
    masked_imgs = masked_imgs.reshape(masked_imgs.shape[0] * masked_imgs.shape[1], *masked_imgs.shape[2:])
    return masked_imgs, _flatten_contrastive_targets(targets)
