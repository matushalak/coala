import re

import torch
import torch.nn.functional as F
import torch.utils.data as data


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


def _parse_corruptions(corruptions: str | tuple[str, ...] | list[str] | None) -> tuple[str, ...]:
    if corruptions is None:
        return ("mask",)
    if isinstance(corruptions, str):
        values = [value.strip() for value in corruptions.split(",")]
    else:
        values = [str(value).strip() for value in corruptions]
    aliases = {
        "gaussian_mix_blur": "mix_blur",
        "gaussian_mix_noise_blur": "mix_blur",
    }
    parsed = tuple(aliases.get(value, value) for value in values if value)
    allowed = {"mask", "gaussian", "mix", "salt_pepper", "blur", "mix_blur"}
    invalid = sorted({value for value in parsed if value not in allowed})
    if invalid:
        raise ValueError(
            "corruptions must contain only: 'mask', 'gaussian', 'mix', 'salt_pepper', 'blur', 'mix_blur'."
        )
    return parsed


def _parse_corruption_sampling(value: str, *, name: str) -> str:
    if value not in ("fixed", "single", "subset"):
        raise ValueError(f"{name} must be one of: 'fixed', 'single', 'subset'.")
    return value


def _parse_schedule(
    value: float | int | tuple[float | int, float | int] | list[float | int],
    *,
    name: str,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float | tuple[float, float]:
    if isinstance(value, (tuple, list)):
        if len(value) != 2:
            raise ValueError(f"{name} schedule must be a scalar or a (start, end) pair.")
        parsed = (float(value[0]), float(value[1]))
        values = parsed
    else:
        parsed = float(value)
        values = (parsed,)

    for item in values:
        if minimum is not None and item < minimum:
            raise ValueError(f"{name} must be >= {minimum}.")
        if maximum is not None and item > maximum:
            raise ValueError(f"{name} must be <= {maximum}.")
    return parsed


class CorruptedSequentialDataset(data.Dataset):
    """
    Sequential dataset with composable image corruptions.

    Args:
        dataset: Which dataset to use (eg. MNIST)
        patch_size: Size of each patch
        mask_ratio: Mask ratio or (start, end) schedule for patch masking
        number_of_masks: Number of distinct sampled masks
        timesteps_per_mask: How many consecutive timesteps to reuse each mask
        corruptions: Available corruptions. If mask is included, masking is applied after visible corruption.
        corruption_sampling: 'fixed', 'single', or 'subset' sampling over the non-mask corruptions
        masked_fill='corrupted': fill masked pixels using an independently sampled corrupted image

    Each retrieved batch is of the shape: (batch_size, num_timeframes, channels, height, width)
    Each __getitem__ is:
        masked_imgs: (num_timeframes, channels, height, width)
        label: (1, *label_dims)
    """

    def __init__(
        self,
        dataset: data.Dataset,
        patch_size: int,
        mask_ratio: float | tuple[float, float] | list[float],
        number_of_masks: int = 100,
        timesteps_per_mask: int = 1,
        mask_pattern: str = "random",
        masked_fill: str | float = 0.0,
        noise_sigma: float | tuple[float, float] | list[float] = 0.25,
        visible_corrupt: bool = False,
        num_digits: int = 1,
        image_visibility: str = "all",
        target_type: str = "label",
        corruptions: str | tuple[str, ...] | list[str] | None = None,
        corruption_sampling: str = "fixed",
        corruption_subset_prob: float = 0.5,
        fill_corruptions: str | tuple[str, ...] | list[str] | None = None,
        fill_corruption_sampling: str | None = None,
        fill_subset_prob: float | None = None,
        mix_alpha: float | tuple[float, float] | list[float] = 0.0,
        mix_noise_sigma: float | tuple[float, float] | list[float] = 1.0,
        salt_pepper_prob: float | tuple[float, float] | list[float] = 0.0,
        blur_kernel_size: int | tuple[int, int] | list[int] = 0,
        blur_sigma: float | tuple[float, float] | list[float] = 1.0,
        epoch: int = 0,
        max_epoch: int | None = None,
    ):
        super().__init__()
        if patch_size <= 0:
            raise ValueError("patch_size must be > 0.")
        if number_of_masks <= 0:
            raise ValueError("number_of_masks must be > 0.")
        if timesteps_per_mask <= 0:
            raise ValueError("timesteps_per_mask must be > 0.")
        if num_digits <= 0:
            raise ValueError("num_digits must be > 0.")
        if epoch < 0:
            raise ValueError("epoch must be >= 0.")
        if max_epoch is not None and max_epoch <= 0:
            raise ValueError("max_epoch must be > 0 when provided.")
        if mask_pattern not in ("random", "structured"):
            raise ValueError("mask_pattern must be one of: 'random', 'structured'.")
        if isinstance(masked_fill, str):
            if masked_fill not in ("random", "corrupted"):
                raise ValueError("masked_fill must be 'random', 'corrupted', or a float.")
            self.masked_fill_mode = masked_fill
            self.masked_fill_value = 0.0
        else:
            try:
                self.masked_fill_value = float(masked_fill)
            except (TypeError, ValueError):
                raise ValueError("masked_fill must be 'random', 'corrupted', or a float.") from None
            self.masked_fill_mode = "constant"
        if target_type not in ("label", "image", "both"):
            raise ValueError("target_type must be one of: 'label', 'image', 'both'.")
        if not (0.0 <= corruption_subset_prob <= 1.0):
            raise ValueError("corruption_subset_prob must be in [0, 1].")
        if fill_subset_prob is not None and not (0.0 <= fill_subset_prob <= 1.0):
            raise ValueError("fill_subset_prob must be in [0, 1].")

        self.dataset = dataset
        self.patch_size = patch_size
        self.mask_ratio = _parse_schedule(mask_ratio, name="mask_ratio", minimum=0.0, maximum=1.0)
        self.number_of_masks = number_of_masks
        self.timesteps_per_mask = timesteps_per_mask
        self.num_digits = num_digits
        self.timeframes_per_digit = number_of_masks * timesteps_per_mask
        self.num_timeframes = self.timeframes_per_digit * self.num_digits
        self.mask_pattern = mask_pattern
        self.target_type = target_type
        self.visible_corrupt = visible_corrupt
        self.noise_sigma = _parse_schedule(noise_sigma, name="noise_sigma", minimum=0.0)
        self.image_visibility = image_visibility
        self.image_visibility_mode, self.image_visibility_stride = self._parse_image_visibility(image_visibility)
        self.corruptions = _parse_corruptions(corruptions)
        self.corruption_sampling = _parse_corruption_sampling(
            corruption_sampling,
            name="corruption_sampling",
        )
        self.corruption_subset_prob = float(corruption_subset_prob)
        self.mix_alpha = _parse_schedule(mix_alpha, name="mix_alpha", minimum=0.0, maximum=1.0)
        self.mix_noise_sigma = _parse_schedule(mix_noise_sigma, name="mix_noise_sigma", minimum=0.0)
        self.salt_pepper_prob = _parse_schedule(
            salt_pepper_prob,
            name="salt_pepper_prob",
            minimum=0.0,
            maximum=1.0,
        )
        self.blur_kernel_size = _parse_schedule(blur_kernel_size, name="blur_kernel_size", minimum=0.0)
        self.blur_sigma = _parse_schedule(blur_sigma, name="blur_sigma", minimum=0.0)
        self.current_epoch = int(epoch)
        self.max_epoch = max_epoch
        self._blur_kernel_cache: dict[tuple[int, float], torch.Tensor] = {}
        self.use_mask = "mask" in self.corruptions
        self.available_corruptions = tuple(corruption for corruption in self.corruptions if corruption != "mask")

        if fill_corruptions is None:
            parsed_fill_corruptions = self.available_corruptions
        else:
            parsed_fill_corruptions = tuple(
                corruption
                for corruption in _parse_corruptions(fill_corruptions)
                if corruption != "mask"
            )
        self.fill_corruptions = parsed_fill_corruptions
        self.fill_corruption_sampling = _parse_corruption_sampling(
            self.corruption_sampling if fill_corruption_sampling is None else fill_corruption_sampling,
            name="fill_corruption_sampling",
        )
        self.fill_subset_prob = (
            self.corruption_subset_prob
            if fill_subset_prob is None
            else float(fill_subset_prob)
        )
        if self.masked_fill_mode == "corrupted" and len(self.fill_corruptions) == 0:
            raise ValueError("masked_fill='corrupted' requires at least one non-mask fill corruption.")

        sample_img, _ = self.dataset[0]
        if sample_img.dim() != 3:
            raise ValueError("Expected images in (channels, height, width) format.")
        _, self.height, self.width = sample_img.shape
        self.blank_stimulus_value = float(sample_img.min().item())
        self.signal_min_value = float(sample_img.min().item())
        self.signal_max_value = float(sample_img.max().item())
        if self.height % patch_size != 0 or self.width % patch_size != 0:
            raise ValueError(
                f"Image size ({self.height}, {self.width}) must be divisible by patch_size={patch_size}."
            )

        self.patches_h = self.height // patch_size
        self.patches_w = self.width // patch_size
        self.num_patches = self.patches_h * self.patches_w

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

    def set_epoch(self, epoch: int, max_epoch: int | None = None) -> None:
        if epoch < 0:
            raise ValueError("epoch must be >= 0.")
        if max_epoch is not None and max_epoch <= 0:
            raise ValueError("max_epoch must be > 0 when provided.")
        self.current_epoch = int(epoch)
        if max_epoch is not None:
            self.max_epoch = int(max_epoch)

    def _curriculum_progress(self) -> float:
        if self.max_epoch is None:
            return 0.0
        if self.max_epoch <= 1:
            return 1.0
        progress = self.current_epoch / float(self.max_epoch - 1)
        return min(1.0, max(0.0, progress))

    def _resolve_schedule_value(self, value: float | tuple[float, float]) -> float:
        if isinstance(value, tuple):
            start, end = value
            return start + (end - start) * self._curriculum_progress()
        return value

    def _resolve_kernel_size(self) -> int:
        kernel_size = int(round(self._resolve_schedule_value(self.blur_kernel_size)))
        if kernel_size <= 1:
            return 0
        if kernel_size % 2 == 0:
            kernel_size += 1
        return kernel_size

    def _mask_counts(self) -> tuple[int, int]:
        mask_ratio = self._resolve_schedule_value(self.mask_ratio)
        num_mask = int(round(mask_ratio * self.num_patches))
        num_mask = max(0, min(self.num_patches, num_mask))
        return num_mask, self.num_patches - num_mask

    def _sample_corruption_plan(
        self,
        candidates: tuple[str, ...],
        sampling: str,
        subset_prob: float,
    ) -> tuple[str, ...]:
        if len(candidates) == 0:
            return ()
        if sampling == "fixed":
            return candidates
        if sampling == "single":
            return (candidates[torch.randint(len(candidates), (1,)).item()],)

        keep = torch.rand(len(candidates)) < subset_prob
        if not torch.any(keep):
            keep[torch.randint(len(candidates), (1,)).item()] = True
        return tuple(candidates[idx] for idx, selected in enumerate(keep.tolist()) if selected)

    def _sample_keep_mask(self) -> torch.BoolTensor:
        m = self.number_of_masks
        t = self.timeframes_per_digit
        p = self.num_patches
        num_mask, num_keep = self._mask_counts()

        if num_keep == 0:
            return torch.zeros(t, p, dtype=torch.bool)
        if num_keep == p:
            return torch.ones(t, p, dtype=torch.bool)

        if self.mask_pattern == "random":
            scores = torch.rand(m, p)
            keep_idx = scores.topk(num_keep, dim=1, largest=True, sorted=False).indices
            keep = torch.zeros(m, p, dtype=torch.bool)
            keep.scatter_(1, keep_idx, True)
            if self.timesteps_per_mask > 1:
                keep = keep.repeat_interleave(self.timesteps_per_mask, dim=0)
            return keep

        # Structure block masks
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

    def _sample_gaussian_noise(self, imgs: torch.Tensor, sigma: float) -> torch.Tensor:
        if sigma <= 0.0:
            return torch.zeros_like(imgs)
        return (sigma * torch.randn_like(imgs)).clamp_(-1.0, 1.0)

    def _apply_mask_corruption(
        self,
        base_imgs: torch.Tensor,
        visible_imgs: torch.Tensor,
        *,
        add_visible_noise: bool,
    ) -> torch.Tensor:
        keep = self._sample_keep_mask()
        keep = keep.view(self.timeframes_per_digit, self.patches_h, self.patches_w)
        keep = keep.repeat_interleave(self.patch_size, dim=1).repeat_interleave(self.patch_size, dim=2)
        keep = keep.unsqueeze(1)

        noise_sigma = self._resolve_schedule_value(self.noise_sigma)
        if add_visible_noise:
            visible = visible_imgs + self._sample_gaussian_noise(base_imgs, noise_sigma)
        else:
            visible = visible_imgs
        if self.masked_fill_mode == "random":
            fill = self._sample_gaussian_noise(base_imgs, noise_sigma)
        elif self.masked_fill_mode == "corrupted":
            fill_plan = self._sample_corruption_plan(
                self.fill_corruptions,
                self.fill_corruption_sampling,
                self.fill_subset_prob,
            )
            fill = self._apply_corruption_plan(base_imgs, fill_plan)
        else:
            fill = torch.full_like(base_imgs, self.masked_fill_value)
        return torch.where(keep, visible, fill).clamp_(self.signal_min_value, self.signal_max_value)

    def _apply_gaussian_corruption(self, imgs: torch.Tensor) -> torch.Tensor:
        sigma = self._resolve_schedule_value(self.noise_sigma)
        return imgs + self._sample_gaussian_noise(imgs, sigma)

    def _apply_mix_corruption(self, imgs: torch.Tensor) -> torch.Tensor:
        mix_alpha = self._resolve_schedule_value(self.mix_alpha)
        if mix_alpha <= 0.0:
            return imgs
        mix_noise_sigma = self._resolve_schedule_value(self.mix_noise_sigma)
        noise = self._sample_gaussian_noise(imgs, mix_noise_sigma)
        return imgs.lerp(noise, mix_alpha)

    def _apply_salt_pepper_corruption(self, imgs: torch.Tensor) -> torch.Tensor:
        prob = self._resolve_schedule_value(self.salt_pepper_prob)
        if prob <= 0.0:
            return imgs
        thresholds = torch.rand_like(imgs)
        salt = thresholds < (0.5 * prob)
        pepper = thresholds > (1.0 - 0.5 * prob)
        corrupted = torch.where(salt, self.signal_max_value, imgs)
        return torch.where(pepper, self.signal_min_value, corrupted)

    def _blur_kernel(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor | None:
        kernel_size = self._resolve_kernel_size()
        blur_sigma = self._resolve_schedule_value(self.blur_sigma)
        if kernel_size == 0 or blur_sigma <= 0.0:
            return None

        cache_key = (kernel_size, float(blur_sigma))
        kernel = self._blur_kernel_cache.get(cache_key)
        if kernel is None:
            coords = torch.arange(kernel_size, dtype=torch.float32)
            coords = coords - 0.5 * (kernel_size - 1)
            kernel_1d = torch.exp(-(coords.square()) / (2.0 * blur_sigma * blur_sigma))
            kernel_1d = kernel_1d / kernel_1d.sum()
            kernel_2d = torch.outer(kernel_1d, kernel_1d)
            kernel = kernel_2d.view(1, 1, kernel_size, kernel_size)
            self._blur_kernel_cache[cache_key] = kernel
        return kernel.to(device=device, dtype=dtype)

    def _apply_blur_corruption(self, imgs: torch.Tensor) -> torch.Tensor:
        kernel = self._blur_kernel(imgs.device, imgs.dtype)
        if kernel is None:
            return imgs
        channels = imgs.shape[1]
        kernel = kernel.expand(channels, 1, -1, -1)
        return F.conv2d(imgs, kernel, padding=kernel.shape[-1] // 2, groups=channels)

    def _apply_mix_blur_corruption(self, imgs: torch.Tensor) -> torch.Tensor:
        return self._apply_blur_corruption(self._apply_mix_corruption(imgs))

    def _apply_corruption(self, imgs: torch.Tensor, corruption: str) -> torch.Tensor:
        if corruption == "gaussian":
            return self._apply_gaussian_corruption(imgs)
        if corruption == "mix":
            return self._apply_mix_corruption(imgs)
        if corruption == "salt_pepper":
            return self._apply_salt_pepper_corruption(imgs)
        if corruption == "blur":
            return self._apply_blur_corruption(imgs)
        if corruption == "mix_blur":
            return self._apply_mix_blur_corruption(imgs)
        raise ValueError(f"Unsupported corruption: {corruption!r}")

    def _apply_corruption_plan(self, imgs: torch.Tensor, plan: tuple[str, ...]) -> torch.Tensor:
        corrupted = imgs
        for corruption in plan:
            corrupted = self._apply_corruption(corrupted, corruption)
        return corrupted.clamp(self.signal_min_value, self.signal_max_value)

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
        visible_plan = self._sample_corruption_plan(
            self.available_corruptions,
            self.corruption_sampling,
            self.corruption_subset_prob,
        )
        visible_imgs = self._apply_corruption_plan(img_t, visible_plan)
        if self.use_mask:
            masked_imgs = self._apply_mask_corruption(
                img_t,
                visible_imgs,
                add_visible_noise=self.visible_corrupt and len(visible_plan) == 0,
            )
        else:
            masked_imgs = visible_imgs
        masked_imgs = self._apply_image_visibility(masked_imgs)

        clean_sequence = img_t.clone()
        label_sequence = torch.full((self.timeframes_per_digit,), int(label), dtype=torch.long)
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
            masked_imgs = torch.cat(masked_parts, dim=0)
            clean_sequence = torch.cat(clean_parts, dim=0)
            label_sequence = torch.cat(label_parts, dim=0)
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
