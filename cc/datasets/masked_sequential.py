import torch
import torch.utils.data as data


RANDOM_FILL_STD = 0.5


class MaskedSequentialDataset(data.Dataset):
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
        target_type: str = "label",
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
        if mask_pattern not in ("random", "structured"):
            raise ValueError("mask_pattern must be one of: 'random', 'structured'.")
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
        if target_type not in ("label", "image"):
            raise ValueError("target_type must be one of: 'label', 'image'.")

        self.dataset = dataset
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio
        self.number_of_masks = number_of_masks
        self.timesteps_per_mask = timesteps_per_mask
        self.num_timeframes = number_of_masks * timesteps_per_mask
        self.mask_pattern = mask_pattern
        self.target_type = target_type

        sample_img, _ = self.dataset[0]
        if sample_img.dim() != 3:
            raise ValueError("Expected images in (channels, height, width) format.")
        _, self.height, self.width = sample_img.shape
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

    def __len__(self) -> int:
        return len(self.dataset)

    def _sample_keep_mask(self) -> torch.BoolTensor:
        m = self.number_of_masks
        t = self.num_timeframes
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
        mask_idx = dists.topk(self.num_mask, dim=1, largest=False, sorted=False).indices
        keep = torch.ones(m, p, dtype=torch.bool)
        keep.scatter_(1, mask_idx, False)
        if self.timesteps_per_mask > 1:
            keep = keep.repeat_interleave(self.timesteps_per_mask, dim=0)
        return keep

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int | torch.Tensor]:
        img, label = self.dataset[idx]
        keep = self._sample_keep_mask()
        keep = keep.view(self.num_timeframes, self.patches_h, self.patches_w)
        keep = keep.repeat_interleave(self.patch_size, dim=1).repeat_interleave(self.patch_size, dim=2)
        keep = keep.unsqueeze(1)

        img_t = img.unsqueeze(0)
        if self.masked_fill_mode == "random":
            img_t = img_t.expand(self.num_timeframes, -1, -1, -1)
            noise = (RANDOM_FILL_STD * torch.randn_like(img_t)).clamp_(-1.0, 1.0)
            masked_imgs = torch.where(keep, img_t+noise, noise)
        elif self.masked_fill_value == 0.0:
            masked_imgs = img_t * keep.to(dtype=img.dtype)
        else:
            img_t = img_t.expand(self.num_timeframes, -1, -1, -1)
            masked_imgs = torch.where(keep, img_t, self.masked_fill_value)
        target = label if self.target_type == "label" else img
        return masked_imgs, target
