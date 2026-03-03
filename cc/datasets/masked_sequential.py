import torch
import torch.utils.data as data


class MaskedSequentialDataset(data.Dataset):
    """
    Masked Sequential dataset where 
        image is divided into non-overlapping patches 
        and each patch is masked with a random binary mask.

    Args:
        dataset: Which dataset to use (eg. MNIST)
        patch_size: Size of each patch
        mask_ratio: Mask ratio (eg. 50% of patches masked)
        num_timeframes: The number of timeframes that model is shown 
            the different masked versions of the image 
            (each timestep different random patches are masked)
    
    
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
        num_timeframes: int,
        mask_pattern: str = "random",
    ):
        super().__init__()
        if patch_size <= 0:
            raise ValueError("patch_size must be > 0.")
        if not (0.0 <= mask_ratio <= 1.0):
            raise ValueError("mask_ratio must be in [0, 1].")
        if num_timeframes <= 0:
            raise ValueError("num_timeframes must be > 0.")
        if mask_pattern not in ("random", "structured"):
            raise ValueError("mask_pattern must be one of: 'random', 'structured'.")

        self.dataset = dataset
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio
        self.num_timeframes = num_timeframes
        self.mask_pattern = mask_pattern

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
        t = self.num_timeframes
        p = self.num_patches

        if self.num_keep == 0:
            return torch.zeros(t, p, dtype=torch.bool)
        if self.num_keep == p:
            return torch.ones(t, p, dtype=torch.bool)

        if self.mask_pattern == "random":
            scores = torch.rand(t, p)
            keep_idx = scores.topk(self.num_keep, dim=1, largest=True, sorted=False).indices
            keep = torch.zeros(t, p, dtype=torch.bool)
            keep.scatter_(1, keep_idx, True)
            return keep

        centers = torch.stack(
            (
                torch.rand(t) * (self.patches_h - 1),
                torch.rand(t) * (self.patches_w - 1),
            ),
            dim=1,
        )
        diffs = self.patch_coords.unsqueeze(0) - centers.unsqueeze(1)
        dists = diffs.square().sum(dim=-1)
        dists = dists + 1e-4 * torch.rand_like(dists)
        mask_idx = dists.topk(self.num_mask, dim=1, largest=False, sorted=False).indices
        keep = torch.ones(t, p, dtype=torch.bool)
        keep.scatter_(1, mask_idx, False)
        return keep

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        img, label = self.dataset[idx]
        keep = self._sample_keep_mask()
        keep = keep.view(self.num_timeframes, self.patches_h, self.patches_w)
        keep = keep.repeat_interleave(self.patch_size, dim=1).repeat_interleave(self.patch_size, dim=2)
        keep = keep.unsqueeze(1).to(dtype=img.dtype)

        masked_imgs = img.unsqueeze(0) * keep
        return masked_imgs, label