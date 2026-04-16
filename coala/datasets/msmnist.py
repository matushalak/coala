import torch
import torch.utils.data as data
import torchvision
from torch.utils.data import random_split
from torchvision import transforms

from coala import DATADIR
from coala.datasets.masked_sequential import MaskedSequentialDataset

def msmnist(
    root: str = DATADIR,
    batch_size: int = 128,
    num_workers: int = 4,
    download: bool = True,
    patch_size: int = 4,
    mask_ratio: float = 0.5,
    mask_pattern: str = "random",
    masked_fill: str | float = 0.0,
    noise_sigma: float = 0.25,
    visible_corrupt:bool = False,
    number_of_masks: int = 100,
    timesteps_per_mask: int = 1,
    num_digits: int = 1,
    image_visibility: str = "all",
    accepted_digits: list[int] | None = None,
    target_type: str = "label",
):
    """
    Returns data loaders for Masked Sequential MNIST.

    Each data batch has shape:
        (batch_size, num_timeframes, channels, height, width),
    where num_timeframes = number_of_masks * timesteps_per_mask * num_digits.

    Targets are labels by default. Use target_type="image" to return the clean
    [-1, 1]-normalized image, or target_type="both" to return both the clean image
    and label in a dictionary. When num_digits > 1, clean targets are returned per
    timestep and labels are returned as per-timestep sequences.
    """
    data_transforms = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5,), std=(0.5,)),
        ]
    )

    dataset = torchvision.datasets.MNIST(
        root, train=True, transform=data_transforms, download=download
    )
    test_set = torchvision.datasets.MNIST(
        root, train=False, transform=data_transforms, download=download
    )

    if accepted_digits is not None:
        accepted_digits = sorted(set(accepted_digits))
        if len(accepted_digits) == 0:
            raise ValueError("accepted_digits must contain at least one digit when provided.")
        if any(d < 0 or d > 9 for d in accepted_digits):
            raise ValueError("accepted_digits values must be in [0, 9].")

        digits = torch.tensor(accepted_digits, dtype=dataset.targets.dtype)

        train_mask = (dataset.targets.unsqueeze(1) == digits.unsqueeze(0)).any(dim=1)
        train_idx = torch.nonzero(train_mask, as_tuple=False).squeeze(1)
        if train_idx.numel() == 0:
            raise ValueError(f"No training samples found for accepted_digits={accepted_digits}.")
        dataset = data.Subset(dataset, train_idx.tolist())

        test_mask = (test_set.targets.unsqueeze(1) == digits.unsqueeze(0)).any(dim=1)
        test_idx = torch.nonzero(test_mask, as_tuple=False).squeeze(1)
        if test_idx.numel() == 0:
            raise ValueError(f"No test samples found for accepted_digits={accepted_digits}.")
        test_set = data.Subset(test_set, test_idx.tolist())

    train_len = int(0.9 * len(dataset))
    if train_len == 0 and len(dataset) > 0:
        train_len = 1
    val_len = len(dataset) - train_len
    train_base, val_base = random_split(
        dataset, lengths=[train_len, val_len], generator=torch.Generator().manual_seed(42)
    )

    train_dataset = MaskedSequentialDataset(
        train_base,
        patch_size=patch_size,
        mask_ratio=mask_ratio,
        number_of_masks=number_of_masks,
        timesteps_per_mask=timesteps_per_mask,
        mask_pattern=mask_pattern,
        masked_fill=masked_fill,
        noise_sigma=noise_sigma,
        visible_corrupt=visible_corrupt,
        num_digits=num_digits,
        image_visibility=image_visibility,
        target_type=target_type,
    )
    val_dataset = MaskedSequentialDataset(
        val_base,
        patch_size=patch_size,
        mask_ratio=mask_ratio,
        number_of_masks=number_of_masks,
        timesteps_per_mask=timesteps_per_mask,
        mask_pattern=mask_pattern,
        masked_fill=masked_fill,
        noise_sigma=noise_sigma,
        visible_corrupt=visible_corrupt,
        num_digits=num_digits,
        image_visibility=image_visibility,
        target_type=target_type,
    )
    test_dataset = MaskedSequentialDataset(
        test_set,
        patch_size=patch_size,
        mask_ratio=mask_ratio,
        number_of_masks=number_of_masks,
        timesteps_per_mask=timesteps_per_mask,
        mask_pattern=mask_pattern,
        masked_fill=masked_fill,
        noise_sigma=noise_sigma,
        visible_corrupt=visible_corrupt,
        num_digits=num_digits,
        image_visibility=image_visibility,
        target_type=target_type,
    )

    train_loader = data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )
    val_loader = data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )
    test_loader = data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader

# visualize a few examples of the dataset
def visualize_msmnist_examples(
    num_examples: int = 4,
    mask_ratio: float = 0.5,
    masked_fill: str | float = 0.0,
    noise_sigma: float = 0.25,
    visible_corrupt: bool = False,
    patch_size:int = 4,
    number_of_masks: int = 100,
    timesteps_per_mask: int = 1,
    num_digits: int = 1,
    image_visibility: str = "all",
    accepted_digits: list[int] | None = None,
    target_type: str = "label",
    show: bool = True,
) -> tuple[torch.Tensor, torch.Tensor | int | dict[str, torch.Tensor]]:
    import matplotlib.pyplot as plt
    train_loader, _, _ = msmnist(
        batch_size=num_examples,
        num_workers=0,
        mask_ratio=mask_ratio,
        masked_fill=masked_fill,
        noise_sigma=noise_sigma,
        visible_corrupt=visible_corrupt,
        number_of_masks=number_of_masks,
        timesteps_per_mask=timesteps_per_mask,
        num_digits=num_digits,
        image_visibility=image_visibility,
        patch_size=patch_size,
        accepted_digits=accepted_digits,
        target_type=target_type,
    )
    batch = next(iter(train_loader))
    masked_imgs, targets = batch
    # masked_imgs shape: (batch_size, num_timeframes, channels, height, width)
    if show:
        batch_size, total_t, channels, height, width = masked_imgs.shape
        vmin, vmax = masked_imgs.min().item(), masked_imgs.max().item()
        padding = 2
        frames = masked_imgs.reshape(batch_size * total_t, channels, height, width)
        grid_img = torchvision.utils.make_grid(
            frames,
            nrow=total_t,
            normalize=False,
            padding=padding,
            pad_value=(vmax - vmin) / 2,
        )
        fig, ax = plt.subplots(figsize=(2 * total_t, 2))
        if channels == 1:
            ax.imshow(grid_img[0], cmap="gray", vmin=vmin, vmax=vmax, interpolation="nearest")
        else:
            ax.imshow(grid_img.permute(1, 2, 0), interpolation="nearest")
        fig.tight_layout()
        plt.show()

    return masked_imgs, targets

if __name__ == "__main__":
    visualize_msmnist_examples()
