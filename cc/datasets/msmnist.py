import torch
import torch.utils.data as data
import torchvision
from torch.utils.data import random_split
from torchvision import transforms

from cc import DATADIR
from cc.datasets.masked_sequential import MaskedSequentialDataset

def msmnist(
    root: str = DATADIR,
    batch_size: int = 128,
    num_workers: int = 4,
    download: bool = True,
    patch_size: int = 4,
    mask_ratio: float = 0.5,
    mask_pattern: str = "random",
    num_timeframes: int = 4,
):
    """
    Returns data loaders for Masked Sequential MNIST.

    Each data batch has shape:
        (batch_size, num_timeframes, channels, height, width)
    """
    mean = (0.1307,)
    std = (0.3081,)
    data_transforms = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )

    dataset = torchvision.datasets.MNIST(
        root, train=True, transform=data_transforms, download=download
    )
    test_set = torchvision.datasets.MNIST(
        root, train=False, transform=data_transforms, download=download
    )

    train_base, val_base = random_split(
        dataset, lengths=[54000, 6000], generator=torch.Generator().manual_seed(42)
    )

    train_dataset = MaskedSequentialDataset(
        train_base,
        patch_size=patch_size,
        mask_ratio=mask_ratio,
        num_timeframes=num_timeframes,
        mask_pattern=mask_pattern,
    )
    val_dataset = MaskedSequentialDataset(
        val_base,
        patch_size=patch_size,
        mask_ratio=mask_ratio,
        num_timeframes=num_timeframes,
        mask_pattern=mask_pattern,
    )
    test_dataset = MaskedSequentialDataset(
        test_set,
        patch_size=patch_size,
        mask_ratio=mask_ratio,
        num_timeframes=num_timeframes,
        mask_pattern=mask_pattern,
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
def visualize_msmnist_examples(num_examples: int = 4, mask_ratio: float = 0.5, num_timeframes: int = 25):
    import matplotlib.pyplot as plt

    train_loader, _, _ = msmnist(batch_size=num_examples, num_workers=0, mask_ratio=mask_ratio, num_timeframes=num_timeframes)
    batch = next(iter(train_loader))
    masked_imgs, labels = batch
    # masked_imgs shape: (batch_size, num_timeframes, channels, height, width)
    fig, axes = plt.subplots(num_examples, masked_imgs.shape[1], figsize=(10, 2 * num_examples))
    for i in range(num_examples):
        for t in range(masked_imgs.shape[1]):
            img = masked_imgs[i, t].squeeze(0).cpu()  # shape: (height, width)
            axes[i, t].imshow(img, cmap='gray', vmin=masked_imgs.min().item(), vmax=masked_imgs.max().item())
            axes[i, t].axis('off')
            if t == 0:
                axes[i, t].set_title(f"Label: {labels[i].item()}")
    fig.tight_layout()
    plt.show()

    return masked_imgs, labels

if __name__ == "__main__":
    visualize_msmnist_examples()