import torch
from torch.utils.data import DataLoader
from torchvision.utils import make_grid


def _extract_images(batch_or_sample):
    if isinstance(batch_or_sample, (tuple, list)):
        return batch_or_sample[0]
    return batch_or_sample


def _to_bchw(images: torch.Tensor) -> torch.Tensor:
    if images.dim() == 4:
        return images
    if images.dim() == 3:
        return images.unsqueeze(1)
    if images.dim() == 2:
        return images.unsqueeze(0).unsqueeze(0)
    raise ValueError(f"Unsupported image tensor shape: {tuple(images.shape)}")


def visualize_dataset(dataset, n_examples=5):
    n_examples = max(1, int(n_examples))

    if isinstance(dataset, DataLoader):
        images = _extract_images(next(iter(dataset)))[:n_examples]
    else:
        count = min(len(dataset), n_examples)
        if count == 0:
            raise ValueError("Dataset is empty.")
        images = [_extract_images(dataset[i]) for i in range(count)]
        images = torch.stack(images, dim=0)

    images = _to_bchw(images).float()

    max_value = images.max().item()
    if max_value > 1.0:
        images = images / max(1.0, max_value)

    nrow = min(images.shape[0], 8, int(n_examples**0.5))
    return make_grid(images, nrow=nrow, padding=2, normalize=False)
