################################################################################
# MIT License
#
# Copyright (c) 2022
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course | Autumn 2022
# Date Created: 2022-11-25
################################################################################

import torchvision
from torchvision import transforms
import torch
import torch.utils.data as data
from torch.utils.data import random_split
import numpy as np

def discretize(x, num_values):
    return (x * num_values).long().clamp_(max=num_values-1)

class DiscretizeTransform:
    '''
    Lambda was leading to picking issues on macOS
    '''
    def __init__(self, num_values: int):
        self.num_values = num_values

    def __call__(self, x):
        return discretize(x, self.num_values)

def mnist(root='../data/', batch_size=128, num_workers=4, download=True):
    """
    Returns data loaders for 4-bit MNIST dataset, i.e. values between 0 and 15.

    Inputs:
        root - Directory in which the MNIST dataset should be downloaded. It is better to
               use the same directory as the part2 of the assignment to prevent duplicate
               downloads.
        batch_size - Batch size to use for the data loaders
        num_workers - Number of workers to use in the data loaders.
        download - If True, MNIST is downloaded if it cannot be found in the specified
                   root directory.
    """
    data_transforms = transforms.Compose([transforms.ToTensor(),
                                          DiscretizeTransform(num_values=16)
                                          # Lambda was leading to picking issues on macOS
                                          # transforms.Lambda(lambda x: discretize(x, num_values=16))
                                        ])

    dataset = torchvision.datasets.MNIST(
        root, train=True, transform=data_transforms, download=download)
    test_set = torchvision.datasets.MNIST(
        root, train=False, transform=data_transforms, download=download)

    train_dataset, val_dataset = random_split(dataset,
                                              lengths=[54000, 6000],
                                              generator=torch.Generator().manual_seed(42))

    # Each data loader returns tuples of (img, label)
    # For the generative models we don't need the labels, which we need to take into account
    # when writing the train code.
    train_loader = data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
        pin_memory=True, persistent_workers=True)
    val_loader = data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=0,
        drop_last=False)
    test_loader = data.DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=0,
        drop_last=False)

    return train_loader, val_loader, test_loader

class InvertDiscretizeTransform:
    def __init__(self, num_values: int):
        self.num_values = num_values

    def __call__(self, x):
        return (self.num_values - 1) - discretize(x, self.num_values)

def inverse_mnist(root='../data/', batch_size=128, num_workers=4, download=True):
    '''
    Black digits on white background, MNIST for generalization testing purposes.
    Returns train / val / test data loaders.
    '''
    data_transforms = transforms.Compose([transforms.ToTensor(),
                                          InvertDiscretizeTransform(num_values=16)
                                        ])

    dataset = torchvision.datasets.MNIST(
        root, train=True, transform=data_transforms, download=download)
    test_set = torchvision.datasets.MNIST(
        root, train=False, transform=data_transforms, download=download)

    train_dataset, val_dataset = random_split(dataset,
                                              lengths=[54000, 6000],
                                              generator=torch.Generator().manual_seed(42))

    train_loader = data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
        pin_memory=True, persistent_workers=(num_workers > 0))
    val_loader = data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=0,
        drop_last=False)
    test_loader = data.DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=0,
        drop_last=False)

    return train_loader, val_loader, test_loader

def _get_targets(dataset):
    if isinstance(dataset, data.Subset):
        base_targets = _get_targets(dataset.dataset)
        if base_targets is None:
            return None
        return base_targets[torch.as_tensor(dataset.indices, dtype=torch.long)]

    targets = getattr(dataset, 'targets', None)
    if targets is None:
        return None
    return torch.as_tensor(targets, dtype=torch.long)


class _DigitDomainDataset(data.Dataset):
    def __init__(self, dataset, domain_label, include_digits=None):
        self.dataset = dataset
        self.domain_label = int(domain_label)
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
        return img, (int(digit_label), self.domain_label)


def _build_grayscale_level_lut(level_idx, n_levels, num_values):
    values = torch.arange(num_values, dtype=torch.float32)
    if n_levels <= 1:
        alpha = 0.0
    else:
        alpha = float(level_idx) / float(n_levels - 1)
    inverse_values = (num_values - 1) - values
    return torch.round((1.0 - alpha) * values + alpha * inverse_values).long()


class _DigitGrayscaleLevelDataset(data.Dataset):
    def __init__(self, dataset, level_label, n_levels, num_values=16, include_digits=None):
        self.dataset = dataset
        self.level_label = int(level_label)
        self.filtered_indices = None
        self.level_lut = _build_grayscale_level_lut(self.level_label, n_levels, num_values)

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
        img = self.level_lut[img.long()]
        return img, (int(digit_label), self.level_label)


def combine_grayscale_levels_mnist(
    mnist_loader,
    n_grayscale_levels,
    batch_size=None,
    shuffle=True,
    num_workers=None,
    drop_last=False,
    num_values=16,
    **kwargs
):
    """
    Combine multiple grayscale-level variants of MNIST into one data loader.

    Grayscale levels are interpolated from regular MNIST (level 0) to inverse MNIST
    (last level), with values quantized to the same 4-bit range.
    Each sample label is (digit_class, grayscale_level).

    Digit filters can be passed as keyword args:
        level_0_digits=[...], level_1_digits=[...], ..., level_N_digits=[...]
    If no kwargs are provided, all digits are included for all levels.
    """
    if n_grayscale_levels < 1:
        raise ValueError("n_grayscale_levels must be >= 1.")

    max_levels = int(num_values)
    n_grayscale_levels = min(int(n_grayscale_levels), max_levels)

    expected_keys = {f"level_{i}_digits" for i in range(n_grayscale_levels)}
    unknown_keys = set(kwargs.keys()) - expected_keys
    if unknown_keys:
        raise ValueError(f"Unknown kwargs: {sorted(unknown_keys)}")

    if kwargs:
        missing_keys = expected_keys - set(kwargs.keys())
        if missing_keys:
            raise ValueError(
                f"Missing digit filters: {sorted(missing_keys)}. "
                "Provide one list per grayscale level or pass no kwargs."
            )
    else:
        kwargs = {key: None for key in expected_keys}

    level_datasets = []
    for level_idx in range(n_grayscale_levels):
        level_digits = kwargs[f"level_{level_idx}_digits"]
        level_dataset = _DigitGrayscaleLevelDataset(
            mnist_loader.dataset,
            level_label=level_idx,
            n_levels=n_grayscale_levels,
            num_values=num_values,
            include_digits=level_digits
        )
        level_datasets.append(level_dataset)

    combined_dataset = data.ConcatDataset(level_datasets)

    if batch_size is None:
        batch_size = mnist_loader.batch_size
    if num_workers is None:
        num_workers = mnist_loader.num_workers

    pin_memory = bool(getattr(mnist_loader, "pin_memory", False))
    persistent_workers = bool(getattr(mnist_loader, "persistent_workers", False)) and num_workers > 0

    return data.DataLoader(
        combined_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers
    )


def combine_mnist_inverse_mnist(
    mnist_loader,
    inverse_mnist_loader,
    mnist_digits=None,
    inverse_mnist_digits=None,
    batch_size=None,
    shuffle=True,
    num_workers=None,
    drop_last=False
):
    '''
    Combine MNIST and inverse-MNIST loaders into one loader.
    Each sample has labels: (digit_class, domain_class), where domain_class is
    0 for regular MNIST and 1 for inverse MNIST.
    '''
    mnist_dataset = _DigitDomainDataset(
        mnist_loader.dataset, domain_label=0, include_digits=mnist_digits)
    inverse_dataset = _DigitDomainDataset(
        inverse_mnist_loader.dataset, domain_label=1, include_digits=inverse_mnist_digits)
    combined_dataset = data.ConcatDataset([mnist_dataset, inverse_dataset])

    if batch_size is None:
        batch_size = mnist_loader.batch_size
    if num_workers is None:
        num_workers = mnist_loader.num_workers

    pin_memory = bool(getattr(mnist_loader, "pin_memory", False))
    persistent_workers = bool(getattr(mnist_loader, "persistent_workers", False)) and num_workers > 0

    return data.DataLoader(
        combined_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers
    )
