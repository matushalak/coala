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
from coala import DATADIR

def cifar10(root=DATADIR, batch_size=128, num_workers=4, download=True):
    """
    Returns data loaders for CIFAR-10 with pixel values in [-1, 1].

    Inputs:
        root - Directory where CIFAR-10 is downloaded/stored.
        batch_size - Batch size to use for the data loaders.
        num_workers - Number of workers to use in the data loaders.
        download - If True, CIFAR-10 is downloaded if not found in root.
    """
    data_transforms = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ]
    )

    dataset = torchvision.datasets.CIFAR10(
        root, train=True, transform=data_transforms, download=download
    )
    test_set = torchvision.datasets.CIFAR10(
        root, train=False, transform=data_transforms, download=download
    )

    train_dataset, val_dataset = random_split(
        dataset,
        lengths=[45000, 5000],
        generator=torch.Generator().manual_seed(42),
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
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader
