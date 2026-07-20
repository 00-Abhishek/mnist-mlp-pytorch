"""Dataset preparation utilities for MNIST."""

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from src.config import BATCH_SIZE, DATA_DIR, SEED, VALIDATION_SPLIT


def get_transform():
    """Return the image transform used for MNIST samples."""
    return transforms.ToTensor()


def load_dataset(train: bool = True):
    """Download if needed and return the MNIST dataset."""
    return datasets.MNIST(
        root=str(DATA_DIR),
        train=train,
        download=True,
        transform=get_transform(),
    )


def split_dataset(dataset):
    """Split a dataset into reproducible train and validation subsets."""
    validation_size = int(len(dataset) * VALIDATION_SPLIT)
    train_size = len(dataset) - validation_size

    return random_split(
        dataset,
        [train_size, validation_size],
        generator=torch.Generator().manual_seed(SEED),
    )


def create_dataloaders(batch_size: int = BATCH_SIZE):
    """Create train and validation dataloaders."""
    dataset = load_dataset(train=True)
    train_dataset, validation_dataset = split_dataset(dataset)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    return train_loader, validation_loader
