"""Dataset utilities for chest X-ray classification."""

from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms


RAW_DATA_DIR = Path("data/raw")
PROCESSED_DATA_DIR = Path("data/processed")
CLASS_NAMES = ("NORMAL", "PNEUMONIA")
IMAGE_SIZE = 224
IMAGE_NET_MEAN = (0.485, 0.456, 0.406)
IMAGE_NET_STD = (0.229, 0.224, 0.225)


def describe_expected_dataset() -> str:
    """Return a short note about the expected dataset layout."""
    return (
        "Place the selected chest X-ray dataset under data/raw/. "
        "A common layout is train/test folders with NORMAL/ and PNEUMONIA/."
    )


def train_transforms(image_size: int = IMAGE_SIZE) -> transforms.Compose:
    """Return transforms for training images."""
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=7),
            transforms.ToTensor(),
            transforms.Normalize(IMAGE_NET_MEAN, IMAGE_NET_STD),
        ]
    )


def eval_transforms(image_size: int = IMAGE_SIZE) -> transforms.Compose:
    """Return deterministic transforms for validation/test images."""
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGE_NET_MEAN, IMAGE_NET_STD),
        ]
    )


def validate_dataset_layout(data_dir: Path = RAW_DATA_DIR) -> None:
    """Raise a helpful error if the expected chest X-ray folders are missing."""
    required_dirs = [
        data_dir / split / class_name
        for split in ("train", "test")
        for class_name in CLASS_NAMES
    ]
    missing = [path for path in required_dirs if not path.exists()]
    if missing:
        missing_text = "\n".join(str(path) for path in missing)
        raise FileNotFoundError(
            "Dataset layout is incomplete. Missing required folders:\n"
            f"{missing_text}\n\n{describe_expected_dataset()}"
        )


def create_dataloaders(
    data_dir: Path = RAW_DATA_DIR,
    batch_size: int = 32,
    num_workers: int = 0,
    val_fraction: float = 0.15,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader, DataLoader, list[str]]:
    """Create train, validation, and test dataloaders.

    The Kaggle chest X-ray dataset has a very small built-in validation split,
    so this function creates validation data from the larger training folder and
    keeps the test folder untouched for final evaluation.
    """
    validate_dataset_layout(data_dir)

    train_dir = data_dir / "train"
    test_dir = data_dir / "test"

    full_train_for_split = datasets.ImageFolder(train_dir, transform=train_transforms())
    full_train_for_eval = datasets.ImageFolder(train_dir, transform=eval_transforms())
    test_dataset = datasets.ImageFolder(test_dir, transform=eval_transforms())

    if full_train_for_split.classes != list(CLASS_NAMES):
        raise ValueError(
            "Unexpected class folders. Expected "
            f"{list(CLASS_NAMES)}, got {full_train_for_split.classes}."
        )

    val_size = max(1, int(len(full_train_for_split) * val_fraction))
    train_size = len(full_train_for_split) - val_size
    generator = torch.Generator().manual_seed(seed)
    train_indices, val_indices = random_split(
        range(len(full_train_for_split)),
        [train_size, val_size],
        generator=generator,
    )

    train_dataset = Subset(full_train_for_split, train_indices.indices)
    val_dataset = Subset(full_train_for_eval, val_indices.indices)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader, full_train_for_split.classes
