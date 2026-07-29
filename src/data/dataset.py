"""Dataset utilities for chest X-ray classification."""

from pathlib import Path


RAW_DATA_DIR = Path("data/raw")
PROCESSED_DATA_DIR = Path("data/processed")


def describe_expected_dataset() -> str:
    """Return a short note about the expected dataset layout."""
    return (
        "Place the selected chest X-ray dataset under data/raw/. "
        "A common layout is one folder per class, such as Normal/ and Pneumonia/."
    )

