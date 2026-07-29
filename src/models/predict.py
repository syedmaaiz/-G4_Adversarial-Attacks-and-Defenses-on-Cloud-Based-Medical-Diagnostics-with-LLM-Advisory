"""Prediction helpers for the trained classifier."""

from pathlib import Path


DEFAULT_CHECKPOINT = Path("artifacts/models/baseline_model.pt")


def load_model(checkpoint_path: Path = DEFAULT_CHECKPOINT):
    """Load a trained model checkpoint.

    Implement this after the baseline architecture is finalized.
    """
    raise NotImplementedError("Model loading will be implemented after training.")

