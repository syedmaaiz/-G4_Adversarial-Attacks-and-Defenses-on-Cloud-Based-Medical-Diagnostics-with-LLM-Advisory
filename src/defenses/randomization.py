"""Defensive randomization utilities."""

from PIL import Image, ImageFilter


def apply_defensive_randomization(
    image: Image.Image,
    resize_delta: int = 12,
    blur_radius: float = 0.4,
) -> Image.Image:
    """Apply lightweight randomized preprocessing before prediction.

    This defense makes small adversarial perturbations less stable by slightly
    resizing and smoothing the input before the classifier sees it.
    """
    image = image.convert("RGB")
    width, height = image.size
    resized = image.resize((width + resize_delta, height + resize_delta))
    restored = resized.resize((width, height))
    return restored.filter(ImageFilter.GaussianBlur(radius=blur_radius))
