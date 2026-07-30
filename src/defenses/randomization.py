"""Defensive randomization utilities."""

from PIL import Image, ImageFilter
import torch
import torch.nn.functional as F


def apply_defensive_randomization(
    image: Image.Image,
    resize_delta: int = 12,
    blur_radius: float = 0.4,
) -> Image.Image:
    """Apply lightweight randomized preprocessing before PIL-based prediction."""
    image = image.convert("RGB")
    width, height = image.size
    resized = image.resize((width + resize_delta, height + resize_delta))
    restored = resized.resize((width, height))
    return restored.filter(ImageFilter.GaussianBlur(radius=blur_radius))


def apply_tensor_randomization(
    image_tensor: torch.Tensor,
    resize_delta: int = 12,
) -> torch.Tensor:
    """Apply differentiable resize randomization to normalized image tensors.

    The tensor shape is expected to be [B, C, H, W]. The output shape matches the
    input shape so it can be passed directly to the classifier.
    """
    if image_tensor.ndim != 4:
        raise ValueError("Expected image_tensor with shape [B, C, H, W].")

    _, _, height, width = image_tensor.shape
    resized = F.interpolate(
        image_tensor,
        size=(height + resize_delta, width + resize_delta),
        mode="bilinear",
        align_corners=False,
    )
    restored = F.interpolate(
        resized,
        size=(height, width),
        mode="bilinear",
        align_corners=False,
    )
    return restored
