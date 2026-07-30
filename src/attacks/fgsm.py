"""Fast Gradient Sign Method attack implementation."""

import torch

from src.data.dataset import IMAGE_NET_MEAN, IMAGE_NET_STD


def normalized_bounds(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Return valid min/max bounds for ImageNet-normalized tensors."""
    mean = torch.tensor(IMAGE_NET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGE_NET_STD, device=device).view(1, 3, 1, 1)
    lower = (0.0 - mean) / std
    upper = (1.0 - mean) / std
    return lower, upper


def clamp_normalized_image(image_tensor: torch.Tensor) -> torch.Tensor:
    """Clamp a normalized image tensor so it still represents valid pixels."""
    lower, upper = normalized_bounds(image_tensor.device)
    return torch.max(torch.min(image_tensor, upper), lower)


def generate_fgsm_example(
    model,
    image_tensor: torch.Tensor,
    label: torch.Tensor,
    epsilon: float = 0.03,
) -> torch.Tensor:
    """Generate an adversarial example with FGSM.

    Args:
        model: PyTorch classifier.
        image_tensor: Normalized input tensor with shape [B, C, H, W].
        label: Ground-truth label tensor with shape [B].
        epsilon: Perturbation size in normalized tensor space.
    """
    model.eval()
    perturbed = image_tensor.clone().detach().requires_grad_(True)
    output = model(perturbed)
    loss = torch.nn.functional.cross_entropy(output, label)
    model.zero_grad(set_to_none=True)
    loss.backward()
    adversarial = perturbed + epsilon * perturbed.grad.sign()
    return clamp_normalized_image(adversarial).detach()


def prediction_changed(clean_logits: torch.Tensor, adversarial_logits: torch.Tensor) -> bool:
    """Return True when the attack changes the predicted class."""
    clean_pred = clean_logits.argmax(dim=1)
    adversarial_pred = adversarial_logits.argmax(dim=1)
    return bool((clean_pred != adversarial_pred).any().item())
