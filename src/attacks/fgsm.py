"""Fast Gradient Sign Method attack implementation."""

import torch


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
    return adversarial.detach()


def prediction_changed(clean_logits: torch.Tensor, adversarial_logits: torch.Tensor) -> bool:
    """Return True when the attack changes the predicted class."""
    clean_pred = clean_logits.argmax(dim=1)
    adversarial_pred = adversarial_logits.argmax(dim=1)
    return bool((clean_pred != adversarial_pred).any().item())
