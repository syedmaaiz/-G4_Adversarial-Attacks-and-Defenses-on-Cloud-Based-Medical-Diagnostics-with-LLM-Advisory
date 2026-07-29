"""Prediction helpers for the trained classifier."""

from pathlib import Path

import torch
from PIL import Image
from torchvision import models

from src.data.dataset import eval_transforms


DEFAULT_CHECKPOINT = Path("artifacts/models/baseline_model.pt")


def build_resnet18(num_classes: int = 2, pretrained: bool = True):
    """Build the baseline ResNet18 model."""
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    return model


def load_model(
    checkpoint_path: Path = DEFAULT_CHECKPOINT,
    device: torch.device | None = None,
):
    """Load a trained model checkpoint.

    Returns the model, class names, and any saved training metadata.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    class_names = checkpoint.get("class_names", ["NORMAL", "PNEUMONIA"])
    model = build_resnet18(num_classes=len(class_names), pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, class_names, checkpoint.get("metrics", {})


@torch.inference_mode()
def predict_image(
    model,
    image: Image.Image,
    class_names: list[str],
    device: torch.device,
) -> dict[str, float | str | list[float]]:
    """Predict the class of one PIL image."""
    image = image.convert("RGB")
    tensor = eval_transforms()(image).unsqueeze(0).to(device)
    logits = model(tensor)
    probabilities = torch.softmax(logits, dim=1).squeeze(0)
    confidence, class_index = torch.max(probabilities, dim=0)
    return {
        "predicted_class": class_names[class_index.item()],
        "confidence": confidence.item(),
        "probabilities": probabilities.cpu().tolist(),
    }
