"""Shared evaluation helpers."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ClassificationMetrics:
    accuracy: float
    precision: float
    recall: float
    f1: float
    confusion_matrix: list[list[int]]


def attack_success_rate(total_attacks: int, successful_attacks: int) -> float:
    if total_attacks == 0:
        return 0.0
    return successful_attacks / total_attacks


def binary_classification_metrics(
    y_true: list[int], y_pred: list[int]
) -> ClassificationMetrics:
    """Calculate binary metrics without requiring scikit-learn at runtime."""
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length.")

    tp = sum(1 for true, pred in zip(y_true, y_pred) if true == 1 and pred == 1)
    tn = sum(1 for true, pred in zip(y_true, y_pred) if true == 0 and pred == 0)
    fp = sum(1 for true, pred in zip(y_true, y_pred) if true == 0 and pred == 1)
    fn = sum(1 for true, pred in zip(y_true, y_pred) if true == 1 and pred == 0)
    total = len(y_true)

    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0.0
    )

    return ClassificationMetrics(
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        confusion_matrix=[[tn, fp], [fn, tp]],
    )


@torch.inference_mode()
def evaluate_classifier(model, dataloader, device: torch.device) -> ClassificationMetrics:
    """Evaluate a PyTorch classifier on a dataloader."""
    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        predictions = outputs.argmax(dim=1)
        y_true.extend(labels.cpu().tolist())
        y_pred.extend(predictions.cpu().tolist())

    return binary_classification_metrics(y_true, y_pred)
