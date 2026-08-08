"""Train a classifier with FGSM adversarial training."""

import argparse
from pathlib import Path
from time import perf_counter

import torch

from src.attacks.fgsm import generate_fgsm_example
from src.data.dataset import RAW_DATA_DIR, create_dataloaders
from src.evaluation.metric_store import ADVERSARIAL_TRAINING_JSON, DEFAULT_METRICS_DIR, metrics_path, write_metrics_json
from src.evaluation.metrics import ClassificationMetrics, evaluate_classifier
from src.models.predict import build_resnet18
from src.models.train_baseline import (
    calculate_class_weights,
    format_metrics,
    save_checkpoint,
)


DEFAULT_ADV_CHECKPOINT = Path("artifacts/models/adversarial_model.pt")


def train_one_adversarial_epoch(
    model,
    dataloader,
    criterion,
    optimizer,
    device: torch.device,
    epsilon: float,
    adversarial_weight: float,
) -> float:
    """Train one epoch on a mix of clean and FGSM adversarial examples."""
    model.train()
    running_loss = 0.0
    sample_count = 0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        clean_outputs = model(images)
        clean_loss = criterion(clean_outputs, labels)

        adversarial_images = generate_fgsm_example(
            model, images, labels, epsilon=epsilon
        )
        model.train()
        adversarial_outputs = model(adversarial_images)
        adversarial_loss = criterion(adversarial_outputs, labels)

        loss = ((1.0 - adversarial_weight) * clean_loss) + (
            adversarial_weight * adversarial_loss
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        batch_size = images.size(0)
        running_loss += loss.item() * batch_size
        sample_count += batch_size

    return running_loss / sample_count


def evaluate_adversarial_accuracy(
    model,
    dataloader,
    device: torch.device,
    epsilon: float,
) -> float:
    """Evaluate accuracy after FGSM perturbation."""
    correct = 0
    total = 0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)
        adversarial_images = generate_fgsm_example(
            model, images, labels, epsilon=epsilon
        )
        outputs = model(adversarial_images)
        predictions = outputs.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    return correct / total if total else 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train with FGSM adversarial examples.")
    parser.add_argument("--data-dir", type=Path, default=RAW_DATA_DIR)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_ADV_CHECKPOINT)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--epsilon", type=float, default=0.01)
    parser.add_argument("--adversarial-weight", type=float, default=0.5)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--metrics-dir", type=Path, default=DEFAULT_METRICS_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--pretrained",
        action="store_true",
        help="Start from ImageNet pretrained ResNet18 weights.",
    )
    parser.add_argument(
        "--no-class-weights",
        action="store_true",
        help="Disable inverse-frequency class weighting.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, test_loader, class_names = create_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_fraction=args.val_fraction,
        seed=args.seed,
    )

    model = build_resnet18(
        num_classes=len(class_names), pretrained=args.pretrained
    ).to(device)

    class_weights = None
    class_weight_values = None
    if not args.no_class_weights:
        class_weights = calculate_class_weights(train_loader, len(class_names), device)
        class_weight_values = [round(value, 6) for value in class_weights.cpu().tolist()]

    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    best_score = -1.0
    best_epoch = 0
    start_time = perf_counter()

    print(f"Device: {device}")
    print(f"Classes: {class_names}")
    print(f"Class weights: {class_weight_values if class_weight_values else 'disabled'}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"FGSM epsilon: {args.epsilon}")
    print(f"Adversarial loss weight: {args.adversarial_weight}")
    print(f"Train batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_adversarial_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            args.epsilon,
            args.adversarial_weight,
        )
        val_metrics = evaluate_classifier(model, val_loader, device)
        val_adversarial_accuracy = evaluate_adversarial_accuracy(
            model, val_loader, device, args.epsilon
        )
        selection_score = (val_metrics.accuracy + val_adversarial_accuracy) / 2

        print(
            f"Epoch {epoch}/{args.epochs} "
            f"loss={train_loss:.4f} val_{format_metrics(val_metrics)} "
            f"val_adversarial_accuracy={val_adversarial_accuracy:.4f} "
            f"selection_score={selection_score:.4f}"
        )

        if selection_score > best_score:
            best_score = selection_score
            best_epoch = epoch
            save_checkpoint(
                args.checkpoint,
                model,
                class_names,
                epoch,
                val_metrics,
                class_weight_values,
            )
            print(f"Saved best adversarial checkpoint to {args.checkpoint}")

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_metrics = evaluate_classifier(model, test_loader, device)
    test_adversarial_accuracy = evaluate_adversarial_accuracy(
        model, test_loader, device, args.epsilon
    )
    save_checkpoint(
        args.checkpoint,
        model,
        class_names,
        best_epoch,
        ClassificationMetrics(**checkpoint["metrics"]["validation"]),
        class_weight_values,
        test_metrics,
    )

    validation_metrics = ClassificationMetrics(**checkpoint["metrics"]["validation"])
    metrics_output = metrics_path(args.metrics_dir, ADVERSARIAL_TRAINING_JSON)
    write_metrics_json(
        metrics_output,
        {
            "run_type": "adversarial_training",
            "checkpoint": str(args.checkpoint),
            "model": "ResNet18",
            "pretrained": args.pretrained,
            "epochs": args.epochs,
            "best_epoch": best_epoch,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "val_fraction": args.val_fraction,
            "epsilon": args.epsilon,
            "adversarial_weight": args.adversarial_weight,
            "seed": args.seed,
            "class_names": class_names,
            "class_weights": class_weight_values,
            "validation": validation_metrics.__dict__,
            "test": test_metrics.__dict__,
            "test_adversarial_accuracy": test_adversarial_accuracy,
        },
    )
    print(f"Metrics JSON written to {metrics_output}")
    elapsed_minutes = (perf_counter() - start_time) / 60
    print(f"Best epoch: {best_epoch}")
    print(f"Test {format_metrics(test_metrics)}")
    print(f"Test adversarial accuracy at epsilon {args.epsilon}: {test_adversarial_accuracy:.4f}")
    print(f"Finished in {elapsed_minutes:.2f} minutes")


if __name__ == "__main__":
    main()


