"""Train the baseline chest X-ray classifier."""

import argparse
from pathlib import Path
from time import perf_counter

import torch

from src.data.dataset import RAW_DATA_DIR, create_dataloaders
from src.evaluation.metric_store import BASELINE_TRAINING_JSON, DEFAULT_METRICS_DIR, metrics_path, write_metrics_json
from src.evaluation.metrics import ClassificationMetrics, evaluate_classifier
from src.models.predict import DEFAULT_CHECKPOINT, build_resnet18


def calculate_class_weights(dataloader, num_classes: int, device: torch.device) -> torch.Tensor:
    """Calculate inverse-frequency class weights from the training dataloader."""
    counts = torch.zeros(num_classes, dtype=torch.float)
    for _, labels in dataloader:
        counts += torch.bincount(labels, minlength=num_classes).float()

    if torch.any(counts == 0):
        raise ValueError(f"Every class must appear in training data. Counts: {counts.tolist()}")

    weights = counts.sum() / (num_classes * counts)
    return weights.to(device)


def train_one_epoch(
    model,
    dataloader,
    criterion,
    optimizer,
    device: torch.device,
) -> float:
    """Train for one epoch and return average loss."""
    model.train()
    running_loss = 0.0
    sample_count = 0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        batch_size = images.size(0)
        running_loss += loss.item() * batch_size
        sample_count += batch_size

    return running_loss / sample_count


def save_checkpoint(
    checkpoint_path: Path,
    model,
    class_names: list[str],
    epoch: int,
    val_metrics: ClassificationMetrics,
    class_weights: list[float] | None = None,
    test_metrics: ClassificationMetrics | None = None,
) -> None:
    """Save model weights and useful metadata."""
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "class_names": class_names,
            "epoch": epoch,
            "class_weights": class_weights,
            "metrics": {
                "validation": val_metrics.__dict__,
                "test": test_metrics.__dict__ if test_metrics else None,
            },
        },
        checkpoint_path,
    )


def format_metrics(metrics: ClassificationMetrics) -> str:
    """Format metrics for console output."""
    return (
        f"accuracy={metrics.accuracy:.4f}, "
        f"precision={metrics.precision:.4f}, "
        f"recall={metrics.recall:.4f}, "
        f"f1={metrics.f1:.4f}, "
        f"confusion_matrix={metrics.confusion_matrix}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train baseline X-ray classifier.")
    parser.add_argument("--data-dir", type=Path, default=RAW_DATA_DIR)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--val-fraction", type=float, default=0.15)
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

    best_f1 = -1.0
    best_epoch = 0
    start_time = perf_counter()

    print(f"Device: {device}")
    print(f"Classes: {class_names}")
    print(f"Class weights: {class_weight_values if class_weight_values else 'disabled'}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Train batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_metrics = evaluate_classifier(model, val_loader, device)

        print(
            f"Epoch {epoch}/{args.epochs} "
            f"loss={train_loss:.4f} val_{format_metrics(val_metrics)}"
        )

        if val_metrics.f1 > best_f1:
            best_f1 = val_metrics.f1
            best_epoch = epoch
            save_checkpoint(
                args.checkpoint,
                model,
                class_names,
                epoch,
                val_metrics,
                class_weight_values,
            )
            print(f"Saved best checkpoint to {args.checkpoint}")

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_metrics = evaluate_classifier(model, test_loader, device)
    validation_metrics = ClassificationMetrics(**checkpoint["metrics"]["validation"])
    save_checkpoint(
        args.checkpoint,
        model,
        class_names,
        best_epoch,
        validation_metrics,
        class_weight_values,
        test_metrics,
    )
    metrics_output = metrics_path(args.metrics_dir, BASELINE_TRAINING_JSON)
    write_metrics_json(
        metrics_output,
        {
            "run_type": "baseline_training",
            "checkpoint": str(args.checkpoint),
            "model": "ResNet18",
            "pretrained": args.pretrained,
            "epochs": args.epochs,
            "best_epoch": best_epoch,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "val_fraction": args.val_fraction,
            "seed": args.seed,
            "class_names": class_names,
            "class_weights": class_weight_values,
            "validation": validation_metrics.__dict__,
            "test": test_metrics.__dict__,
        },
    )
    print(f"Metrics JSON written to {metrics_output}")

    elapsed_minutes = (perf_counter() - start_time) / 60
    print(f"Best epoch: {best_epoch}")
    print(f"Test {format_metrics(test_metrics)}")
    print(f"Finished in {elapsed_minutes:.2f} minutes")


if __name__ == "__main__":
    main()
