"""Evaluate FGSM attack success against a trained checkpoint."""

import argparse
from pathlib import Path

import torch

from src.attacks.fgsm import generate_fgsm_example
from src.data.dataset import RAW_DATA_DIR, create_dataloaders
from src.evaluation.metrics import attack_success_rate
from src.models.predict import DEFAULT_CHECKPOINT, load_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate FGSM attack success.")
    parser.add_argument("--data-dir", type=Path, default=RAW_DATA_DIR)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epsilon", type=float, default=0.03)
    parser.add_argument("--max-batches", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, class_names, _ = load_model(args.checkpoint, device)
    _, _, test_loader, _ = create_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    total = 0
    successful = 0
    clean_correct = 0
    adversarial_correct = 0

    for batch_index, (images, labels) in enumerate(test_loader, start=1):
        if args.max_batches and batch_index > args.max_batches:
            break

        images = images.to(device)
        labels = labels.to(device)

        with torch.inference_mode():
            clean_logits = model(images)
            clean_predictions = clean_logits.argmax(dim=1)

        adversarial_images = generate_fgsm_example(
            model, images, labels, epsilon=args.epsilon
        )
        with torch.inference_mode():
            adversarial_logits = model(adversarial_images)
            adversarial_predictions = adversarial_logits.argmax(dim=1)

        clean_correct += (clean_predictions == labels).sum().item()
        adversarial_correct += (adversarial_predictions == labels).sum().item()
        successful += (clean_predictions != adversarial_predictions).sum().item()
        total += labels.size(0)

    print(f"Classes: {class_names}")
    print(f"Evaluated samples: {total}")
    print(f"Epsilon: {args.epsilon}")
    print(f"Clean accuracy: {clean_correct / total:.4f}")
    print(f"Adversarial accuracy: {adversarial_correct / total:.4f}")
    print(f"Attack success rate: {attack_success_rate(total, successful):.4f}")


if __name__ == "__main__":
    main()
