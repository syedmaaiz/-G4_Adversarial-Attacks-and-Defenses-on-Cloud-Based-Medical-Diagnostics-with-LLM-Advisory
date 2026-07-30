"""Evaluate defensive randomization against FGSM adversarial examples."""

import argparse
from pathlib import Path

import torch

from src.attacks.fgsm import generate_fgsm_example
from src.data.dataset import RAW_DATA_DIR, create_dataloaders
from src.defenses.randomization import apply_tensor_randomization
from src.models.predict import DEFAULT_CHECKPOINT, load_model


def safe_ratio(numerator: int, denominator: int) -> float:
    """Return numerator / denominator, guarding against zero."""
    return numerator / denominator if denominator else 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate defensive randomization against FGSM."
    )
    parser.add_argument("--data-dir", type=Path, default=RAW_DATA_DIR)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epsilon", type=float, default=0.01)
    parser.add_argument("--resize-delta", type=int, default=12)
    parser.add_argument(
        "--max-batches",
        type=int,
        default=0,
        help="Limit evaluated batches for smoke tests. Use 0 for the full test set.",
    )
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
    clean_correct = 0
    adversarial_correct = 0
    defended_clean_correct = 0
    defended_adversarial_correct = 0
    attack_candidates = 0
    successful_attacks = 0
    defended_recoveries = 0

    class_totals = [0 for _ in class_names]
    class_clean_correct = [0 for _ in class_names]
    class_adversarial_correct = [0 for _ in class_names]
    class_defended_adversarial_correct = [0 for _ in class_names]

    for batch_index, (images, labels) in enumerate(test_loader, start=1):
        if args.max_batches and batch_index > args.max_batches:
            break

        images = images.to(device)
        labels = labels.to(device)

        with torch.inference_mode():
            clean_predictions = model(images).argmax(dim=1)
            defended_clean_predictions = model(
                apply_tensor_randomization(images, resize_delta=args.resize_delta)
            ).argmax(dim=1)

        adversarial_images = generate_fgsm_example(
            model, images, labels, epsilon=args.epsilon
        )
        defended_adversarial_images = apply_tensor_randomization(
            adversarial_images, resize_delta=args.resize_delta
        )

        with torch.inference_mode():
            adversarial_predictions = model(adversarial_images).argmax(dim=1)
            defended_adversarial_predictions = model(defended_adversarial_images).argmax(dim=1)

        clean_is_correct = clean_predictions == labels
        adversarial_is_correct = adversarial_predictions == labels
        defended_adversarial_is_correct = defended_adversarial_predictions == labels
        became_wrong = clean_is_correct & (~adversarial_is_correct)
        recovered = became_wrong & defended_adversarial_is_correct

        total += labels.size(0)
        clean_correct += clean_is_correct.sum().item()
        adversarial_correct += adversarial_is_correct.sum().item()
        defended_clean_correct += (defended_clean_predictions == labels).sum().item()
        defended_adversarial_correct += defended_adversarial_is_correct.sum().item()
        attack_candidates += clean_is_correct.sum().item()
        successful_attacks += became_wrong.sum().item()
        defended_recoveries += recovered.sum().item()

        for class_index in range(len(class_names)):
            class_mask = labels == class_index
            class_totals[class_index] += class_mask.sum().item()
            class_clean_correct[class_index] += (clean_is_correct & class_mask).sum().item()
            class_adversarial_correct[class_index] += (
                adversarial_is_correct & class_mask
            ).sum().item()
            class_defended_adversarial_correct[class_index] += (
                defended_adversarial_is_correct & class_mask
            ).sum().item()

    clean_accuracy = safe_ratio(clean_correct, total)
    adversarial_accuracy = safe_ratio(adversarial_correct, total)
    defended_clean_accuracy = safe_ratio(defended_clean_correct, total)
    defended_adversarial_accuracy = safe_ratio(defended_adversarial_correct, total)
    attack_success = safe_ratio(successful_attacks, attack_candidates)
    recovery_rate = safe_ratio(defended_recoveries, successful_attacks)

    print(f"Device: {device}")
    print(f"Classes: {class_names}")
    print(f"Evaluated samples: {total}")
    print(f"Epsilon: {args.epsilon}")
    print(f"Resize delta: {args.resize_delta}")
    print(f"Clean accuracy: {clean_accuracy:.4f}")
    print(f"Defended clean accuracy: {defended_clean_accuracy:.4f}")
    print(f"Adversarial accuracy: {adversarial_accuracy:.4f}")
    print(f"Defended adversarial accuracy: {defended_adversarial_accuracy:.4f}")
    print(f"Standard attack success rate: {attack_success:.4f}")
    print(f"Defense recovery rate: {recovery_rate:.4f}")
    print(f"Recovered successful attacks: {defended_recoveries}/{successful_attacks}")

    print("Per-class accuracy clean -> adversarial -> defended adversarial:")
    for class_index, class_name in enumerate(class_names):
        print(
            f"  {class_name}: "
            f"{safe_ratio(class_clean_correct[class_index], class_totals[class_index]):.4f} -> "
            f"{safe_ratio(class_adversarial_correct[class_index], class_totals[class_index]):.4f} -> "
            f"{safe_ratio(class_defended_adversarial_correct[class_index], class_totals[class_index]):.4f}"
        )

    print(
        "CSV summary: "
        "epsilon,resize_delta,total,clean_accuracy,defended_clean_accuracy,"
        "adversarial_accuracy,defended_adversarial_accuracy,"
        "standard_attack_success_rate,defense_recovery_rate"
    )
    print(
        "CSV values: "
        f"{args.epsilon},{args.resize_delta},{total},{clean_accuracy:.4f},"
        f"{defended_clean_accuracy:.4f},{adversarial_accuracy:.4f},"
        f"{defended_adversarial_accuracy:.4f},{attack_success:.4f},{recovery_rate:.4f}"
    )


if __name__ == "__main__":
    main()
