"""Evaluate FGSM attack success against a trained checkpoint."""

import argparse
from pathlib import Path

import torch

from src.attacks.fgsm import generate_fgsm_example
from src.data.dataset import RAW_DATA_DIR, create_dataloaders
from src.models.predict import DEFAULT_CHECKPOINT, load_model


def safe_ratio(numerator: int, denominator: int) -> float:
    """Return numerator / denominator, guarding against zero."""
    return numerator / denominator if denominator else 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate FGSM attack success.")
    parser.add_argument("--data-dir", type=Path, default=RAW_DATA_DIR)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epsilon", type=float, default=0.03)
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
    prediction_changed = 0
    attack_candidates = 0
    successful_attacks = 0
    class_totals = [0 for _ in class_names]
    class_clean_correct = [0 for _ in class_names]
    class_successful_attacks = [0 for _ in class_names]

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

        clean_is_correct = clean_predictions == labels
        adversarial_is_correct = adversarial_predictions == labels
        became_wrong = clean_is_correct & (~adversarial_is_correct)

        total += labels.size(0)
        clean_correct += clean_is_correct.sum().item()
        adversarial_correct += adversarial_is_correct.sum().item()
        prediction_changed += (clean_predictions != adversarial_predictions).sum().item()
        attack_candidates += clean_is_correct.sum().item()
        successful_attacks += became_wrong.sum().item()

        for class_index in range(len(class_names)):
            class_mask = labels == class_index
            class_totals[class_index] += class_mask.sum().item()
            class_clean_correct[class_index] += (clean_is_correct & class_mask).sum().item()
            class_successful_attacks[class_index] += (became_wrong & class_mask).sum().item()

    clean_accuracy = safe_ratio(clean_correct, total)
    adversarial_accuracy = safe_ratio(adversarial_correct, total)
    prediction_change_rate = safe_ratio(prediction_changed, total)
    standard_attack_success_rate = safe_ratio(successful_attacks, attack_candidates)

    print(f"Device: {device}")
    print(f"Classes: {class_names}")
    print(f"Evaluated samples: {total}")
    print(f"Epsilon: {args.epsilon}")
    print(f"Clean accuracy: {clean_accuracy:.4f}")
    print(f"Adversarial accuracy: {adversarial_accuracy:.4f}")
    print(f"Prediction change rate: {prediction_change_rate:.4f}")
    print(f"Attack candidates clean-correct: {attack_candidates}")
    print(f"Successful attacks clean-correct-to-wrong: {successful_attacks}")
    print(f"Standard attack success rate: {standard_attack_success_rate:.4f}")

    print("Per-class attack success:")
    for class_index, class_name in enumerate(class_names):
        print(
            f"  {class_name}: "
            f"clean_correct={class_clean_correct[class_index]}/"
            f"{class_totals[class_index]}, "
            f"successful_attacks={class_successful_attacks[class_index]}, "
            f"success_rate="
            f"{safe_ratio(class_successful_attacks[class_index], class_clean_correct[class_index]):.4f}"
        )

    print(
        "CSV summary: "
        "epsilon,total,clean_accuracy,adversarial_accuracy,"
        "prediction_change_rate,standard_attack_success_rate"
    )
    print(
        "CSV values: "
        f"{args.epsilon},{total},{clean_accuracy:.4f},{adversarial_accuracy:.4f},"
        f"{prediction_change_rate:.4f},{standard_attack_success_rate:.4f}"
    )


if __name__ == "__main__":
    main()
