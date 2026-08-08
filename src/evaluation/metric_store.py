"""Save and load structured experiment metrics for reports and dashboards."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


DEFAULT_METRICS_DIR = Path("reports/metrics")
BASELINE_TRAINING_JSON = "baseline_training.json"
ADVERSARIAL_TRAINING_JSON = "adversarial_training.json"
FGSM_BASELINE_JSON = "fgsm_baseline.json"
FGSM_ADVERSARIAL_JSON = "fgsm_adversarial.json"
RANDOMIZATION_JSON = "randomization.json"


def write_metrics_json(path: Path, payload: dict[str, Any]) -> None:
    """Write metrics in a stable JSON format."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def read_metrics_json(path: Path) -> dict[str, Any] | None:
    """Read a metrics JSON file, returning None when it does not exist."""
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def fgsm_output_name(checkpoint_path: Path) -> str:
    """Choose a conventional output filename for an FGSM evaluation."""
    if "adversarial" in checkpoint_path.stem.lower():
        return FGSM_ADVERSARIAL_JSON
    return FGSM_BASELINE_JSON


def metrics_path(metrics_dir: Path, filename: str) -> Path:
    return metrics_dir / filename


def _round_or_none(value: Any) -> float | None:
    if value is None:
        return None
    return round(float(value), 6)


def build_advisory_metrics_from_files(
    mode: str,
    metrics_dir: Path = DEFAULT_METRICS_DIR,
) -> dict[str, Any] | None:
    """Build advisor metrics from saved run JSON files when available."""
    baseline_fgsm = read_metrics_json(metrics_path(metrics_dir, FGSM_BASELINE_JSON))
    if not baseline_fgsm:
        return None

    attack = baseline_fgsm.get("attack", {})
    per_class = attack.get("per_class", {})
    metrics: dict[str, Any] = {
        "advisor_mode": mode,
        "dataset": "Chest X-ray classification: NORMAL vs PNEUMONIA",
        "model": "ResNet18 image classifier",
        "attack": {
            "name": "FGSM",
            "epsilon": attack.get("epsilon"),
            "clean_accuracy_baseline": _round_or_none(attack.get("clean_accuracy")),
            "adversarial_accuracy_baseline": _round_or_none(attack.get("adversarial_accuracy")),
            "prediction_change_rate_baseline": _round_or_none(attack.get("prediction_change_rate")),
            "standard_attack_success_rate_baseline": _round_or_none(attack.get("standard_attack_success_rate")),
            "per_class_attack_success_baseline": {
                class_name: _round_or_none(values.get("success_rate"))
                for class_name, values in per_class.items()
            },
        },
    }

    if mode == "pre-defense":
        return metrics

    randomization = read_metrics_json(metrics_path(metrics_dir, RANDOMIZATION_JSON))
    adversarial_fgsm = read_metrics_json(metrics_path(metrics_dir, FGSM_ADVERSARIAL_JSON))
    adversarial_training = read_metrics_json(metrics_path(metrics_dir, ADVERSARIAL_TRAINING_JSON))
    if not (randomization and (adversarial_fgsm or adversarial_training)):
        return None

    randomization_eval = randomization.get("randomization", {})
    adversarial_attack = (adversarial_fgsm or {}).get("attack", {})
    adversarial_per_class = adversarial_attack.get("per_class", {})
    adversarial_test = (adversarial_training or {}).get("test", {})

    metrics["advisor_mode"] = "post-defense"
    metrics["defenses"] = {
        "defensive_randomization": {
            "resize_delta": randomization_eval.get("resize_delta"),
            "defended_clean_accuracy": _round_or_none(randomization_eval.get("defended_clean_accuracy")),
            "defended_adversarial_accuracy": _round_or_none(randomization_eval.get("defended_adversarial_accuracy")),
            "defense_recovery_rate": _round_or_none(randomization_eval.get("defense_recovery_rate")),
        },
        "adversarial_training": {
            "clean_accuracy": _round_or_none(
                adversarial_attack.get("clean_accuracy", adversarial_test.get("accuracy"))
            ),
            "adversarial_accuracy": _round_or_none(
                adversarial_attack.get(
                    "adversarial_accuracy",
                    (adversarial_training or {}).get("test_adversarial_accuracy"),
                )
            ),
            "prediction_change_rate": _round_or_none(adversarial_attack.get("prediction_change_rate")),
            "standard_attack_success_rate": _round_or_none(adversarial_attack.get("standard_attack_success_rate")),
            "per_class_attack_success": {
                class_name: _round_or_none(values.get("success_rate"))
                for class_name, values in adversarial_per_class.items()
            },
        },
    }
    return metrics
