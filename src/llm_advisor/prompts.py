"""Prompt construction for the LLM defense advisor."""

from __future__ import annotations

import json
from typing import Any


SYSTEM_PROMPT = """You are a cybersecurity advisor for a medical imaging ML system.
Your task is to recommend defenses against adversarial machine-learning attacks.
Be concise, evidence-based, and careful: do not provide medical diagnosis advice.
Focus only on model security, robustness, evaluation, and deployment risk.
"""


DEFAULT_PROJECT_METRICS: dict[str, Any] = {
    "dataset": "Chest X-ray classification: NORMAL vs PNEUMONIA",
    "model": "ResNet18 image classifier",
    "attack": {
        "name": "FGSM",
        "epsilon": 0.01,
        "clean_accuracy_baseline": 0.8574,
        "adversarial_accuracy_baseline": 0.5369,
        "prediction_change_rate_baseline": 0.3205,
        "standard_attack_success_rate_baseline": 0.3738,
        "per_class_attack_success_baseline": {
            "NORMAL": 0.9865,
            "PNEUMONIA": 0.1395,
        },
    },
    "defenses": {
        "defensive_randomization": {
            "resize_delta": 8,
            "defended_clean_accuracy": 0.8381,
            "defended_adversarial_accuracy": 0.5705,
            "defense_recovery_rate": 0.1050,
        },
        "adversarial_training": {
            "clean_accuracy": 0.8590,
            "adversarial_accuracy": 0.7404,
            "prediction_change_rate": 0.1186,
            "standard_attack_success_rate": 0.1381,
            "per_class_attack_success": {
                "NORMAL": 0.5034,
                "PNEUMONIA": 0.0,
            },
        },
    },
}


def build_advisory_prompt(metrics: dict[str, Any]) -> str:
    """Build the user prompt from structured project metrics."""
    metrics_json = json.dumps(metrics, indent=2, sort_keys=True)
    return f"""Analyze the following adversarial ML experiment results and recommend the best defense strategy.

Metrics:
{metrics_json}

Please return:
1. Recommended defense.
2. Evidence from the metrics.
3. Why weaker defenses may be insufficient.
4. Deployment cautions for a cloud-hosted medical diagnostics model.
5. Next evaluation steps.

Keep the response suitable for inclusion in a course project report.
"""
