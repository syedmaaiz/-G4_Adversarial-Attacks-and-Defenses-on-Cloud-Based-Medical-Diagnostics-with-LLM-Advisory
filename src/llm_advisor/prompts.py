"""Prompt construction for the LLM defense advisor."""

from __future__ import annotations

import json
from typing import Any


SYSTEM_PROMPT = """You are a cybersecurity advisor for a medical imaging ML system.
Your task is to recommend defenses against adversarial machine-learning attacks.
Be concise, evidence-based, and careful: do not provide medical diagnosis advice.
Focus only on model security, robustness, evaluation, and deployment risk.
"""


PRE_DEFENSE_METRICS: dict[str, Any] = {
    "advisor_mode": "pre-defense",
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
}


POST_DEFENSE_METRICS: dict[str, Any] = {
    **PRE_DEFENSE_METRICS,
    "advisor_mode": "post-defense",
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


DEFAULT_PROJECT_METRICS = POST_DEFENSE_METRICS


MODE_METRICS = {
    "pre-defense": PRE_DEFENSE_METRICS,
    "post-defense": POST_DEFENSE_METRICS,
}


def get_default_metrics(mode: str) -> dict[str, Any]:
    """Return built-in metrics for the requested advisor mode."""
    try:
        return MODE_METRICS[mode]
    except KeyError as exc:
        raise ValueError("mode must be 'pre-defense' or 'post-defense'") from exc


def build_advisory_prompt(metrics: dict[str, Any], mode: str = "post-defense") -> str:
    """Build the user prompt from structured project metrics."""
    metrics_json = json.dumps(metrics, indent=2, sort_keys=True)
    if mode == "pre-defense":
        request = """The defense has not been implemented yet. Based only on the baseline attack evidence, recommend candidate defenses to implement and explain the expected reason for each choice.

Please return:
1. Immediate security interpretation of the attack result.
2. Top 2-3 defense candidates to implement next.
3. Why each candidate is appropriate for this FGSM result.
4. What metrics should be collected after implementing defenses.
5. Deployment cautions for a cloud-hosted medical diagnostics model."""
    else:
        request = """Candidate defenses have already been implemented and evaluated. Compare the defense results and recommend which defense should be selected for deployment.

Please return:
1. Recommended final defense.
2. Evidence from the metrics.
3. Why weaker defenses were insufficient.
4. Deployment cautions for a cloud-hosted medical diagnostics model.
5. Next evaluation steps."""

    return f"""Analyze the following adversarial ML experiment results.

Advisor mode: {mode}

Metrics:
{metrics_json}

{request}

Keep the response suitable for inclusion in a course project report.
"""
