"""CLI for requesting an LLM defense recommendation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.evaluation.metric_store import DEFAULT_METRICS_DIR, build_advisory_metrics_from_files
from src.llm_advisor.advisor import recommend_defense
from src.llm_advisor.prompts import get_default_metrics


def load_metrics(
    metrics_json: Path | None,
    mode: str,
    metrics_dir: Path = DEFAULT_METRICS_DIR,
) -> dict[str, Any]:
    """Load metrics from JSON, saved run files, or built-in representative metrics."""
    if metrics_json is None:
        saved_metrics = build_advisory_metrics_from_files(mode, metrics_dir)
        if saved_metrics is not None:
            return saved_metrics
        return dict(get_default_metrics(mode))
    metrics = json.loads(metrics_json.read_text(encoding="utf-8"))
    metrics.setdefault("advisor_mode", mode)
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ask the LLM advisor for a defense recommendation.")
    parser.add_argument(
        "--mode",
        choices=("pre-defense", "post-defense"),
        default="post-defense",
        help="Use attack-only advice or final defense-selection advice.",
    )
    parser.add_argument(
        "--metrics-json",
        type=Path,
        default=None,
        help="Optional JSON file containing attack/defense metrics.",
    )
    parser.add_argument(
        "--metrics-dir",
        type=Path,
        default=DEFAULT_METRICS_DIR,
        help="Directory containing saved run metrics JSON files.",
    )
    parser.add_argument(
        "--no-fallback",
        action="store_true",
        help="Fail if the LLM API call fails instead of returning fallback text.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = load_metrics(args.metrics_json, args.mode, args.metrics_dir)
    recommendation = recommend_defense(metrics, use_fallback=not args.no_fallback)
    print(recommendation)


if __name__ == "__main__":
    main()
