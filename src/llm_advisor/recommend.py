"""CLI for requesting an LLM defense recommendation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.llm_advisor.advisor import recommend_defense
from src.llm_advisor.prompts import get_default_metrics


def load_metrics(metrics_json: Path | None, mode: str) -> dict[str, Any]:
    """Load metrics from JSON, or use built-in metrics for the selected mode."""
    if metrics_json is None:
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
        "--no-fallback",
        action="store_true",
        help="Fail if the LLM API call fails instead of returning fallback text.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = load_metrics(args.metrics_json, args.mode)
    recommendation = recommend_defense(metrics, use_fallback=not args.no_fallback)
    print(recommendation)


if __name__ == "__main__":
    main()
