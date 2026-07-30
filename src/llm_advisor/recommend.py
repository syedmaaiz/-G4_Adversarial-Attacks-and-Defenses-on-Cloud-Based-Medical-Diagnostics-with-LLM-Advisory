"""CLI for requesting an LLM defense recommendation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.llm_advisor.advisor import recommend_defense
from src.llm_advisor.prompts import DEFAULT_PROJECT_METRICS


def load_metrics(metrics_json: Path | None) -> dict[str, Any]:
    """Load metrics from JSON, or use the latest project results."""
    if metrics_json is None:
        return DEFAULT_PROJECT_METRICS
    return json.loads(metrics_json.read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ask the LLM advisor for a defense recommendation.")
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
    metrics = load_metrics(args.metrics_json)
    recommendation = recommend_defense(metrics, use_fallback=not args.no_fallback)
    print(recommendation)


if __name__ == "__main__":
    main()
