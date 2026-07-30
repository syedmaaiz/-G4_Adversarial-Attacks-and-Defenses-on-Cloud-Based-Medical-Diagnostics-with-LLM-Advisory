"""LLM defense advisor.

The primary path calls an OpenAI-compatible chat completions endpoint using
LLM_API_KEY from the environment. A deterministic fallback is returned only when
API credentials are missing or the request fails, so the project demo still runs.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

from src.llm_advisor.prompts import SYSTEM_PROMPT, build_advisory_prompt


@dataclass(frozen=True)
class LLMAdvisorConfig:
    api_key: str | None
    model: str
    api_url: str
    timeout_seconds: int


class LLMAdvisorError(RuntimeError):
    """Raised when the LLM advisor cannot return a usable response."""


def load_dotenv_if_available() -> None:
    """Load .env values when python-dotenv is installed."""
    try:
        from dotenv import load_dotenv
    except ModuleNotFoundError:
        return
    load_dotenv()


def load_config() -> LLMAdvisorConfig:
    """Load advisor configuration from .env/environment variables."""
    load_dotenv_if_available()
    return LLMAdvisorConfig(
        api_key=os.getenv("LLM_API_KEY"),
        model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
        api_url=os.getenv(
            "LLM_API_URL", "https://api.openai.com/v1/chat/completions"
        ),
        timeout_seconds=int(os.getenv("LLM_TIMEOUT_SECONDS", "60")),
    )


def fallback_recommendation(metrics: dict[str, Any], reason: str) -> str:
    """Return a deterministic recommendation if the LLM call is unavailable."""
    attack = metrics.get("attack", {})
    defenses = metrics.get("defenses", {})
    randomization = defenses.get("defensive_randomization", {})
    adversarial_training = defenses.get("adversarial_training", {})

    baseline_adv_acc = attack.get("adversarial_accuracy_baseline")
    randomization_adv_acc = randomization.get("defended_adversarial_accuracy")
    adv_training_adv_acc = adversarial_training.get("adversarial_accuracy")
    baseline_asr = attack.get("standard_attack_success_rate_baseline")
    adv_training_asr = adversarial_training.get("standard_attack_success_rate")

    return (
        "LLM advisor fallback used because: "
        f"{reason}\n\n"
        "Recommended defense: adversarial training.\n\n"
        "Evidence: defensive randomization only modestly improved adversarial "
        f"accuracy from {baseline_adv_acc} to {randomization_adv_acc}, while "
        f"adversarial training improved adversarial accuracy to {adv_training_adv_acc}. "
        f"The standard attack success rate also decreased from {baseline_asr} "
        f"to {adv_training_asr}.\n\n"
        "Deployment caution: continue monitoring per-class robustness because "
        "NORMAL images remain more vulnerable than PNEUMONIA images."
    )


def call_llm(metrics: dict[str, Any], config: LLMAdvisorConfig) -> str:
    """Call the configured OpenAI-compatible chat completion endpoint."""
    if not config.api_key:
        raise LLMAdvisorError("LLM_API_KEY is not set")

    payload = {
        "model": config.model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_advisory_prompt(metrics)},
        ],
        "temperature": 0.2,
    }
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        config.api_url,
        data=body,
        headers={
            "Authorization": f"Bearer {config.api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=config.timeout_seconds) as response:
            response_body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        raise LLMAdvisorError(
            f"LLM API request failed with status {exc.code}: {error_body[:500]}"
        ) from exc
    except urllib.error.URLError as exc:
        raise LLMAdvisorError(f"LLM API request failed: {exc}") from exc

    data = json.loads(response_body)
    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise LLMAdvisorError(f"Unexpected LLM response shape: {data}") from exc

    if not content or not content.strip():
        raise LLMAdvisorError("LLM returned an empty recommendation")
    return content.strip()


def recommend_defense(
    attack_context: dict[str, Any],
    use_fallback: bool = True,
) -> str:
    """Recommend a defense strategy from attack/defense metadata."""
    try:
        return call_llm(attack_context, load_config())
    except Exception as exc:
        if not use_fallback:
            raise
        return fallback_recommendation(attack_context, str(exc))
