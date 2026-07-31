"""LLM defense advisor.

Supports OpenAI-compatible chat completions, Google Gemini generateContent,
and local Ollama generation. Provider is selected with LLM_PROVIDER.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any

from src.llm_advisor.prompts import SYSTEM_PROMPT, build_advisory_prompt


@dataclass(frozen=True)
class LLMAdvisorConfig:
    provider: str
    openai_api_key: str | None
    openai_model: str
    openai_api_url: str
    gemini_api_key: str | None
    gemini_model: str
    gemini_api_url: str
    ollama_model: str
    ollama_base_url: str
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
        provider=os.getenv("LLM_PROVIDER", "openai").strip().lower(),
        openai_api_key=os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY"),
        openai_model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
        openai_api_url=os.getenv("LLM_API_URL", "https://api.openai.com/v1/chat/completions"),
        gemini_api_key=os.getenv("GEMINI_API_KEY"),
        gemini_model=os.getenv("GEMINI_MODEL", "gemini-2.0-flash"),
        gemini_api_url=os.getenv("GEMINI_API_URL", "https://generativelanguage.googleapis.com/v1beta"),
        ollama_model=os.getenv("OLLAMA_MODEL", "llama3.2:1b"),
        ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434"),
        timeout_seconds=int(os.getenv("LLM_TIMEOUT_SECONDS", "120")),
    )


def advisor_mode(metrics: dict[str, Any]) -> str:
    """Return the advisory mode stored in metrics."""
    return str(metrics.get("advisor_mode", "post-defense"))


def provider_prompt(metrics: dict[str, Any]) -> str:
    """Build a combined prompt for providers without a separate system role."""
    return f"{SYSTEM_PROMPT}\n\n{build_advisory_prompt(metrics, advisor_mode(metrics))}"


def fallback_recommendation(metrics: dict[str, Any], reason: str) -> str:
    """Return a deterministic recommendation if the LLM call is unavailable."""
    attack = metrics.get("attack", {})
    defenses = metrics.get("defenses", {})
    baseline_adv_acc = attack.get("adversarial_accuracy_baseline")
    baseline_asr = attack.get("standard_attack_success_rate_baseline")
    normal_asr = attack.get("per_class_attack_success_baseline", {}).get("NORMAL")

    if not defenses:
        return (
            "LLM advisor fallback used because: "
            f"{reason}\n\n"
            "Pre-defense recommendation: implement adversarial training first, "
            "and test defensive randomization as a lightweight comparison.\n\n"
            f"Evidence: FGSM reduced baseline adversarial accuracy to {baseline_adv_acc} "
            f"and achieved a standard attack success rate of {baseline_asr}. "
            f"NORMAL images were especially vulnerable, with attack success {normal_asr}.\n\n"
            "Metrics to collect next: defended adversarial accuracy, clean accuracy "
            "after defense, attack success rate after defense, and per-class attack success."
        )

    randomization = defenses.get("defensive_randomization", {})
    adversarial_training = defenses.get("adversarial_training", {})
    randomization_adv_acc = randomization.get("defended_adversarial_accuracy")
    adv_training_adv_acc = adversarial_training.get("adversarial_accuracy")
    adv_training_asr = adversarial_training.get("standard_attack_success_rate")

    return (
        "LLM advisor fallback used because: "
        f"{reason}\n\n"
        "Post-defense recommendation: select adversarial training for deployment.\n\n"
        "Evidence: defensive randomization only modestly improved adversarial "
        f"accuracy from {baseline_adv_acc} to {randomization_adv_acc}, while "
        f"adversarial training improved adversarial accuracy to {adv_training_adv_acc}. "
        f"The standard attack success rate also decreased from {baseline_asr} "
        f"to {adv_training_asr}.\n\n"
        "Deployment caution: continue monitoring per-class robustness because "
        "NORMAL images remain more vulnerable than PNEUMONIA images."
    )


def post_json(url: str, payload: dict[str, Any], headers: dict[str, str], timeout: int) -> dict[str, Any]:
    """POST JSON and return parsed JSON, normalizing HTTP errors."""
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json", **headers},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            response_body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        raise LLMAdvisorError(f"LLM API request failed with status {exc.code}: {error_body[:500]}") from exc
    except urllib.error.URLError as exc:
        raise LLMAdvisorError(f"LLM API request failed: {exc}") from exc

    return json.loads(response_body)


def call_openai(metrics: dict[str, Any], config: LLMAdvisorConfig) -> str:
    """Call an OpenAI-compatible chat completion endpoint."""
    if not config.openai_api_key:
        raise LLMAdvisorError("LLM_API_KEY or OPENAI_API_KEY is not set")

    payload = {
        "model": config.openai_model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_advisory_prompt(metrics, advisor_mode(metrics))},
        ],
        "temperature": 0.2,
    }
    data = post_json(
        config.openai_api_url,
        payload,
        {"Authorization": f"Bearer {config.openai_api_key}"},
        config.timeout_seconds,
    )
    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise LLMAdvisorError(f"Unexpected OpenAI response shape: {data}") from exc

    if not content or not content.strip():
        raise LLMAdvisorError("OpenAI returned an empty recommendation")
    return content.strip()


def call_gemini(metrics: dict[str, Any], config: LLMAdvisorConfig) -> str:
    """Call the Google Gemini generateContent endpoint."""
    if not config.gemini_api_key:
        raise LLMAdvisorError("GEMINI_API_KEY is not set")

    model_name = config.gemini_model
    if not model_name.startswith("models/"):
        model_name = f"models/{model_name}"

    base_url = config.gemini_api_url.rstrip("/")
    encoded_model = urllib.parse.quote(model_name, safe="/")
    url = f"{base_url}/{encoded_model}:generateContent"
    payload = {
        "contents": [{"role": "user", "parts": [{"text": provider_prompt(metrics)}]}],
        "generationConfig": {"temperature": 0.2},
    }
    data = post_json(url, payload, {"x-goog-api-key": config.gemini_api_key}, config.timeout_seconds)

    try:
        parts = data["candidates"][0]["content"]["parts"]
        content = "".join(part.get("text", "") for part in parts)
    except (KeyError, IndexError, TypeError) as exc:
        raise LLMAdvisorError(f"Unexpected Gemini response shape: {data}") from exc

    if not content or not content.strip():
        raise LLMAdvisorError("Gemini returned an empty recommendation")
    return content.strip()


def call_ollama(metrics: dict[str, Any], config: LLMAdvisorConfig) -> str:
    """Call a local Ollama model through /api/generate."""
    url = f"{config.ollama_base_url.rstrip('/')}/api/generate"
    payload = {
        "model": config.ollama_model,
        "prompt": provider_prompt(metrics),
        "stream": False,
        "options": {"temperature": 0.2},
    }
    data = post_json(url, payload, {}, config.timeout_seconds)
    content = data.get("response", "")
    if not content or not content.strip():
        raise LLMAdvisorError(f"Unexpected Ollama response shape: {data}")
    return content.strip()


def call_llm(metrics: dict[str, Any], config: LLMAdvisorConfig) -> str:
    """Call the configured LLM provider."""
    if config.provider == "openai":
        return call_openai(metrics, config)
    if config.provider == "gemini":
        return call_gemini(metrics, config)
    if config.provider == "ollama":
        return call_ollama(metrics, config)
    raise LLMAdvisorError(f"Unsupported LLM_PROVIDER '{config.provider}'. Use 'openai', 'gemini', or 'ollama'.")


def recommend_defense(attack_context: dict[str, Any], use_fallback: bool = True) -> str:
    """Recommend a defense strategy from attack/defense metadata."""
    try:
        return call_llm(attack_context, load_config())
    except Exception as exc:
        if not use_fallback:
            raise
        return fallback_recommendation(attack_context, str(exc))
