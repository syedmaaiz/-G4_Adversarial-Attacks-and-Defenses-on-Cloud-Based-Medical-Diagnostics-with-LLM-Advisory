"""LLM defense advisor with a mock fallback."""


def recommend_defense(attack_context: dict) -> str:
    """Recommend a defense strategy from attack metadata."""
    attack_type = attack_context.get("attack_type", "unknown attack")
    return (
        f"Mock advisor recommendation for {attack_type}: "
        "try defensive randomization first, then evaluate adversarial training."
    )

