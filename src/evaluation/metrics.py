"""Shared evaluation helpers."""


def attack_success_rate(total_attacks: int, successful_attacks: int) -> float:
    if total_attacks == 0:
        return 0.0
    return successful_attacks / total_attacks

