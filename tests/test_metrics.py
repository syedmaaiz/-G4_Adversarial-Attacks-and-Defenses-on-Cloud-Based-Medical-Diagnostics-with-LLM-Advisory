from src.evaluation.metrics import attack_success_rate


def test_attack_success_rate_handles_zero_total():
    assert attack_success_rate(0, 0) == 0.0


def test_attack_success_rate_calculates_ratio():
    assert attack_success_rate(10, 3) == 0.3

