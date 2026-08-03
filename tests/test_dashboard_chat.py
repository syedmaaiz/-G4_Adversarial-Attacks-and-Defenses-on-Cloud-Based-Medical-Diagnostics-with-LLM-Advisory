"""Tests for the interactive results dashboard."""

from src.dashboard import chat
from src.dashboard.generate_dashboard import render_dashboard


def test_chat_prompt_contains_question_metrics_and_level() -> None:
    prompt = chat.build_chat_prompt("What happened after FGSM?", "technical", [])

    assert "What happened after FGSM?" in prompt
    assert "technical" in prompt
    assert "0.5369" in prompt
    assert "Never diagnose" in prompt


def test_fallback_answers_are_grounded() -> None:
    answer = chat.fallback_chat_answer("Which defense is best?", "beginner")

    assert "Adversarial training" in answer
    assert "74.04%" in answer
    assert "85.90%" in answer


def test_dashboard_contains_accessible_chat_ui() -> None:
    dashboard = render_dashboard()

    assert 'id="chat-panel"' in dashboard
    assert 'aria-label="Security Results Assistant"' in dashboard
    assert "fetch('api/chat'" in dashboard
    assert "Not medical advice" in dashboard


def test_dashboard_contains_interactive_stage_explorer() -> None:
    dashboard = render_dashboard()

    assert 'id="experiment-explorer"' in dashboard
    assert 'data-stage="fgsm"' in dashboard
    assert 'id="ask-stage"' in dashboard
    assert "selectStage" in dashboard
