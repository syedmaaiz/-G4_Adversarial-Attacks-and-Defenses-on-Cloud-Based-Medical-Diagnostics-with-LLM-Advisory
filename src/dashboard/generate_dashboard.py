"""Generate a static dashboard for attack and defense results."""

from __future__ import annotations

import argparse
import html
from dataclasses import dataclass
from pathlib import Path

from src.evaluation.metric_store import (
    ADVERSARIAL_TRAINING_JSON,
    DEFAULT_METRICS_DIR,
    FGSM_ADVERSARIAL_JSON,
    FGSM_BASELINE_JSON,
    RANDOMIZATION_JSON,
    metrics_path,
    read_metrics_json,
)


DEFAULT_OUTPUT = Path("reports/dashboard.html")


@dataclass(frozen=True)
class ModelResult:
    name: str
    clean_accuracy: float
    adversarial_accuracy: float
    attack_success_rate: float
    prediction_change_rate: float


@dataclass(frozen=True)
class DefenseResult:
    name: str
    defended_clean_accuracy: float | None
    defended_adversarial_accuracy: float
    recovery_rate: float | None
    notes: str


BASELINE = ModelResult("Baseline ResNet18", 0.8574, 0.5369, 0.3738, 0.3205)
ADVERSARIAL_MODEL = ModelResult("Adversarially Trained ResNet18", 0.8590, 0.7404, 0.1381, 0.1186)
RANDOMIZATION = DefenseResult("Defensive Randomization", 0.8381, 0.5705, 0.1050, "Small robustness gain, but weak recovery against NORMAL-image attacks.")
ADVERSARIAL_TRAINING = DefenseResult("Adversarial Training", 0.8590, 0.7404, None, "Strongest defense: improves robustness while preserving clean accuracy.")

PER_CLASS_ROWS = [
    ("NORMAL", 0.6325, 0.0085, 0.0128, 0.5034),
    ("PNEUMONIA", 0.9923, 0.8538, 0.9051, 0.0000),
]



def _ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def load_dashboard_metrics(metrics_dir: Path = DEFAULT_METRICS_DIR) -> str:
    """Load latest saved metrics into dashboard globals when available."""
    global BASELINE, ADVERSARIAL_MODEL, RANDOMIZATION, ADVERSARIAL_TRAINING, PER_CLASS_ROWS

    source_note = "representative project metrics"
    baseline_fgsm = read_metrics_json(metrics_path(metrics_dir, FGSM_BASELINE_JSON))
    randomization = read_metrics_json(metrics_path(metrics_dir, RANDOMIZATION_JSON))
    adversarial_fgsm = read_metrics_json(metrics_path(metrics_dir, FGSM_ADVERSARIAL_JSON))
    adversarial_training = read_metrics_json(metrics_path(metrics_dir, ADVERSARIAL_TRAINING_JSON))

    if baseline_fgsm:
        attack = baseline_fgsm.get("attack", {})
        BASELINE = ModelResult(
            "Baseline ResNet18",
            float(attack.get("clean_accuracy", BASELINE.clean_accuracy)),
            float(attack.get("adversarial_accuracy", BASELINE.adversarial_accuracy)),
            float(attack.get("standard_attack_success_rate", BASELINE.attack_success_rate)),
            float(attack.get("prediction_change_rate", BASELINE.prediction_change_rate)),
        )
        source_note = f"latest saved metrics from {metrics_dir.as_posix()}"

    if randomization:
        randomization_metrics = randomization.get("randomization", {})
        RANDOMIZATION = DefenseResult(
            "Defensive Randomization",
            randomization_metrics.get("defended_clean_accuracy", RANDOMIZATION.defended_clean_accuracy),
            float(randomization_metrics.get("defended_adversarial_accuracy", RANDOMIZATION.defended_adversarial_accuracy)),
            randomization_metrics.get("defense_recovery_rate", RANDOMIZATION.recovery_rate),
            "Small robustness gain, but weak recovery against NORMAL-image attacks.",
        )

    if adversarial_fgsm or adversarial_training:
        adv_attack = (adversarial_fgsm or {}).get("attack", {})
        adv_test = (adversarial_training or {}).get("test", {})
        ADVERSARIAL_MODEL = ModelResult(
            "Adversarially Trained ResNet18",
            float(adv_attack.get("clean_accuracy", adv_test.get("accuracy", ADVERSARIAL_MODEL.clean_accuracy))),
            float(
                adv_attack.get(
                    "adversarial_accuracy",
                    (adversarial_training or {}).get("test_adversarial_accuracy", ADVERSARIAL_MODEL.adversarial_accuracy),
                )
            ),
            float(adv_attack.get("standard_attack_success_rate", ADVERSARIAL_MODEL.attack_success_rate)),
            float(adv_attack.get("prediction_change_rate", ADVERSARIAL_MODEL.prediction_change_rate)),
        )
        ADVERSARIAL_TRAINING = DefenseResult(
            "Adversarial Training",
            ADVERSARIAL_MODEL.clean_accuracy,
            ADVERSARIAL_MODEL.adversarial_accuracy,
            None,
            "Strongest defense: improves robustness while preserving clean accuracy.",
        )

    if baseline_fgsm:
        baseline_classes = baseline_fgsm.get("attack", {}).get("per_class", {})
        randomized_classes = (randomization or {}).get("randomization", {}).get("per_class", {})
        adversarial_classes = (adversarial_fgsm or {}).get("attack", {}).get("per_class", {})
        rows = []
        for class_name, values in baseline_classes.items():
            total = values.get("total", 0)
            clean_correct = values.get("clean_correct", 0)
            randomized = randomized_classes.get(class_name, {})
            adversarial = adversarial_classes.get(class_name, {})
            rows.append(
                (
                    class_name,
                    randomized.get("clean_accuracy", values.get("clean_accuracy", _ratio(clean_correct, total))),
                    randomized.get("adversarial_accuracy", values.get("adversarial_accuracy", 0.0)),
                    randomized.get("defended_adversarial_accuracy", values.get("adversarial_accuracy", 0.0)),
                    adversarial.get("success_rate", values.get("success_rate", 0.0)),
                )
            )
        if rows:
            PER_CLASS_ROWS = [
                (
                    class_name,
                    float(clean),
                    float(attacked if attacked is not None else 0.0),
                    float(randomized if randomized is not None else 0.0),
                    float(adv_success),
                )
                for class_name, clean, attacked, randomized, adv_success in rows
            ]

    return source_note

def pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def delta(new: float, old: float) -> str:
    sign = "+" if new >= old else ""
    return f"{sign}{(new - old) * 100:.2f} pp"


def bar(value: float, label: str, tone: str = "blue") -> str:
    width = max(0, min(100, value * 100))
    return f'''
    <div class="bar-row" aria-label="{html.escape(label)} {pct(value)}">
      <div class="bar-label"><span>{html.escape(label)}</span><strong>{pct(value)}</strong></div>
      <div class="bar-track"><div class="bar-fill {tone}" style="width: {width:.2f}%"></div></div>
    </div>
    '''


def metric_card(title: str, value: str, caption: str, tone: str = "neutral") -> str:
    return f'''
    <section class="metric-card {tone}">
      <span>{html.escape(title)}</span>
      <strong>{html.escape(value)}</strong>
      <p>{html.escape(caption)}</p>
    </section>
    '''


def story_card(step: str, title: str, body: str, tone: str = "") -> str:
    return f'''
    <article class="story-card {tone}">
      <span>{html.escape(step)}</span>
      <h3>{html.escape(title)}</h3>
      <p>{html.escape(body)}</p>
    </article>
    '''


def plain_english_summary() -> str:
    return f'''
    <section class="panel wide">
      <div class="panel-heading">
        <h2>What We Did In Simple Words</h2>
        <p>A non-technical walkthrough of the project result.</p>
      </div>
      <div class="story-grid">
        {story_card("Step 1", "We trained a normal X-ray classifier", f"The baseline model learned to classify chest X-rays as NORMAL or PNEUMONIA. Before any attack, it was correct {pct(BASELINE.clean_accuracy)} of the time.", "good")}
        {story_card("Step 2", "We attacked the model", f"FGSM added tiny calculated changes to the X-ray images. After the attack, baseline accuracy fell to {pct(BASELINE.adversarial_accuracy)}, meaning many predictions were fooled.", "risk")}
        {story_card("Step 3", "A simple defense helped only a little", f"Defensive randomization slightly changed images before prediction. It improved attacked-image accuracy to {pct(RANDOMIZATION.defended_adversarial_accuracy)}, but that was only a small recovery.", "warn")}
        {story_card("Step 4", "Training with attacked images helped a lot", f"Adversarial training taught the model using clean and attacked images. Attacked-image accuracy improved to {pct(ADVERSARIAL_MODEL.adversarial_accuracy)} while clean accuracy stayed about the same.", "good")}
      </div>
    </section>
    '''


def metric_explainer() -> str:
    return '''
    <section class="panel wide">
      <div class="panel-heading">
        <h2>How To Read The Metrics</h2>
        <p>These short definitions make the dashboard readable without a statistics background.</p>
      </div>
      <div class="definition-grid">
        <div><strong>Clean accuracy</strong><p>How often the model is right on original X-ray images.</p></div>
        <div><strong>Adversarial accuracy</strong><p>How often the model is still right after the attacker changes the image.</p></div>
        <div><strong>Attack success rate</strong><p>Out of images the model originally got right, how many became wrong after attack. Lower is better.</p></div>
        <div><strong>Defense recovery rate</strong><p>Out of successful attacks, how many were corrected by a defense.</p></div>
      </div>
    </section>
    '''

def advisor_story() -> str:
    return f'''
    <section class="panel wide">
      <div class="panel-heading">
        <h2>LLM Advisory Role</h2>
        <p>The advisor has two phases: first it suggests defenses after seeing the attack, then it selects the best defense after experiments.</p>
      </div>
      <div class="split">
        <div class="advisor-card">
          <span>Before defenses were implemented</span>
          <h3>Pre-defense advice</h3>
          <p>After seeing that FGSM dropped accuracy to {pct(BASELINE.adversarial_accuracy)} and strongly affected NORMAL images, the advisor recommends trying adversarial training as the main defense and defensive randomization as a lightweight comparison.</p>
        </div>
        <div class="advisor-card selected">
          <span>After defenses were evaluated</span>
          <h3>Post-defense advice</h3>
          <p>After comparing results, the advisor recommends adversarial training because it raised attacked-image accuracy to {pct(ADVERSARIAL_MODEL.adversarial_accuracy)} and lowered attack success to {pct(ADVERSARIAL_MODEL.attack_success_rate)}.</p>
        </div>
      </div>
      <div class="explain-box">
        <strong>Plain-English result</strong>
        <p>The LLM is not the defense engine. It is the decision-support layer: it reads evidence, recommends what to try, and then explains which defense worked best.</p>
      </div>
    </section>
    '''

def model_comparison() -> str:
    return f'''
    <section class="panel wide">
      <div class="panel-heading">
        <h2>Baseline vs Adversarial Training</h2>
        <p>Higher clean/adversarial accuracy is better. Lower attack success is better.</p>
      </div>
      <div class="split">
        <div>
          <h3>{BASELINE.name}</h3>
          {bar(BASELINE.clean_accuracy, "Clean accuracy", "green")}
          {bar(BASELINE.adversarial_accuracy, "Adversarial accuracy", "red")}
          {bar(BASELINE.attack_success_rate, "Attack success rate", "red")}
        </div>
        <div>
          <h3>{ADVERSARIAL_MODEL.name}</h3>
          {bar(ADVERSARIAL_MODEL.clean_accuracy, "Clean accuracy", "green")}
          {bar(ADVERSARIAL_MODEL.adversarial_accuracy, "Adversarial accuracy", "blue")}
          {bar(ADVERSARIAL_MODEL.attack_success_rate, "Attack success rate", "amber")}
        </div>
      </div>
      <div class="explain-box">
        <strong>Plain-English result</strong>
        <p>The baseline model handled original images well, but the attack exposed a weakness. Adversarial training kept normal-image performance stable and made attacked images much more likely to be classified correctly.</p>
      </div>
    </section>
    '''


def defense_table() -> str:
    rows = []
    for defense in (RANDOMIZATION, ADVERSARIAL_TRAINING):
        clean = pct(defense.defended_clean_accuracy) if defense.defended_clean_accuracy is not None else "N/A"
        recovery = pct(defense.recovery_rate) if defense.recovery_rate is not None else "N/A"
        rows.append(f"""
          <tr>
            <td>{html.escape(defense.name)}</td>
            <td>{clean}</td>
            <td>{pct(defense.defended_adversarial_accuracy)}</td>
            <td>{recovery}</td>
            <td>{html.escape(defense.notes)}</td>
          </tr>
        """)
    return f'''
    <section class="panel wide">
      <div class="panel-heading">
        <h2>Defense Comparison</h2>
        <p>Randomization gives a small gain; adversarial training gives the strongest robustness improvement.</p>
      </div>
      <table>
        <thead><tr><th>Defense</th><th>Clean Accuracy</th><th>Adversarial Accuracy</th><th>Recovery Rate</th><th>Interpretation</th></tr></thead>
        <tbody>{''.join(rows)}</tbody>
      </table>
      <div class="explain-box">
        <strong>Plain-English result</strong>
        <p>Randomization was a small patch. Adversarial training was a stronger fix because the model practiced against attacked images during training.</p>
      </div>
    </section>
    '''


def per_class_table() -> str:
    rows = []
    for class_name, clean, attacked, randomized, adv_trained_attack_success in PER_CLASS_ROWS:
        rows.append(f"""
          <tr>
            <td>{class_name}</td>
            <td>{pct(clean)}</td>
            <td>{pct(attacked)}</td>
            <td>{pct(randomized)}</td>
            <td>{pct(adv_trained_attack_success)}</td>
          </tr>
        """)
    return f'''
    <section class="panel wide">
      <div class="panel-heading">
        <h2>Per-Class Behavior</h2>
        <p>NORMAL images were the main vulnerability under FGSM. Adversarial training reduced, but did not eliminate, that class-specific risk.</p>
      </div>
      <table>
        <thead><tr><th>Class</th><th>Clean Accuracy</th><th>Baseline FGSM Accuracy</th><th>Randomized FGSM Accuracy</th><th>Adv-Trained Attack Success</th></tr></thead>
        <tbody>{''.join(rows)}</tbody>
      </table>
      <div class="explain-box">
        <strong>Plain-English result</strong>
        <p>The attack mostly fooled NORMAL examples. This matters because a security problem can affect one class much more than another, even when the overall accuracy looks acceptable.</p>
      </div>
    </section>
    '''


def render_dashboard(metrics_dir: Path = DEFAULT_METRICS_DIR) -> str:
    source_note = load_dashboard_metrics(metrics_dir)
    improvement = delta(ADVERSARIAL_MODEL.adversarial_accuracy, BASELINE.adversarial_accuracy)
    attack_reduction = delta(ADVERSARIAL_MODEL.attack_success_rate, BASELINE.attack_success_rate)
    return f'''<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Adversarial Medical Diagnostics Dashboard</title>
  <style>
    :root {{ --bg: #f4f7fb; --ink: #152033; --muted: #627089; --panel: #ffffff; --line: #d9e1ec; --blue: #2f6fed; --green: #16865a; --red: #c93c3c; --amber: #b87613; --shadow: 0 18px 45px rgba(21, 32, 51, 0.11); }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: var(--ink); background: var(--bg); line-height: 1.45; }}
    header {{ padding: 34px 28px 20px; background: linear-gradient(135deg, #0b1f38, #174b72); color: white; }}
    header h1 {{ margin: 0 0 8px; font-size: 32px; letter-spacing: 0; }}
    header p {{ margin: 0; max-width: 900px; color: #cfdbeb; }}
    main {{ padding: 24px; max-width: 1180px; margin: 0 auto; }}
    .metrics {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 14px; margin-bottom: 18px; }}
    .metric-card, .panel {{ background: var(--panel); border: 1px solid var(--line); border-radius: 8px; box-shadow: var(--shadow); }}
    .metric-card {{ padding: 16px; min-height: 130px; }}
    .metric-card span {{ color: var(--muted); font-size: 13px; font-weight: 700; text-transform: uppercase; }}
    .metric-card strong {{ display: block; font-size: 30px; margin: 8px 0 4px; }}
    .metric-card p {{ margin: 0; color: var(--muted); font-size: 14px; }}
    .metric-card.good strong {{ color: var(--green); }} .metric-card.warn strong {{ color: var(--amber); }} .metric-card.risk strong {{ color: var(--red); }}
    .grid {{ display: grid; grid-template-columns: 1fr; gap: 18px; }}
    .panel {{ padding: 20px; margin-bottom: 18px; }}
    .panel-heading {{ display: flex; justify-content: space-between; gap: 20px; align-items: end; border-bottom: 1px solid var(--line); padding-bottom: 12px; margin-bottom: 16px; }}
    .panel-heading h2 {{ margin: 0; font-size: 21px; }}
    .panel-heading p {{ margin: 0; color: var(--muted); max-width: 560px; }}
    .split, .story-grid, .definition-grid {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 18px; }}
    .story-grid, .definition-grid {{ grid-template-columns: repeat(4, minmax(0, 1fr)); }}
    h3 {{ margin: 0 0 14px; font-size: 17px; }}
    .story-card, .definition-grid div, .explain-box {{ border: 1px solid var(--line); border-radius: 8px; background: #fbfdff; padding: 14px; }}
    .story-card {{ min-height: 190px; }}
    .story-card span {{ color: var(--muted); font-size: 12px; font-weight: 800; text-transform: uppercase; }}
    .story-card h3 {{ margin: 8px 0; font-size: 16px; }}
    .story-card p, .definition-grid p, .explain-box p {{ margin: 0; color: var(--muted); font-size: 14px; }}
    .story-card.good {{ border-top: 4px solid var(--green); }} .story-card.warn {{ border-top: 4px solid var(--amber); }} .story-card.risk {{ border-top: 4px solid var(--red); }}
    .definition-grid strong, .explain-box strong {{ display: block; margin-bottom: 6px; }}
    .bar-row {{ margin-bottom: 14px; }}
    .bar-label {{ display: flex; justify-content: space-between; gap: 12px; font-size: 14px; margin-bottom: 6px; }}
    .bar-label span {{ color: var(--muted); }} .bar-track {{ height: 14px; background: #e8eef6; border-radius: 999px; overflow: hidden; }}
    .bar-fill {{ height: 100%; border-radius: inherit; }} .bar-fill.blue {{ background: var(--blue); }} .bar-fill.green {{ background: var(--green); }} .bar-fill.red {{ background: var(--red); }} .bar-fill.amber {{ background: var(--amber); }}
    table {{ width: 100%; border-collapse: collapse; font-size: 14px; }}
    th, td {{ border-bottom: 1px solid var(--line); padding: 11px 10px; text-align: left; vertical-align: top; }}
    th {{ color: var(--muted); font-size: 12px; text-transform: uppercase; }}
    .callout {{ border-left: 5px solid var(--blue); padding: 16px 18px; background: #eef4ff; border-radius: 6px; margin-top: 18px; }}
    .callout strong {{ display: block; margin-bottom: 6px; }}
    footer {{ color: var(--muted); padding: 18px 24px 34px; max-width: 1180px; margin: 0 auto; font-size: 13px; }}
    .chat-launcher {{ position: fixed; right: 24px; bottom: 24px; z-index: 20; display: flex; align-items: center; gap: 10px; border: 0; border-radius: 999px; padding: 13px 18px; background: #10243f; color: white; box-shadow: 0 16px 38px rgba(16, 36, 63, .28); font: inherit; font-weight: 750; cursor: pointer; }}
    .chat-launcher:hover {{ background: #174b72; transform: translateY(-1px); }}
    .chat-launcher .pulse {{ width: 10px; height: 10px; border-radius: 50%; background: #55d6a5; box-shadow: 0 0 0 5px rgba(85, 214, 165, .16); }}
    .chat-panel {{ position: fixed; right: 24px; bottom: 84px; z-index: 30; display: flex; flex-direction: column; width: min(410px, calc(100vw - 28px)); height: min(650px, calc(100vh - 112px)); overflow: hidden; border: 1px solid #cad6e5; border-radius: 18px; background: white; box-shadow: 0 28px 80px rgba(8, 25, 47, .28); transform-origin: bottom right; transition: opacity .18s ease, transform .18s ease; }}
    .chat-panel[hidden] {{ display: none; }}
    .chat-header {{ display: flex; align-items: center; gap: 12px; padding: 16px; background: #10243f; color: white; }}
    .bot-mark {{ display: grid; place-items: center; flex: 0 0 40px; height: 40px; border-radius: 12px; background: linear-gradient(135deg, #4d8cff, #55d6a5); font-size: 20px; }}
    .chat-title {{ min-width: 0; flex: 1; }} .chat-title strong {{ display: block; }} .chat-title span {{ color: #bcd0e5; font-size: 12px; }}
    .chat-close {{ border: 0; background: transparent; color: white; font-size: 24px; cursor: pointer; }}
    .chat-controls {{ display: flex; align-items: center; justify-content: space-between; gap: 10px; padding: 10px 14px; border-bottom: 1px solid var(--line); background: #f7faff; font-size: 12px; color: var(--muted); }}
    .level-toggle {{ display: flex; padding: 3px; border: 1px solid var(--line); border-radius: 999px; background: white; }}
    .level-toggle button {{ border: 0; border-radius: 999px; padding: 5px 9px; background: transparent; color: var(--muted); cursor: pointer; font: inherit; }}
    .level-toggle button.active {{ background: #e7f0ff; color: #174b72; font-weight: 800; }}
    .chat-messages {{ flex: 1; overflow-y: auto; padding: 16px; background: #f7f9fc; }}
    .message {{ max-width: 88%; margin: 0 0 12px; padding: 11px 13px; border-radius: 14px; white-space: pre-wrap; font-size: 14px; }}
    .message.assistant {{ border: 1px solid var(--line); border-bottom-left-radius: 4px; background: white; color: var(--ink); }}
    .message.user {{ margin-left: auto; border-bottom-right-radius: 4px; background: #2167d5; color: white; }}
    .message.status {{ color: var(--muted); font-style: italic; }}
    .quick-prompts {{ display: flex; gap: 7px; overflow-x: auto; padding: 10px 14px 4px; background: white; }}
    .quick-prompts button {{ flex: none; border: 1px solid #b9cbe3; border-radius: 999px; padding: 7px 10px; background: white; color: #245b91; font-size: 12px; cursor: pointer; }}
    .chat-form {{ display: flex; gap: 8px; padding: 10px 14px 6px; background: white; }}
    .chat-form textarea {{ flex: 1; min-height: 44px; max-height: 100px; resize: none; border: 1px solid #b8c6d8; border-radius: 12px; padding: 11px; font: inherit; }}
    .chat-form textarea:focus {{ outline: 3px solid rgba(47, 111, 237, .16); border-color: var(--blue); }}
    .chat-send {{ align-self: end; width: 44px; height: 44px; border: 0; border-radius: 12px; background: var(--blue); color: white; font-size: 18px; cursor: pointer; }}
    .chat-send:disabled {{ opacity: .55; cursor: wait; }}
    .chat-disclaimer {{ margin: 0; padding: 4px 14px 11px; background: white; color: var(--muted); text-align: center; font-size: 10px; }}
    .stage-tabs {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 8px; padding: 6px; margin-bottom: 18px; border: 1px solid var(--line); border-radius: 14px; background: #f0f4f9; }}
    .stage-tab {{ min-height: 48px; border: 1px solid transparent; border-radius: 10px; padding: 9px 12px; background: transparent; color: var(--muted); font: inherit; font-weight: 750; cursor: pointer; transition: background .16s, color .16s, transform .16s; }}
    .stage-tab:hover {{ background: white; color: var(--ink); }}
    .stage-tab.active {{ border-color: #b9cce8; background: white; color: #174b72; box-shadow: 0 5px 16px rgba(21, 58, 95, .09); }}
    .stage-tab span {{ display: inline-grid; place-items: center; width: 23px; height: 23px; margin-right: 6px; border-radius: 50%; background: #dce7f5; font-size: 12px; }}
    .stage-explorer {{ display: grid; grid-template-columns: minmax(240px, .8fr) minmax(300px, 1.2fr); gap: 28px; align-items: center; }}
    .score-ring {{ --score: 85.74; --ring-color: var(--green); position: relative; display: grid; place-items: center; width: 190px; height: 190px; margin: 4px auto; border-radius: 50%; background: conic-gradient(var(--ring-color) calc(var(--score) * 1%), #e7edf5 0); }}
    .score-ring::after {{ content: ''; position: absolute; width: 145px; height: 145px; border-radius: 50%; background: white; }}
    .score-content {{ position: relative; z-index: 1; text-align: center; }} .score-content strong {{ display: block; font-size: 34px; line-height: 1; }} .score-content span {{ color: var(--muted); font-size: 12px; font-weight: 700; text-transform: uppercase; }}
    .stage-copy .eyebrow {{ color: var(--blue); font-size: 12px; font-weight: 850; letter-spacing: .08em; text-transform: uppercase; }}
    .stage-copy h3 {{ margin: 7px 0 8px; font-size: 24px; }} .stage-copy > p {{ margin: 0 0 16px; color: var(--muted); }}
    .stage-facts {{ display: grid; grid-template-columns: repeat(2, 1fr); gap: 9px; margin-bottom: 16px; }}
    .stage-fact {{ padding: 11px; border: 1px solid var(--line); border-radius: 10px; background: #fbfdff; }} .stage-fact span {{ display: block; color: var(--muted); font-size: 11px; text-transform: uppercase; }} .stage-fact strong {{ display: block; margin-top: 3px; font-size: 14px; }}
    .stage-actions {{ display: flex; flex-wrap: wrap; gap: 9px; }}
    .button-primary, .button-secondary {{ border-radius: 10px; padding: 10px 14px; font: inherit; font-weight: 750; cursor: pointer; }}
    .button-primary {{ border: 1px solid var(--blue); background: var(--blue); color: white; }} .button-secondary {{ border: 1px solid #b8c8dc; background: white; color: #245b91; }}
    @media (max-width: 860px) {{ .metrics, .split, .story-grid, .definition-grid {{ grid-template-columns: 1fr 1fr; }} .panel-heading {{ display: block; }} table {{ display: block; overflow-x: auto; white-space: nowrap; }} .stage-explorer {{ grid-template-columns: 1fr; }} }}
    @media (max-width: 560px) {{ .metrics, .split, .story-grid, .definition-grid {{ grid-template-columns: 1fr; }} header h1 {{ font-size: 26px; }} .chat-launcher {{ right: 14px; bottom: 14px; }} .chat-panel {{ right: 14px; bottom: 72px; height: calc(100vh - 90px); }} .stage-tabs {{ display: flex; overflow-x: auto; scroll-snap-type: x mandatory; }} .stage-tab {{ flex: 0 0 165px; scroll-snap-align: start; }} .stage-facts {{ grid-template-columns: 1fr; }} }}
  </style>
</head>
<body>
  <header>
    <h1>Adversarial Medical Diagnostics Dashboard</h1>
    <p>Chest X-ray classification results explained for both technical and non-technical readers.</p>
  </header>
  <main>
    <section class="metrics">
      {metric_card("Clean Baseline", pct(BASELINE.clean_accuracy), "How well the original model worked before attack", "good")}
      {metric_card("After FGSM Attack", pct(BASELINE.adversarial_accuracy), "How much accuracy remained after the attack", "risk")}
      {metric_card("Defense Improvement", improvement, "How much adversarial training raised attacked-image accuracy", "good")}
      {metric_card("Attack Success Drop", attack_reduction, "How much adversarial training lowered successful attacks", "warn")}
    </section>
    <section class="panel wide" id="experiment-explorer">
      <div class="panel-heading">
        <div><h2>Interactive Experiment Explorer</h2><p>Select a stage to see what changed and how the model performed.</p></div>
      </div>
      <div class="stage-tabs" role="tablist" aria-label="Experiment stages">
        <button class="stage-tab active" role="tab" aria-selected="true" data-stage="baseline"><span>1</span>Baseline</button>
        <button class="stage-tab" role="tab" aria-selected="false" data-stage="fgsm"><span>2</span>FGSM Attack</button>
        <button class="stage-tab" role="tab" aria-selected="false" data-stage="randomization"><span>3</span>Randomization</button>
        <button class="stage-tab" role="tab" aria-selected="false" data-stage="adversarial"><span>4</span>Adv. Training</button>
      </div>
      <div class="stage-explorer" role="tabpanel" aria-live="polite">
        <div class="score-ring" id="stage-ring"><div class="score-content"><strong id="stage-score">85.74%</strong><span id="stage-score-label">Clean accuracy</span></div></div>
        <div class="stage-copy">
          <span class="eyebrow" id="stage-eyebrow">Stage 1 · Reference model</span>
          <h3 id="stage-title">Train the baseline ResNet18</h3>
          <p id="stage-description">The model learns to classify clean chest X-rays as NORMAL or PNEUMONIA. This establishes the performance reference before any attack.</p>
          <div class="stage-facts">
            <div class="stage-fact"><span>Input</span><strong id="stage-input">Clean test images</strong></div>
            <div class="stage-fact"><span>Model weights</span><strong id="stage-weights">Trained on clean images</strong></div>
            <div class="stage-fact"><span>Key parameter</span><strong id="stage-parameter">ResNet18 · 8 epochs</strong></div>
            <div class="stage-fact"><span>Conclusion</span><strong id="stage-conclusion">Strong clean performance</strong></div>
          </div>
          <div class="stage-actions"><button class="button-primary" id="ask-stage">Ask AI about this stage</button><button class="button-secondary" id="next-stage">Next stage →</button></div>
        </div>
      </div>
    </section>
    <section class="panel wide" id="executive-summary">
      <div class="panel-heading">
        <h2>Executive Summary</h2>
        <p>The baseline model worked on normal images, but FGSM exposed a real weakness. Adversarial training was the strongest defense.</p>
      </div>
      <div class="callout">
        <strong>Recommended defense: adversarial training</strong>
        It raised attacked-image accuracy from {pct(BASELINE.adversarial_accuracy)} to {pct(ADVERSARIAL_MODEL.adversarial_accuracy)} and reduced attack success from {pct(BASELINE.attack_success_rate)} to {pct(ADVERSARIAL_MODEL.attack_success_rate)}, while clean accuracy stayed essentially unchanged.
      </div>
    </section>
    {plain_english_summary()}
    {metric_explainer()}
    {advisor_story()}
    <div class="grid" id="results">
      {model_comparison()}
      {defense_table()}
      {per_class_table()}
    </div>
  </main>
  <footer>Generated from {html.escape(source_note)}. This dashboard is for cybersecurity and ML robustness analysis only; it is not medical advice.</footer>
  <button class="chat-launcher" id="chat-launcher" aria-controls="chat-panel" aria-expanded="false"><span class="pulse"></span>Ask the Security Assistant</button>
  <aside class="chat-panel" id="chat-panel" aria-label="Security Results Assistant" hidden>
    <div class="chat-header">
      <div class="bot-mark" aria-hidden="true">✦</div>
      <div class="chat-title"><strong>Security Results Assistant</strong><span>Grounded in this experiment's metrics</span></div>
      <button class="chat-close" id="chat-close" aria-label="Close chat">×</button>
    </div>
    <div class="chat-controls"><span>Explanation level</span><div class="level-toggle"><button class="active" data-level="beginner">Beginner</button><button data-level="technical">Technical</button></div></div>
    <div class="chat-messages" id="chat-messages" aria-live="polite">
      <div class="message assistant">Hi! I can explain the FGSM attack, compare defenses, or walk through the measured results. What would you like to understand?</div>
    </div>
    <div class="quick-prompts" aria-label="Suggested questions">
      <button data-question="Explain FGSM in simple words">Explain FGSM</button>
      <button data-question="Which defense performed best and why?">Compare defenses</button>
      <button data-question="Why were NORMAL images more vulnerable?">Class risk</button>
      <button data-question="Summarize the experiment results">Summarize</button>
    </div>
    <form class="chat-form" id="chat-form">
      <textarea id="chat-input" maxlength="2000" rows="1" placeholder="Ask about attacks, defenses, or metrics…" aria-label="Chat question" required></textarea>
      <button class="chat-send" id="chat-send" type="submit" aria-label="Send question">➜</button>
    </form>
    <p class="chat-disclaimer">ML security education only · Not medical advice</p>
  </aside>
  <script>
    (() => {{
      const launcher = document.getElementById('chat-launcher');
      const panel = document.getElementById('chat-panel');
      const close = document.getElementById('chat-close');
      const form = document.getElementById('chat-form');
      const input = document.getElementById('chat-input');
      const send = document.getElementById('chat-send');
      const messages = document.getElementById('chat-messages');
      let level = 'beginner';
      const history = [];
      const stageOrder = ['baseline', 'fgsm', 'randomization', 'adversarial'];
      const stages = {{
        baseline: {{score: 85.74, color: '#16865a', scoreLabel: 'Clean accuracy', eyebrow: 'Stage 1 · Reference model', title: 'Train the baseline ResNet18', description: 'The model learns to classify clean chest X-rays as NORMAL or PNEUMONIA. This establishes the performance reference before any attack.', input: 'Clean test images', weights: 'Trained on clean images', parameter: 'ResNet18 · 8 epochs', conclusion: 'Strong clean performance', question: 'Explain what the baseline model does and why its 85.74% clean accuracy matters.'}},
        fgsm: {{score: 53.69, color: '#c93c3c', scoreLabel: 'Attacked accuracy', eyebrow: 'Stage 2 · Attack evaluation', title: 'Apply the FGSM attack', description: 'FGSM adds small, gradient-directed changes to test images. The unchanged baseline model now makes substantially more errors.', input: 'FGSM test images', weights: 'Unchanged during attack', parameter: 'Epsilon · 0.01', conclusion: 'Accuracy drops 32.05 pp', question: 'Explain how FGSM reduced accuracy from 85.74% to 53.69% without changing model weights.'}},
        randomization: {{score: 57.05, color: '#b87613', scoreLabel: 'Defended accuracy', eyebrow: 'Stage 3 · Preprocessing defense', title: 'Resize attacked images', description: 'Bilinear resizing disrupts some adversarial noise before prediction. It recovers a small number of predictions but leaves substantial risk.', input: 'Resized FGSM images', weights: 'Baseline remains unchanged', parameter: 'Resize delta · 8 px', conclusion: 'Only 3.36 pp recovered', question: 'Explain why resize randomization only improved attacked accuracy from 53.69% to 57.05%.'}},
        adversarial: {{score: 74.04, color: '#2f6fed', scoreLabel: 'Robust accuracy', eyebrow: 'Stage 4 · Stronger defense', title: 'Train against adversarial examples', description: 'A separate ResNet18 learns from clean and FGSM-attacked training images, improving robustness while preserving clean accuracy.', input: 'Clean + FGSM training images', weights: 'Updated using combined loss', parameter: '50% clean · 50% adversarial', conclusion: 'Best evaluated defense', question: 'Explain why adversarial training raised attacked accuracy to 74.04% while clean accuracy stayed 85.90%.'}}
      }};
      let activeStage = 'baseline';

      const toggleChat = (open) => {{
        panel.hidden = !open;
        launcher.setAttribute('aria-expanded', String(open));
        if (open) input.focus();
      }};
      launcher.addEventListener('click', () => toggleChat(panel.hidden));
      close.addEventListener('click', () => toggleChat(false));

      document.querySelectorAll('[data-level]').forEach(button => button.addEventListener('click', () => {{
        level = button.dataset.level;
        document.querySelectorAll('[data-level]').forEach(item => item.classList.toggle('active', item === button));
      }}));

      const addMessage = (content, role, extra = '') => {{
        const node = document.createElement('div');
        node.className = `message ${{role}} ${{extra}}`;
        node.textContent = content;
        messages.appendChild(node);
        messages.scrollTop = messages.scrollHeight;
        return node;
      }};

      const selectStage = (name) => {{
        const stage = stages[name];
        activeStage = name;
        document.querySelectorAll('[data-stage]').forEach(tab => {{
          const active = tab.dataset.stage === name;
          tab.classList.toggle('active', active);
          tab.setAttribute('aria-selected', String(active));
        }});
        const ring = document.getElementById('stage-ring');
        ring.style.setProperty('--score', stage.score);
        ring.style.setProperty('--ring-color', stage.color);
        document.getElementById('stage-score').textContent = `${{stage.score.toFixed(2)}}%`;
        document.getElementById('stage-score-label').textContent = stage.scoreLabel;
        ['eyebrow', 'title', 'description', 'input', 'weights', 'parameter', 'conclusion'].forEach(key => {{
          document.getElementById(`stage-${{key}}`).textContent = stage[key];
        }});
        document.getElementById('next-stage').hidden = name === 'adversarial';
      }};

      const ask = async (question) => {{
        const cleanQuestion = question.trim();
        if (!cleanQuestion || send.disabled) return;
        addMessage(cleanQuestion, 'user');
        input.value = '';
        send.disabled = true;
        const status = addMessage('Reviewing the experiment evidence…', 'assistant', 'status');
        try {{
          const response = await fetch('api/chat', {{
            method: 'POST',
            headers: {{'Content-Type': 'application/json'}},
            body: JSON.stringify({{message: cleanQuestion, level, history: history.slice(-10)}})
          }});
          if (!response.ok) throw new Error(`Request failed (${{response.status}})`);
          const data = await response.json();
          status.remove();
          addMessage(data.answer, 'assistant');
          history.push({{role: 'user', content: cleanQuestion}}, {{role: 'assistant', content: data.answer}});
        }} catch (error) {{
          status.textContent = 'The chat API is not connected. Start the interactive server with: uvicorn src.dashboard.chat_api:app --reload';
          status.classList.remove('status');
        }} finally {{
          send.disabled = false;
          input.focus();
        }}
      }};

      form.addEventListener('submit', event => {{ event.preventDefault(); ask(input.value); }});
      input.addEventListener('keydown', event => {{
        if (event.key === 'Enter' && !event.shiftKey) {{ event.preventDefault(); form.requestSubmit(); }}
      }});
      document.querySelectorAll('[data-question]').forEach(button => button.addEventListener('click', () => ask(button.dataset.question)));
      document.querySelectorAll('[data-stage]').forEach(button => button.addEventListener('click', () => selectStage(button.dataset.stage)));
      document.getElementById('next-stage').addEventListener('click', () => selectStage(stageOrder[stageOrder.indexOf(activeStage) + 1]));
      document.getElementById('ask-stage').addEventListener('click', () => {{ toggleChat(true); ask(stages[activeStage].question); }});
    }})();
  </script>
</body>
</html>
'''


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate the project results dashboard.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--metrics-dir", type=Path, default=DEFAULT_METRICS_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render_dashboard(args.metrics_dir), encoding="utf-8")
    print(f"Dashboard written to {args.output}")


if __name__ == "__main__":
    main()
