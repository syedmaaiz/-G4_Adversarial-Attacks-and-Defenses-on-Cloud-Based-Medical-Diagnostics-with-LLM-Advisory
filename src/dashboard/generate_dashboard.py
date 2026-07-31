"""Generate a static dashboard for attack and defense results."""

from __future__ import annotations

import argparse
import html
from dataclasses import dataclass
from pathlib import Path


DEFAULT_OUTPUT = Path("reports/dashboard.html")


@dataclass(frozen=True)
class ModelResult:
    name: str
    clean_accuracy: float
    adversarial_accuracy: float
    attack_success_rate: float
    prediction_change_rate: float | None = None


@dataclass(frozen=True)
class DefenseResult:
    name: str
    defended_clean_accuracy: float | None
    defended_adversarial_accuracy: float
    recovery_rate: float | None
    notes: str


BASELINE = ModelResult(
    name="Baseline ResNet18",
    clean_accuracy=0.8574,
    adversarial_accuracy=0.5369,
    attack_success_rate=0.3738,
    prediction_change_rate=0.3205,
)

ADVERSARIAL_MODEL = ModelResult(
    name="Adversarially Trained ResNet18",
    clean_accuracy=0.8590,
    adversarial_accuracy=0.7404,
    attack_success_rate=0.1381,
    prediction_change_rate=0.1186,
)

RANDOMIZATION = DefenseResult(
    name="Defensive Randomization",
    defended_clean_accuracy=0.8381,
    defended_adversarial_accuracy=0.5705,
    recovery_rate=0.1050,
    notes="Small robustness gain, but weak recovery against NORMAL-image attacks.",
)

ADVERSARIAL_TRAINING = DefenseResult(
    name="Adversarial Training",
    defended_clean_accuracy=0.8590,
    defended_adversarial_accuracy=0.7404,
    recovery_rate=None,
    notes="Strongest defense: improves robustness while preserving clean accuracy.",
)

PER_CLASS_ROWS = [
    ("NORMAL", 0.6325, 0.0085, 0.0128, 0.5034),
    ("PNEUMONIA", 0.9923, 0.8538, 0.9051, 0.0000),
]


def pct(value: float) -> str:
    """Format a ratio as a percentage."""
    return f"{value * 100:.2f}%"


def delta(new: float, old: float) -> str:
    """Format a percentage-point delta."""
    sign = "+" if new >= old else ""
    return f"{sign}{(new - old) * 100:.2f} pp"


def bar(value: float, label: str, tone: str = "blue") -> str:
    """Render a horizontal metric bar."""
    width = max(0, min(100, value * 100))
    return f'''
    <div class="bar-row" aria-label="{html.escape(label)} {pct(value)}">
      <div class="bar-label"><span>{html.escape(label)}</span><strong>{pct(value)}</strong></div>
      <div class="bar-track"><div class="bar-fill {tone}" style="width: {width:.2f}%"></div></div>
    </div>
    '''


def metric_card(title: str, value: str, caption: str, tone: str = "neutral") -> str:
    """Render a compact metric card."""
    return f'''
    <section class="metric-card {tone}">
      <span>{html.escape(title)}</span>
      <strong>{html.escape(value)}</strong>
      <p>{html.escape(caption)}</p>
    </section>
    '''


def model_comparison() -> str:
    """Render baseline vs adversarial-training comparison."""
    return f'''
    <section class="panel wide">
      <div class="panel-heading">
        <h2>Baseline vs Adversarial Training</h2>
        <p>FGSM epsilon = 0.01, test set = 624 images</p>
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
    </section>
    '''


def defense_table() -> str:
    """Render defense summary table."""
    rows = []
    for defense in (RANDOMIZATION, ADVERSARIAL_TRAINING):
        clean = pct(defense.defended_clean_accuracy) if defense.defended_clean_accuracy is not None else "N/A"
        recovery = pct(defense.recovery_rate) if defense.recovery_rate is not None else "N/A"
        rows.append(
            f"""
            <tr>
              <td>{html.escape(defense.name)}</td>
              <td>{clean}</td>
              <td>{pct(defense.defended_adversarial_accuracy)}</td>
              <td>{recovery}</td>
              <td>{html.escape(defense.notes)}</td>
            </tr>
            """
        )
    return f'''
    <section class="panel wide">
      <div class="panel-heading">
        <h2>Defense Comparison</h2>
        <p>Randomization gives a small gain; adversarial training gives the strongest robustness improvement.</p>
      </div>
      <table>
        <thead>
          <tr>
            <th>Defense</th>
            <th>Clean Accuracy</th>
            <th>Adversarial Accuracy</th>
            <th>Recovery Rate</th>
            <th>Interpretation</th>
          </tr>
        </thead>
        <tbody>{''.join(rows)}</tbody>
      </table>
    </section>
    '''


def per_class_table() -> str:
    """Render per-class vulnerability table."""
    rows = []
    for class_name, clean, attacked, randomized, adv_trained_attack_success in PER_CLASS_ROWS:
        rows.append(
            f"""
            <tr>
              <td>{class_name}</td>
              <td>{pct(clean)}</td>
              <td>{pct(attacked)}</td>
              <td>{pct(randomized)}</td>
              <td>{pct(adv_trained_attack_success)}</td>
            </tr>
            """
        )
    return f'''
    <section class="panel wide">
      <div class="panel-heading">
        <h2>Per-Class Behavior</h2>
        <p>NORMAL images were the main vulnerability under FGSM. Adversarial training reduced, but did not eliminate, that class-specific risk.</p>
      </div>
      <table>
        <thead>
          <tr>
            <th>Class</th>
            <th>Clean Accuracy</th>
            <th>Baseline FGSM Accuracy</th>
            <th>Randomized FGSM Accuracy</th>
            <th>Adv-Trained Attack Success</th>
          </tr>
        </thead>
        <tbody>{''.join(rows)}</tbody>
      </table>
    </section>
    '''


def render_dashboard() -> str:
    """Render the full dashboard HTML."""
    improvement = delta(ADVERSARIAL_MODEL.adversarial_accuracy, BASELINE.adversarial_accuracy)
    attack_reduction = delta(ADVERSARIAL_MODEL.attack_success_rate, BASELINE.attack_success_rate)
    return f'''<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Adversarial Medical Diagnostics Dashboard</title>
  <style>
    :root {{
      --bg: #f4f7fb;
      --ink: #152033;
      --muted: #627089;
      --panel: #ffffff;
      --line: #d9e1ec;
      --blue: #2f6fed;
      --green: #16865a;
      --red: #c93c3c;
      --amber: #b87613;
      --shadow: 0 18px 45px rgba(21, 32, 51, 0.11);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: var(--ink);
      background: var(--bg);
      line-height: 1.45;
    }}
    header {{
      padding: 34px 28px 20px;
      background: #10243f;
      color: white;
    }}
    header h1 {{ margin: 0 0 8px; font-size: 32px; letter-spacing: 0; }}
    header p {{ margin: 0; max-width: 900px; color: #cfdbeb; }}
    main {{ padding: 24px; max-width: 1180px; margin: 0 auto; }}
    .metrics {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 14px; margin-bottom: 18px; }}
    .metric-card, .panel {{ background: var(--panel); border: 1px solid var(--line); border-radius: 8px; box-shadow: var(--shadow); }}
    .metric-card {{ padding: 16px; min-height: 130px; }}
    .metric-card span {{ color: var(--muted); font-size: 13px; font-weight: 700; text-transform: uppercase; }}
    .metric-card strong {{ display: block; font-size: 30px; margin: 8px 0 4px; }}
    .metric-card p {{ margin: 0; color: var(--muted); font-size: 14px; }}
    .metric-card.good strong {{ color: var(--green); }}
    .metric-card.warn strong {{ color: var(--amber); }}
    .metric-card.risk strong {{ color: var(--red); }}
    .grid {{ display: grid; grid-template-columns: 1fr; gap: 18px; }}
    .panel {{ padding: 20px; }}
    .panel-heading {{ display: flex; justify-content: space-between; gap: 20px; align-items: end; border-bottom: 1px solid var(--line); padding-bottom: 12px; margin-bottom: 16px; }}
    .panel-heading h2 {{ margin: 0; font-size: 21px; }}
    .panel-heading p {{ margin: 0; color: var(--muted); max-width: 560px; }}
    .split {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 28px; }}
    h3 {{ margin: 0 0 14px; font-size: 17px; }}
    .bar-row {{ margin-bottom: 14px; }}
    .bar-label {{ display: flex; justify-content: space-between; gap: 12px; font-size: 14px; margin-bottom: 6px; }}
    .bar-label span {{ color: var(--muted); }}
    .bar-track {{ height: 14px; background: #e8eef6; border-radius: 999px; overflow: hidden; }}
    .bar-fill {{ height: 100%; border-radius: inherit; }}
    .bar-fill.blue {{ background: var(--blue); }}
    .bar-fill.green {{ background: var(--green); }}
    .bar-fill.red {{ background: var(--red); }}
    .bar-fill.amber {{ background: var(--amber); }}
    table {{ width: 100%; border-collapse: collapse; font-size: 14px; }}
    th, td {{ border-bottom: 1px solid var(--line); padding: 11px 10px; text-align: left; vertical-align: top; }}
    th {{ color: var(--muted); font-size: 12px; text-transform: uppercase; }}
    .callout {{ border-left: 5px solid var(--blue); padding: 16px 18px; background: #eef4ff; border-radius: 6px; margin-top: 18px; }}
    .callout strong {{ display: block; margin-bottom: 6px; }}
    footer {{ color: var(--muted); padding: 18px 24px 34px; max-width: 1180px; margin: 0 auto; font-size: 13px; }}
    @media (max-width: 860px) {{
      .metrics {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
      .split {{ grid-template-columns: 1fr; }}
      .panel-heading {{ display: block; }}
      table {{ display: block; overflow-x: auto; white-space: nowrap; }}
    }}
    @media (max-width: 520px) {{ .metrics {{ grid-template-columns: 1fr; }} header h1 {{ font-size: 26px; }} }}
  </style>
</head>
<body>
  <header>
    <h1>Adversarial Medical Diagnostics Dashboard</h1>
    <p>Chest X-ray classification results for baseline training, FGSM attack evaluation, defensive randomization, adversarial training, and LLM advisory conclusions.</p>
  </header>
  <main>
    <section class="metrics">
      {metric_card("Clean Baseline", pct(BASELINE.clean_accuracy), "ResNet18 test accuracy before attack", "good")}
      {metric_card("FGSM Accuracy", pct(BASELINE.adversarial_accuracy), "Baseline accuracy after epsilon 0.01 attack", "risk")}
      {metric_card("Adv Training Gain", improvement, "Adversarial accuracy improvement", "good")}
      {metric_card("Attack Success Drop", attack_reduction, "Lower is better after defense", "warn")}
    </section>
    <section class="panel wide">
      <div class="panel-heading">
        <h2>Executive Summary</h2>
        <p>The model is useful on clean images but vulnerable to FGSM. Defensive randomization helps slightly; adversarial training is the recommended defense.</p>
      </div>
      <div class="callout">
        <strong>Recommended defense: adversarial training</strong>
        It raised adversarial accuracy from {pct(BASELINE.adversarial_accuracy)} to {pct(ADVERSARIAL_MODEL.adversarial_accuracy)} and reduced standard attack success from {pct(BASELINE.attack_success_rate)} to {pct(ADVERSARIAL_MODEL.attack_success_rate)}, while clean accuracy stayed essentially unchanged.
      </div>
    </section>
    <div class="grid">
      {model_comparison()}
      {defense_table()}
      {per_class_table()}
    </div>
  </main>
  <footer>
    Generated from project metrics. This dashboard is for cybersecurity and ML robustness analysis only; it is not medical advice.
  </footer>
</body>
</html>
'''


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate the project results dashboard.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render_dashboard(), encoding="utf-8")
    print(f"Dashboard written to {args.output}")


if __name__ == "__main__":
    main()
