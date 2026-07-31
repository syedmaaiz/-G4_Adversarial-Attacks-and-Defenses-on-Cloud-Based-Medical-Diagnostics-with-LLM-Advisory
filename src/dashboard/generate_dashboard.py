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


def render_dashboard() -> str:
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
    header {{ padding: 34px 28px 20px; background: #10243f; color: white; }}
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
    @media (max-width: 860px) {{ .metrics, .split, .story-grid, .definition-grid {{ grid-template-columns: 1fr 1fr; }} .panel-heading {{ display: block; }} table {{ display: block; overflow-x: auto; white-space: nowrap; }} }}
    @media (max-width: 560px) {{ .metrics, .split, .story-grid, .definition-grid {{ grid-template-columns: 1fr; }} header h1 {{ font-size: 26px; }} }}
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
    <section class="panel wide">
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
    <div class="grid">
      {model_comparison()}
      {defense_table()}
      {per_class_table()}
    </div>
  </main>
  <footer>Generated from project metrics. This dashboard is for cybersecurity and ML robustness analysis only; it is not medical advice.</footer>
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



