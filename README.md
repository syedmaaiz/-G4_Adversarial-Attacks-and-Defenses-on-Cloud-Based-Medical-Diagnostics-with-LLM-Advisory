# Adversarial Attacks and Defenses on Cloud-Based Medical Diagnostics

Python project for evaluating adversarial attacks and defenses on a cloud-hosted chest X-ray classifier, with an LLM advisory component for defense recommendations.

For a complete fresh Colab walkthrough, see [docs/COLAB_GRADER_RUNBOOK.md](docs/COLAB_GRADER_RUNBOOK.md).
For a ready-to-run Colab notebook, use [notebooks/Grader_Colab_Runbook.ipynb](notebooks/Grader_Colab_Runbook.ipynb).

## Project Goals

1. Train a baseline medical image classifier on clean chest X-ray images.
2. Deploy the trained model behind a simulated cloud prediction API.
3. Generate test-time adversarial examples against the classifier.
4. Evaluate how attacks affect model predictions and confidence.
5. Implement defenses such as defensive randomization and adversarial training.
6. Use an LLM advisor to recommend defensive strategies based on attack context.

## Team Roles

- Cloud Architect: dataset pipeline, baseline model training, simulated cloud API.
- Attacker: adversarial attack implementation and attack evaluation.
- Defender: defense methods, LLM advisory workflow, before/after comparison.

## Folder Structure

```text
.
|-- src/
|   |-- data/          # Dataset loading, preprocessing, train/val/test splits
|   |-- models/        # Model definitions, training, checkpoint loading
|   |-- attacks/       # FGSM/PGD or other adversarial attack methods
|   |-- defenses/      # Defensive randomization, adversarial training, ADV-SVM experiments
|   |-- api/           # Simulated cloud API for model inference
|   |-- llm_advisor/   # LLM prompts, advisory logic, mock fallback
|   `-- evaluation/    # Metrics, plots, attack/defense reports
|-- data/
|   |-- raw/           # Local raw dataset files, not committed
|   `-- processed/     # Processed dataset files, not committed
|-- artifacts/
|   |-- models/        # Trained model checkpoints, not committed
|   `-- figures/       # Generated plots and visual outputs
|-- notebooks/         # Exploration and experiments
|-- reports/           # Draft result summaries and course deliverables
|-- docs/              # Planning notes and architecture documentation
`-- tests/             # Unit/integration tests
```

## Dataset Layout

Place the chest X-ray dataset under `data/raw/` like this:

```text
data/raw/
|-- train/
|   |-- NORMAL/
|   `-- PNEUMONIA/
|-- test/
|   |-- NORMAL/
|   `-- PNEUMONIA/
`-- val/               # Optional original Kaggle validation split
    |-- NORMAL/
    `-- PNEUMONIA/
```

The training code creates its own validation split from `train/` because the original Kaggle `val/` folder is very small.

## First Run

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Train the baseline classifier:

```powershell
python -m src.models.train_baseline --epochs 5 --batch-size 32
```

For better accuracy, use pretrained weights if your environment can download them. This project now uses inverse-frequency class weights by default to reduce bias toward the larger `PNEUMONIA` class:

```powershell
python -m src.models.train_baseline --epochs 8 --batch-size 32 --pretrained
```

To disable weighted loss for comparison:

```powershell
python -m src.models.train_baseline --epochs 8 --batch-size 32 --pretrained --no-class-weights
```

## Google Colab Update Flow

If you already cloned the repo in Colab, pull the latest training improvement before rerunning:

```python
%cd /content/-G4_Adversarial-Attacks-and-Defenses-on-Cloud-Based-Medical-Diagnostics-with-LLM-Advisory
!git pull
!rm -f artifacts/models/baseline_model.pt
!python -m src.models.train_baseline --epochs 8 --batch-size 32 --pretrained
```

The improved run should print a line like:

```text
Class weights: [about 1.94, about 0.67]
```

Those numbers confirm that the `NORMAL` class is being weighted more heavily during training.

Run the simulated cloud API after a checkpoint exists:

```powershell
uvicorn src.api.app:app --reload
```

Evaluate FGSM attack success after training:

```powershell
python -m src.attacks.evaluate_fgsm --epsilon 0.03
```



## FGSM Evaluation in Colab

After pulling the latest code and training a checkpoint, run the full test set:

```python
!python -m src.attacks.evaluate_fgsm --epsilon 0.01
!python -m src.attacks.evaluate_fgsm --epsilon 0.02
!python -m src.attacks.evaluate_fgsm --epsilon 0.03
!python -m src.attacks.evaluate_fgsm --epsilon 0.05
```

Use `Standard attack success rate` in the report. It measures how many clean-correct predictions become wrong after the attack. FGSM adversarial images are clamped back to the valid ImageNet-normalized pixel range before evaluation.


## Defense Evaluation in Colab

After FGSM evaluation, compare the baseline attack with defensive randomization:

```python
!python -m src.defenses.evaluate_randomization --epsilon 0.01
```

For a small comparison sweep:

```python
!python -m src.defenses.evaluate_randomization --epsilon 0.01 --resize-delta 8
!python -m src.defenses.evaluate_randomization --epsilon 0.01 --resize-delta 12
!python -m src.defenses.evaluate_randomization --epsilon 0.01 --resize-delta 20
```

For the report, compare `Clean accuracy`, `Adversarial accuracy`, and `Defended adversarial accuracy`. Also include `Defense recovery rate`, which measures how many successful FGSM attacks were corrected by the defense.

## Adversarial Training in Colab

Adversarial training creates a second checkpoint at `artifacts/models/adversarial_model.pt` using a mix of clean and FGSM examples:

```python
!python -m src.defenses.train_adversarial --epochs 8 --batch-size 32 --pretrained --epsilon 0.01
```

Evaluate the adversarially trained model against FGSM:

```python
!python -m src.attacks.evaluate_fgsm --checkpoint artifacts/models/adversarial_model.pt --epsilon 0.01
```

Compare that output to the baseline checkpoint:

```python
!python -m src.attacks.evaluate_fgsm --checkpoint artifacts/models/baseline_model.pt --epsilon 0.01
```

For the report, compare baseline `Adversarial accuracy` and `Standard attack success rate` against the same metrics from `adversarial_model.pt`.

## LLM Advisory

The advisor supports Ollama, Gemini, and OpenAI-compatible chat completions. Ollama is the recommended no-quota option for this project. Set your API key locally or in Colab before calling the advisor:

```powershell
$env:LLM_PROVIDER="gemini"
$env:GEMINI_API_KEY="your_gemini_key_here"
$env:GEMINI_MODEL="gemini-2.0-flash"
```

In Colab, use:

```python
import os
os.environ["LLM_PROVIDER"] = "gemini"
os.environ["GEMINI_API_KEY"] = "your_gemini_key_here"
os.environ["GEMINI_MODEL"] = "gemini-2.0-flash"
```

Run the advisor with the latest project metrics:

```powershell
python -m src.llm_advisor.recommend
```

In Colab:

```python
!python -m src.llm_advisor.recommend
```

The advisor sends the project's attack and defense metrics to the selected LLM provider and asks for a concise defense recommendation. If no API key is configured, the command returns fallback guidance so the demo still works.

The API also exposes the advisor:

```text
POST /advisor/recommend
```


## Ollama Advisory in Colab

Ollama can run the advisor locally without OpenAI or Gemini quota. In Colab, pull the latest code first:

```python
%cd /content/-G4_Adversarial-Attacks-and-Defenses-on-Cloud-Based-Medical-Diagnostics-with-LLM-Advisory
!git pull
```

Install and start Ollama:

```python
!curl -fsSL https://ollama.com/install.sh | sh
!nohup ollama serve > ollama.log 2>&1 &
```

Pull a small model:

```python
!ollama pull llama3.2:1b
```

Run the advisor through Ollama:

```python
import os
os.environ["LLM_PROVIDER"] = "ollama"
os.environ["OLLAMA_MODEL"] = "llama3.2:1b"
os.environ["OLLAMA_BASE_URL"] = "http://127.0.0.1:11434"
os.environ["LLM_TIMEOUT_SECONDS"] = "180"
```

```python
!python -m src.llm_advisor.recommend
```

If `llama3.2:1b` is too weak or unavailable, try `qwen2.5:1.5b`:

```python
!ollama pull qwen2.5:1.5b
os.environ["OLLAMA_MODEL"] = "qwen2.5:1.5b"
!python -m src.llm_advisor.recommend
```

## Results Dashboard

The `dashboard-ui` branch includes a static HTML dashboard for understanding the project results. Generate it with:

```powershell
python -m src.dashboard.generate_dashboard
```

This writes:

```text
reports/dashboard.html
```

### Interactive dashboard chatbot

The dashboard also includes a grounded Security Results Assistant with beginner and
technical explanation modes, suggested questions, conversation history, and a safe
fallback when the configured LLM is unavailable. Start the dashboard and chat API with:

```powershell
uvicorn src.dashboard.chat_api:app --reload
```

Then open `http://127.0.0.1:8000`. The chatbot uses the same `LLM_PROVIDER` configuration
as the advisor. For local Ollama:

```powershell
$env:LLM_PROVIDER="ollama"
$env:OLLAMA_MODEL="qwen2.5:1.5b"
$env:OLLAMA_BASE_URL="http://127.0.0.1:11434"
uvicorn src.dashboard.chat_api:app --reload
```

Opening `reports/dashboard.html` directly still displays the full report and chat UI,
but live chat requests require the FastAPI server. The assistant is restricted to ML
security education and project metrics; it does not provide medical advice.

In Colab, display it inline:

```python
!python -m src.dashboard.generate_dashboard
from IPython.display import HTML, display
with open("reports/dashboard.html", "r", encoding="utf-8") as f:
    display(HTML(f.read()))
```

The dashboard summarizes clean accuracy, FGSM adversarial accuracy, defensive randomization, adversarial training, per-class vulnerability, and the final recommended defense.

### LLM Advisory Modes

Run pre-defense advice after the FGSM attack but before defense implementation:

```python
!python -m src.llm_advisor.recommend --mode pre-defense
```

Run post-defense advice after randomization and adversarial training have been evaluated:

```python
!python -m src.llm_advisor.recommend --mode post-defense
```

Use both outputs in the project story: pre-defense advice explains what defenses to try, while post-defense advice explains which defense worked best.
