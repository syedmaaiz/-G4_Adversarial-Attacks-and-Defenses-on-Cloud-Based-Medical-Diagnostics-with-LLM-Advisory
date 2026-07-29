# Adversarial Attacks and Defenses on Cloud-Based Medical Diagnostics

Python project for evaluating adversarial attacks and defenses on a cloud-hosted chest X-ray classifier, with an LLM advisory component for defense recommendations.

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
├── src/
│   ├── data/          # Dataset loading, preprocessing, train/val/test splits
│   ├── models/        # Model definitions, training, checkpoint loading
│   ├── attacks/       # FGSM/PGD or other adversarial attack methods
│   ├── defenses/      # Defensive randomization, adversarial training, ADV-SVM experiments
│   ├── api/           # Simulated cloud API for model inference
│   ├── llm_advisor/   # LLM prompts, advisory logic, mock fallback
│   └── evaluation/    # Metrics, plots, attack/defense reports
├── data/
│   ├── raw/           # Local raw dataset files, not committed
│   └── processed/     # Processed dataset files, not committed
├── artifacts/
│   ├── models/        # Trained model checkpoints, not committed
│   └── figures/       # Generated plots and visual outputs
├── notebooks/         # Exploration and experiments
├── reports/           # Draft result summaries and course deliverables
├── docs/              # Planning notes and architecture documentation
└── tests/             # Unit/integration tests
```

## First Run

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Then start with:

```powershell
python -m src.models.train_baseline
```

The training script is currently a placeholder until the dataset is selected and placed under `data/raw/`.

