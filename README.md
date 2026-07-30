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
python -m src.attacks.evaluate_fgsm --epsilon 0.03 --max-batches 10
```


