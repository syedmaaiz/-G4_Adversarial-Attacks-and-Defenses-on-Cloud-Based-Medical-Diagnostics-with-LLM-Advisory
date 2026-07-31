# Colab Grader Runbook

This runbook starts from a fresh Google Colab runtime and reproduces the project through the final dashboard. It assumes the grader has access to the dataset zip in Google Drive as `raw.zip`.

Expected total runtime on Colab GPU: about 45-75 minutes, depending on GPU and Ollama model download speed.

## Before Running

In Colab, choose:

```text
Runtime -> Change runtime type -> T4 GPU
```

The dataset zip should be in Google Drive:

```text
/content/drive/MyDrive/raw.zip
```

The zip may unzip as either:

```text
data/raw/train/...
```

or:

```text
data/raw/raw/train/...
```

The setup cell below handles both cases.

## Cell 1: Mount Drive, Clone Repo, Checkout Dashboard Branch, Install Dependencies

```python
from google.colab import drive

drive.mount('/content/drive')

REPO_URL = "https://github.com/syedmaaiz/-G4_Adversarial-Attacks-and-Defenses-on-Cloud-Based-Medical-Diagnostics-with-LLM-Advisory.git"
REPO_DIR = "/content/-G4_Adversarial-Attacks-and-Defenses-on-Cloud-Based-Medical-Diagnostics-with-LLM-Advisory"

import os

if not os.path.exists(REPO_DIR):
    !git clone --branch dashboard-ui {REPO_URL}
else:
    %cd {REPO_DIR}
    !git fetch
    !git checkout dashboard-ui
    !git pull

%cd {REPO_DIR}
!pip install -r requirements.txt
```

## Cell 2: Unzip Dataset And Verify Folder Layout

```python
import os

%cd /content/-G4_Adversarial-Attacks-and-Defenses-on-Cloud-Based-Medical-Diagnostics-with-LLM-Advisory

DATA_ZIP = "/content/drive/MyDrive/raw.zip"
assert os.path.exists(DATA_ZIP), f"Dataset zip not found at {DATA_ZIP}"

!rm -rf data/raw
!mkdir -p data/raw
!unzip -q "{DATA_ZIP}" -d data/raw

# Some zips create data/raw/raw/train instead of data/raw/train. Fix that automatically.
if os.path.exists("data/raw/raw/train"):
    !mv data/raw/raw/* data/raw/
    !rmdir data/raw/raw || true

print("Dataset folders:")
!find data/raw -maxdepth 3 -type d | sort

assert os.path.exists("data/raw/train/NORMAL")
assert os.path.exists("data/raw/train/PNEUMONIA")
assert os.path.exists("data/raw/test/NORMAL")
assert os.path.exists("data/raw/test/PNEUMONIA")
print("Dataset layout is correct.")
```

## Cell 3: Train Baseline Model

This trains the class-weighted baseline ResNet18 model. It creates:

```text
artifacts/models/baseline_model.pt
```

```python
%cd /content/-G4_Adversarial-Attacks-and-Defenses-on-Cloud-Based-Medical-Diagnostics-with-LLM-Advisory

!rm -f artifacts/models/baseline_model.pt
!python -m src.models.train_baseline --epochs 8 --batch-size 32 --pretrained
```

Expected result should be close to:

```text
Test accuracy around 0.85-0.86
```

## Cell 4: Evaluate FGSM Attack On Baseline

This shows how much the attack hurts the baseline model.

```python
%cd /content/-G4_Adversarial-Attacks-and-Defenses-on-Cloud-Based-Medical-Diagnostics-with-LLM-Advisory

!python -m src.attacks.evaluate_fgsm --checkpoint artifacts/models/baseline_model.pt --epsilon 0.01
```

Expected result should be close to:

```text
Clean accuracy: 0.8574
Adversarial accuracy: 0.5369
Standard attack success rate: 0.3738
```

## Cell 5: Evaluate Simple Defense, Defensive Randomization

This is a lightweight defense. It should help only a little.

```python
%cd /content/-G4_Adversarial-Attacks-and-Defenses-on-Cloud-Based-Medical-Diagnostics-with-LLM-Advisory

!python -m src.defenses.evaluate_randomization --epsilon 0.01 --resize-delta 8
```

Expected result should be close to:

```text
Adversarial accuracy: 0.5369
Defended adversarial accuracy: 0.5705
Defense recovery rate: 0.1050
```

## Cell 6: Train Stronger Defense, Adversarial Training

This trains a second model using clean images plus FGSM-attacked images. It creates:

```text
artifacts/models/adversarial_model.pt
```

```python
%cd /content/-G4_Adversarial-Attacks-and-Defenses-on-Cloud-Based-Medical-Diagnostics-with-LLM-Advisory

!rm -f artifacts/models/adversarial_model.pt
!python -m src.defenses.train_adversarial --epochs 8 --batch-size 32 --pretrained --epsilon 0.01
```

Expected result should be close to:

```text
Test accuracy around 0.8590
Test adversarial accuracy at epsilon 0.01 around 0.7404
```

## Cell 7: Compare Baseline vs Adversarially Trained Model

```python
%cd /content/-G4_Adversarial-Attacks-and-Defenses-on-Cloud-Based-Medical-Diagnostics-with-LLM-Advisory

print("BASELINE MODEL AGAINST FGSM")
!python -m src.attacks.evaluate_fgsm --checkpoint artifacts/models/baseline_model.pt --epsilon 0.01

print("\nADVERSARIALLY TRAINED MODEL AGAINST FGSM")
!python -m src.attacks.evaluate_fgsm --checkpoint artifacts/models/adversarial_model.pt --epsilon 0.01
```

Expected comparison:

```text
Baseline adversarial accuracy: about 0.5369
Adversarially trained adversarial accuracy: about 0.7404

Baseline attack success rate: about 0.3738
Adversarially trained attack success rate: about 0.1381
```

## Cell 8: Install Ollama And Run LLM Advisory

This avoids OpenAI/Gemini quota issues by running a small local LLM inside Colab.

```python
%cd /content/-G4_Adversarial-Attacks-and-Defenses-on-Cloud-Based-Medical-Diagnostics-with-LLM-Advisory

!sudo apt-get update -qq
!sudo apt-get install -y zstd > /dev/null
!curl -fsSL https://ollama.com/install.sh | sh
!nohup ollama serve > ollama.log 2>&1 &

import time, os

time.sleep(5)
!curl -s http://127.0.0.1:11434/api/tags

!ollama pull qwen2.5:1.5b

os.environ["LLM_PROVIDER"] = "ollama"
os.environ["OLLAMA_MODEL"] = "qwen2.5:1.5b"
os.environ["OLLAMA_BASE_URL"] = "http://127.0.0.1:11434"
os.environ["LLM_TIMEOUT_SECONDS"] = "240"
```

## Cell 9: Run Pre-Defense And Post-Defense LLM Advice

Pre-defense advice only sees the baseline attack result. Post-defense advice sees all defense results.

```python
%cd /content/-G4_Adversarial-Attacks-and-Defenses-on-Cloud-Based-Medical-Diagnostics-with-LLM-Advisory

print("PRE-DEFENSE LLM ADVICE")
!python -m src.llm_advisor.recommend --mode pre-defense

print("\nPOST-DEFENSE LLM ADVICE")
!python -m src.llm_advisor.recommend --mode post-defense
```

The expected recommendation is that adversarial training should be the main defense, while defensive randomization is only a weaker comparison defense.

## Cell 10: Generate And Display Dashboard

```python
%cd /content/-G4_Adversarial-Attacks-and-Defenses-on-Cloud-Based-Medical-Diagnostics-with-LLM-Advisory

!python -m src.dashboard.generate_dashboard

from IPython.display import HTML, display

with open("reports/dashboard.html", "r", encoding="utf-8") as f:
    display(HTML(f.read()))
```

The dashboard should show:

- Clean baseline accuracy
- Accuracy drop after FGSM attack
- Defensive randomization's small improvement
- Adversarial training's larger improvement
- Per-class vulnerability, especially NORMAL images
- Pre-defense and post-defense LLM advisory roles
- Plain-English explanations for non-statistics readers

## Optional Cell 11: Save Trained Checkpoints To Google Drive

Run this if the grader wants to preserve trained models after the Colab runtime disconnects.

```python
%cd /content/-G4_Adversarial-Attacks-and-Defenses-on-Cloud-Based-Medical-Diagnostics-with-LLM-Advisory

!mkdir -p "/content/drive/MyDrive/adversarial_project_checkpoints"
!cp artifacts/models/baseline_model.pt "/content/drive/MyDrive/adversarial_project_checkpoints/"
!cp artifacts/models/adversarial_model.pt "/content/drive/MyDrive/adversarial_project_checkpoints/"
!ls -lh "/content/drive/MyDrive/adversarial_project_checkpoints"
```

## Important Notes

- Do not commit or share API keys in notebooks.
- The project can run the advisory step through Ollama without paid cloud API quota.
- The dataset and trained checkpoints are intentionally not committed to GitHub.
- If exact numbers differ slightly, that is acceptable because training has randomness and Colab GPU environments can vary.
