# Implementation Roadmap

## Phase 1: Setup

- Confirm dataset choice.
- Create Python virtual environment.
- Install dependencies.
- Verify all teammates can run the same starter commands.

## Phase 2: Data Pipeline

- Load chest X-ray images from `data/raw/`.
- Resize and normalize images.
- Create train, validation, and test splits.
- Save reusable preprocessing utilities in `src/data/`.

## Phase 3: Baseline Model

- Train a clean-image classifier.
- Save the trained checkpoint to `artifacts/models/`.
- Record clean accuracy, precision, recall, F1-score, and confusion matrix.

## Phase 4: Simulated Cloud API

- Load the trained checkpoint.
- Expose a `/predict` endpoint with FastAPI.
- Return predicted class and confidence.

## Phase 5: Adversarial Attack

- Implement FGSM first.
- Compare clean image prediction vs adversarial image prediction.
- Track attack success rate and confidence changes.

## Phase 6: Defense

- Implement defensive randomization.
- Add adversarial training if time allows.
- Compare clean accuracy, attacked accuracy, and defended accuracy.

## Phase 7: LLM Advisor

- Summarize attack context for the advisor.
- Use a mock advisor first so the project works without an API key.
- Add optional real LLM API integration later.

## Phase 8: Final Demo

- Show clean prediction.
- Show adversarial prediction.
- Show defended prediction.
- Show LLM recommendation.
- Generate final charts and metrics.

