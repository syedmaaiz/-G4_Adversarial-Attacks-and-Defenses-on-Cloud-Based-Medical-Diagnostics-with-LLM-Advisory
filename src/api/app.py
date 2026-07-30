"""FastAPI app for simulated cloud model inference."""

from io import BytesIO
from pathlib import Path
from typing import Any

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field
from PIL import Image, UnidentifiedImageError

from src.defenses.randomization import apply_defensive_randomization
from src.llm_advisor.advisor import recommend_defense
from src.llm_advisor.prompts import DEFAULT_PROJECT_METRICS
from src.models.predict import DEFAULT_CHECKPOINT, load_model, predict_image


app = FastAPI(title="Medical Diagnostics Cloud API")
_model = None
_class_names: list[str] | None = None
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class AdvisorRequest(BaseModel):
    metrics: dict[str, Any] = Field(
        default_factory=lambda: DEFAULT_PROJECT_METRICS,
        description="Attack and defense metrics for the LLM advisor.",
    )
    use_fallback: bool = Field(
        default=True,
        description="Return deterministic fallback advice if the LLM API is unavailable.",
    )


def get_loaded_model(checkpoint_path: Path = DEFAULT_CHECKPOINT):
    """Lazy-load the model so the app can start even before training exists."""
    global _model, _class_names
    if _model is None or _class_names is None:
        if not checkpoint_path.exists():
            raise HTTPException(
                status_code=503,
                detail=(
                    "Model checkpoint not found. Train the baseline first with "
                    "python -m src.models.train_baseline"
                ),
            )
        _model, _class_names, _ = load_model(checkpoint_path, _device)
    return _model, _class_names


async def read_upload_image(file: UploadFile) -> Image.Image:
    """Read an uploaded image into PIL."""
    try:
        contents = await file.read()
        return Image.open(BytesIO(contents)).convert("RGB")
    except UnidentifiedImageError as exc:
        raise HTTPException(status_code=400, detail="Uploaded file is not an image.") from exc


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    use_defense: bool = False,
) -> dict[str, object]:
    """Predict an uploaded X-ray image with optional defensive randomization."""
    model, class_names = get_loaded_model()
    image = await read_upload_image(file)
    if use_defense:
        image = apply_defensive_randomization(image)
    result = predict_image(model, image, class_names, _device)
    result["defense_applied"] = use_defense
    return result


@app.post("/advisor/recommend")
def advisor_recommend(request: AdvisorRequest) -> dict[str, object]:
    """Ask the LLM advisor to recommend an adversarial defense."""
    try:
        recommendation = recommend_defense(
            request.metrics,
            use_fallback=request.use_fallback,
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return {
        "recommendation": recommendation,
        "fallback_allowed": request.use_fallback,
    }
