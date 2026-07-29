"""FastAPI app for simulated cloud model inference."""

from io import BytesIO
from pathlib import Path

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

from src.defenses.randomization import apply_defensive_randomization
from src.models.predict import DEFAULT_CHECKPOINT, load_model, predict_image


app = FastAPI(title="Medical Diagnostics Cloud API")
_model = None
_class_names: list[str] | None = None
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
