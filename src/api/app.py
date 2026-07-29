"""FastAPI app for simulated cloud model inference."""

from fastapi import FastAPI


app = FastAPI(title="Medical Diagnostics Cloud API")


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict")
def predict() -> dict[str, str]:
    return {
        "message": "Prediction endpoint placeholder. Connect trained model here."
    }

