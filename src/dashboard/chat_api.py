"""FastAPI server for the interactive results dashboard and chatbot."""

from __future__ import annotations

from typing import Literal

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from src.dashboard.chat import answer_chat
from src.dashboard.generate_dashboard import render_dashboard


app = FastAPI(title="Adversarial Medical Diagnostics Dashboard")


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=4000)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    level: Literal["beginner", "technical"] = "beginner"
    history: list[ChatMessage] = Field(default_factory=list, max_length=12)


@app.get("/", response_class=HTMLResponse)
def dashboard() -> str:
    """Serve the generated interactive dashboard."""
    return render_dashboard()


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/chat")
def chat(request: ChatRequest) -> dict[str, object]:
    history = [{"role": item.role, "content": item.content} for item in request.history]
    answer, fallback = answer_chat(request.message, request.level, history)
    return {
        "answer": answer,
        "fallback": fallback,
        "disclaimer": "ML security education only; not medical advice.",
    }
