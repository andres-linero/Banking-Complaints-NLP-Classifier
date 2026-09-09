"""Door for applications: a FastAPI service around the predictor.

Run with: uv run uvicorn complaints.api:app --reload
    POST /predict   {"text": "..."}  ->  product, confidence, needs_review, probabilities
    GET  /health    model name, threshold, classes
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from complaints.config import load_serving_config
from complaints.predictor import get_predictor


class ComplaintIn(BaseModel):
    text: str = Field(..., min_length=1, description="The complaint narrative, as written")


class PredictionOut(BaseModel):
    product: str
    confidence: float
    needs_review: bool
    probabilities: dict[str, float]


@asynccontextmanager
async def lifespan(app: FastAPI):
    get_predictor()  # load the model at startup, not on the first request
    yield


app = FastAPI(title="Banking complaints classifier", version="0.2.0", lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    predictor = get_predictor()
    return {
        "status": "ok",
        "model": predictor.model_name,
        "review_threshold": predictor.review_threshold,
        "classes": predictor.classes,
    }


@app.post("/predict", response_model=PredictionOut)
def predict(complaint: ComplaintIn) -> PredictionOut:
    limit = load_serving_config()["max_text_chars"]
    if len(complaint.text) > limit:
        raise HTTPException(status_code=413, detail=f"Text longer than {limit} characters")
    if not complaint.text.strip():
        raise HTTPException(status_code=422, detail="Text is empty")
    return PredictionOut(**get_predictor().predict(complaint.text).to_dict())
