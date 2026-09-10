"""Door for applications: a FastAPI service around the predictor.

Run with: uv run uvicorn complaints.api:app --reload
    POST /predict      {"text": "..."}  ->  product, confidence, needs_review, probabilities
    GET  /health       model name, threshold, classes
    GET  /inbox        demo: n synthetic complaint emails built from the test set
    GET  /inbox/next   demo: the next email from that inbox, cycling
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from itertools import count

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from complaints.config import load_serving_config
from complaints.inbox import make_inbox
from complaints.predictor import get_predictor

INBOX_SIZE = 20
INBOX_SEED = 42
_next_email = count()


class ComplaintIn(BaseModel):
    text: str = Field(..., min_length=1, description="The complaint narrative, as written")


class PredictionOut(BaseModel):
    product: str
    confidence: float
    needs_review: bool
    probabilities: dict[str, float]
    destination: str


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
        "destinations": predictor.routing["destinations"],
        "review_queue": predictor.routing["review_queue"],
    }


@app.post("/predict", response_model=PredictionOut)
def predict(complaint: ComplaintIn) -> PredictionOut:
    limit = load_serving_config()["max_text_chars"]
    if len(complaint.text) > limit:
        raise HTTPException(status_code=413, detail=f"Text longer than {limit} characters")
    if not complaint.text.strip():
        raise HTTPException(status_code=422, detail="Text is empty")
    return PredictionOut(**get_predictor().predict(complaint.text).to_dict())


@app.get("/inbox")
def inbox(n: int = Query(INBOX_SIZE, ge=1, le=200), seed: int = INBOX_SEED) -> list[dict]:
    """Demo mail source: synthetic emails wrapping real held-out complaints."""
    return [email.to_dict() for email in make_inbox(n=n, seed=seed)]


@app.get("/inbox/next")
def inbox_next() -> dict:
    """Demo mail source: hand out one email per call, cycling through the inbox."""
    emails = make_inbox(n=INBOX_SIZE, seed=INBOX_SEED)
    return emails[next(_next_email) % len(emails)].to_dict()
