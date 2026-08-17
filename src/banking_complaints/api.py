from fastapi import FastAPI
from pydantic import BaseModel, Field

from banking_complaints.predict import predict_product

app = FastAPI(title="Banking Complaints NLP API")


class ComplaintRequest(BaseModel):
    text: str = Field(..., min_length=5)


class ComplaintPrediction(BaseModel):
    product: str
    sentiment: str
    confidence: float | None = None


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/predict", response_model=ComplaintPrediction)
def predict(request: ComplaintRequest) -> dict:
    return predict_product(request.text)
