"""Stage 6: the predictor every serving door calls.

Loads the saved pipeline once, applies the same text normalisation the model
was trained on, and returns the product, the confidence, and whether the
complaint should go to a person. FastAPI, Streamlit, and the MCP server are
thin wrappers around this class and hold no logic of their own.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path

import joblib

from complaints.clean import normalize_text
from complaints.config import (
    RUNTIME_CONFIG_PATH,
    SERVING_CONFIG_PATH,
    load_runtime_config,
    load_serving_config,
)


@dataclass(frozen=True)
class Prediction:
    product: str
    confidence: float
    needs_review: bool
    probabilities: dict[str, float]

    def to_dict(self) -> dict:
        return asdict(self)


class Predictor:
    def __init__(self, pipeline, review_threshold: float, model_name: str = "baseline") -> None:
        self.pipeline = pipeline
        self.review_threshold = float(review_threshold)
        self.model_name = model_name
        self.classes: list[str] = [str(c) for c in pipeline.classes_]

    @classmethod
    def load(
        cls,
        model_path: str | Path | None = None,
        serving_config: str | Path = SERVING_CONFIG_PATH,
        runtime_config: str | Path = RUNTIME_CONFIG_PATH,
    ) -> Predictor:
        """Build a predictor from serving.yaml and the saved joblib file."""
        serving = load_serving_config(serving_config)
        path = (
            Path(model_path)
            if model_path
            else (load_runtime_config(runtime_config).models_dir / f"{serving['model']}.joblib")
        )
        if not path.exists():
            raise FileNotFoundError(
                f"Model not found at {path}. Train it with: uv run python -m complaints.train"
            )
        return cls(joblib.load(path), serving["review_threshold"], serving["model"])

    def predict(self, text: str) -> Prediction:
        return self.predict_many([text])[0]

    def predict_many(self, texts: list[str]) -> list[Prediction]:
        """Normalise, score, and flag. One pipeline call for the whole batch."""
        cleaned = [normalize_text(t) for t in texts]
        proba = self.pipeline.predict_proba(cleaned)
        results = []
        for row in proba:
            best = int(row.argmax())
            confidence = float(row[best])
            results.append(
                Prediction(
                    product=self.classes[best],
                    confidence=round(confidence, 4),
                    needs_review=confidence < self.review_threshold,
                    probabilities={
                        c: round(float(p), 4) for c, p in zip(self.classes, row, strict=True)
                    },
                )
            )
        return results


@lru_cache(maxsize=1)
def get_predictor() -> Predictor:
    """Process-wide predictor, loaded on first use. The doors call this."""
    return Predictor.load()
