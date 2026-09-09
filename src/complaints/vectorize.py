"""Text vectorization: turn complaint text into TF-IDF vectors inside the model.

The vectorizer is fitted on the training texts only and then saved as the
first step of the model pipeline, so at prediction time raw text goes in and
the same vocabulary and weights are applied. Nothing here reads test data.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from sklearn.feature_extraction.text import TfidfVectorizer

from complaints.config import PROJECT_ROOT

BASELINE_CONFIG_PATH = PROJECT_ROOT / "configs" / "baseline.yaml"


def load_model_config(path: str | Path = BASELINE_CONFIG_PATH) -> dict:
    """Read a model yaml with `features` and `classifier` sections."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    for section in ("features", "classifier"):
        if section not in raw or not isinstance(raw[section], dict):
            raise ValueError(f"Model config must have a `{section}` mapping")
    return raw


def build_vectorizer(features: dict) -> TfidfVectorizer:
    """Configure TF-IDF from the `features` section of the model yaml."""
    params = dict(features)
    if "ngram_range" in params:
        params["ngram_range"] = tuple(params["ngram_range"])
    return TfidfVectorizer(**params)


def describe_vectors(vectorizer: TfidfVectorizer, matrix) -> dict:
    """Summarise a fitted vectorizer and the matrix it produced."""
    vocabulary = vectorizer.get_feature_names_out()
    return {
        "rows": int(matrix.shape[0]),
        "vocabulary_size": int(matrix.shape[1]),
        "bigrams": int(sum(" " in term for term in vocabulary)),
        "nonzero_per_row": float(matrix.nnz / matrix.shape[0]),
        "density": float(matrix.nnz / (matrix.shape[0] * matrix.shape[1])),
    }
