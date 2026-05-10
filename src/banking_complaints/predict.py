from pathlib import Path
from typing import Union

import joblib

from banking_complaints.config import MODELS_DIR
from banking_complaints.preprocessing import sentiment_label

DEFAULT_MODEL_PATH = MODELS_DIR / "complaint_classifier.joblib"


def predict_product(text: str, model_path: Union[str, Path] = DEFAULT_MODEL_PATH) -> dict:
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}. "
            "Train it with `python -m banking_complaints.train_sklearn`."
        )

    model = joblib.load(model_path)
    product = model.predict([text])[0]
    response = {
        "product": product,
        "sentiment": sentiment_label(text),
    }

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba([text])[0]
        best_probability = max(probabilities)
        response["confidence"] = float(best_probability)

    return response
