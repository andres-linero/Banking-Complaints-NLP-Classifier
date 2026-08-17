import json
import time
from pathlib import Path

from banking_complaints.config import (
    DATA_PATH,
    GROUPED_TARGET_COLUMN,
    NORMALIZED_TARGET_COLUMN,
    REPORTS_DIR,
    TEXT_COLUMN,
)
from banking_complaints.data import group_rare_classes, load_complaints, normalize_product_labels
from banking_complaints.labels import friendly_label
from banking_complaints.predict import predict_product

DEFAULT_METRICS_PATH = REPORTS_DIR / "sklearn_metrics.json"


def load_model_metrics(path: str | Path = DEFAULT_METRICS_PATH) -> dict:
    path = Path(path)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def summarize_metrics(metrics: dict) -> dict:
    report = metrics.get("classification_report", {})
    macro_avg = report.get("macro avg", {})
    weighted_avg = report.get("weighted avg", {})

    average_labels = {"macro avg", "weighted avg"}
    return {
        "model_type": metrics.get("model_type", "unknown"),
        "target_column": metrics.get("target_column", GROUPED_TARGET_COLUMN),
        "accuracy": metrics.get("accuracy"),
        "macro_f1": macro_avg.get("f1-score"),
        "weighted_f1": weighted_avg.get("f1-score"),
        "classes": sorted(
            friendly_label(label)
            for label, values in report.items()
            if isinstance(values, dict) and "f1-score" in values and label not in average_labels
        ),
    }


def classify_complaint_tool(text: str) -> dict:
    """Classify a complaint narrative using the saved local model."""
    if not text.strip():
        raise ValueError("Complaint text cannot be empty.")

    started_at = time.perf_counter()
    prediction = predict_product(text)
    latency_ms = (time.perf_counter() - started_at) * 1000
    model_label = prediction["product"]

    return {
        "category": friendly_label(model_label),
        "model_label": model_label,
        "sentiment": prediction["sentiment"],
        "confidence": prediction.get("confidence"),
        "latency_ms": round(latency_ms, 2),
    }


def get_model_metrics_tool() -> dict:
    """Return saved model quality metrics."""
    metrics = load_model_metrics()
    if not metrics:
        return {
            "status": "missing",
            "message": "Train the model first with `python -m banking_complaints.train_sklearn`.",
        }
    return {"status": "ok", **summarize_metrics(metrics)}


def get_supported_categories_tool() -> list[dict]:
    """Return product categories known from the saved metrics or training dataset."""
    metrics = load_model_metrics()
    report = metrics.get("classification_report", {})
    average_labels = {"macro avg", "weighted avg"}
    labels = [
        label
        for label, values in report.items()
        if isinstance(values, dict) and "f1-score" in values and label not in average_labels
    ]

    if not labels and DATA_PATH.exists():
        df = load_complaints(DATA_PATH)
        df = normalize_product_labels(df)
        df = group_rare_classes(
            df,
            min_count=50,
            target_column=NORMALIZED_TARGET_COLUMN,
            output_column=GROUPED_TARGET_COLUMN,
        )
        labels = sorted(df[GROUPED_TARGET_COLUMN].dropna().unique().tolist())

    return [
        {"model_label": label, "display_label": friendly_label(label)} for label in sorted(labels)
    ]


def get_category_examples_tool(category: str, limit: int = 3) -> dict:
    """Return real training examples for a category or friendly category name."""
    if not DATA_PATH.exists():
        return {"status": "missing_data", "examples": []}

    limit = max(1, min(limit, 10))
    wanted = category.strip().lower()
    df = load_complaints(DATA_PATH)
    df = normalize_product_labels(df)
    df = group_rare_classes(
        df,
        min_count=50,
        target_column=NORMALIZED_TARGET_COLUMN,
        output_column=GROUPED_TARGET_COLUMN,
    )

    labels = df[GROUPED_TARGET_COLUMN].dropna().unique().tolist()
    matched_label: str | None = None
    for label in labels:
        if wanted in {label.lower(), friendly_label(label).lower()}:
            matched_label = label
            break

    if matched_label is None:
        return {
            "status": "unknown_category",
            "requested_category": category,
            "available_categories": [friendly_label(label) for label in sorted(labels)],
            "examples": [],
        }

    examples = (
        df[df[GROUPED_TARGET_COLUMN] == matched_label][[TEXT_COLUMN]]
        .head(limit)
        .assign(display_label=friendly_label(matched_label), model_label=matched_label)
    )
    return {
        "status": "ok",
        "category": friendly_label(matched_label),
        "model_label": matched_label,
        "examples": examples.to_dict(orient="records"),
    }


def create_mcp_server():
    try:
        from mcp.server.fastmcp import FastMCP
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "MCP support requires Python 3.10+ and the official MCP SDK. "
            "Install it with `python -m pip install -r requirements-mcp.txt`."
        ) from exc

    mcp = FastMCP(
        "Banking Complaints Classifier",
        stateless_http=True,
        json_response=True,
        instructions=(
            "Use these tools to classify banking complaint narratives, inspect model metrics, "
            "and retrieve examples from the local training dataset."
        ),
    )

    mcp.tool()(classify_complaint_tool)
    mcp.tool()(get_model_metrics_tool)
    mcp.tool()(get_supported_categories_tool)
    mcp.tool()(get_category_examples_tool)
    return mcp


def main() -> None:
    mcp = create_mcp_server()
    mcp.run(transport="streamable-http")


if __name__ == "__main__":
    main()
