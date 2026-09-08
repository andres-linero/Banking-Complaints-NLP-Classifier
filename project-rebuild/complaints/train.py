"""Stage 4: train the baseline classifier and save it.

Run with: uv run python -m complaints.train

Reads train.parquet only. Builds one scikit-learn Pipeline (TF-IDF vectorizer
followed by logistic regression) from configs/baseline.yaml, checks it with
5-fold cross-validation on the training rows, fits it on all of them, and
saves the fitted pipeline with joblib. Every run is logged to MLflow with its
config and cross-validation scores. The test split is never read here.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline

from complaints.clean import CLASS_COLUMN, CLEAN_TEXT_COLUMN
from complaints.config import RUNTIME_CONFIG_PATH, load_runtime_config
from complaints.vectorize import BASELINE_CONFIG_PATH, build_vectorizer, load_model_config

MODEL_NAME = "baseline"
CV_FOLDS = 5
CV_SEED = 42


def build_pipeline(config: dict) -> Pipeline:
    """Vectorizer then classifier, as one object that takes raw text."""
    return Pipeline(
        [
            ("vectorize", build_vectorizer(config["features"])),
            ("classify", LogisticRegression(**config["classifier"])),
        ]
    )


def cross_validate_pipeline(
    pipeline: Pipeline, texts: pd.Series, labels: pd.Series, folds: int = CV_FOLDS
) -> dict:
    """Score an unfitted pipeline on the training rows only, fold by fold."""
    scores = cross_validate(
        pipeline,
        texts,
        labels,
        cv=StratifiedKFold(folds, shuffle=True, random_state=CV_SEED),
        scoring=["accuracy", "f1_macro"],
    )
    return {
        "folds": folds,
        "accuracy_mean": float(scores["test_accuracy"].mean()),
        "accuracy_std": float(scores["test_accuracy"].std()),
        "f1_macro_mean": float(scores["test_f1_macro"].mean()),
        "f1_macro_std": float(scores["test_f1_macro"].std()),
    }


def train(train_df: pd.DataFrame, config: dict, folds: int = CV_FOLDS) -> tuple[Pipeline, dict]:
    """Cross-validate, then fit on every training row. Returns the fitted pipeline and a report."""
    texts, labels = train_df[CLEAN_TEXT_COLUMN], train_df[CLASS_COLUMN]

    cv = cross_validate_pipeline(build_pipeline(config), texts, labels, folds=folds)
    pipeline = build_pipeline(config).fit(texts, labels)

    report = {
        "model": MODEL_NAME,
        "train_rows": int(len(train_df)),
        "classes": [str(c) for c in pipeline.classes_],
        "vocabulary_size": int(len(pipeline.named_steps["vectorize"].vocabulary_)),
        "cross_validation": cv,
        "config": config,
    }
    return pipeline, report


def log_to_mlflow(report: dict, model_path: Path, tracking_dir: Path) -> str:
    """Record one training run: config as params, CV scores as metrics, the model as an artifact.

    Tracking lives in a local SQLite file under `tracking_dir`, artifacts next to it.
    No server is needed; `uv run mlflow ui --backend-store-uri sqlite:///<dir>/mlflow.db`
    opens the comparison table in a browser.
    """
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    import mlflow

    tracking_dir = tracking_dir.resolve()
    tracking_dir.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{tracking_dir / 'mlflow.db'}")
    experiment = "complaints-classifier"
    if mlflow.get_experiment_by_name(experiment) is None:
        mlflow.create_experiment(
            experiment, artifact_location=(tracking_dir / "artifacts").as_uri()
        )
    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name=report["model"]) as run:
        for section in ("features", "classifier"):
            for key, value in report["config"][section].items():
                mlflow.log_param(f"{section}.{key}", value)
        mlflow.log_param("train_rows", report["train_rows"])
        mlflow.log_param("vocabulary_size", report["vocabulary_size"])
        for key, value in report["cross_validation"].items():
            if key != "folds":
                mlflow.log_metric(f"cv_{key}", value)
        mlflow.log_artifact(str(model_path))
        return run.info.run_id


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the baseline complaint classifier.")
    parser.add_argument("--config", default=str(RUNTIME_CONFIG_PATH))
    parser.add_argument("--model-config", default=str(BASELINE_CONFIG_PATH))
    parser.add_argument("--train-path")
    parser.add_argument("--model-path")
    parser.add_argument("--reports-dir")
    parser.add_argument("--mlflow-dir")
    parser.add_argument("--no-mlflow", action="store_true", help="skip experiment tracking")
    args = parser.parse_args()

    runtime = load_runtime_config(args.config)
    config = load_model_config(args.model_config)
    train_df = pd.read_parquet(args.train_path or runtime.train_data_path)

    pipeline, report = train(train_df, config)

    model_path = Path(args.model_path or runtime.models_dir / f"{MODEL_NAME}.joblib")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_path)

    if not args.no_mlflow:
        report["mlflow_run_id"] = log_to_mlflow(
            report, model_path, Path(args.mlflow_dir or runtime.mlflow_dir)
        )

    report_path = Path(args.reports_dir or runtime.reports_dir) / "train" / f"{MODEL_NAME}.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    cv = report["cross_validation"]
    print(f"Saved {model_path}")
    print(f"train rows {report['train_rows']}  vocabulary {report['vocabulary_size']}")
    print(
        f"{cv['folds']}-fold CV on train: "
        f"accuracy {cv['accuracy_mean']:.3f} ± {cv['accuracy_std']:.3f}   "
        f"macro-F1 {cv['f1_macro_mean']:.3f} ± {cv['f1_macro_std']:.3f}"
    )
    if "mlflow_run_id" in report:
        print(f"mlflow run {report['mlflow_run_id']}")


if __name__ == "__main__":
    main()
