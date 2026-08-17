import argparse
import json
from pathlib import Path

import joblib
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from banking_complaints.config import (
    DATA_PATH,
    GROUPED_TARGET_COLUMN,
    MODELS_DIR,
    NORMALIZED_TARGET_COLUMN,
    REPORTS_DIR,
    TEXT_COLUMN,
)
from banking_complaints.data import group_rare_classes, load_complaints, normalize_product_labels
from banking_complaints.preprocessing import SpacyLemmaTransformer
from banking_complaints.visualize import save_training_figures


def build_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("spacy_lemma", SpacyLemmaTransformer()),
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=50_000,
                    ngram_range=(1, 2),
                    min_df=2,
                    sublinear_tf=True,
                    stop_words="english",
                ),
            ),
            (
                "model",
                CalibratedClassifierCV(
                    LinearSVC(class_weight="balanced", C=1.0, random_state=42),
                    method="sigmoid",
                    cv=5,
                ),
            ),
        ]
    )


def train(
    data_path: str | Path = DATA_PATH,
    model_path: str | Path = MODELS_DIR / "complaint_classifier.joblib",
    report_path: str | Path = REPORTS_DIR / "sklearn_metrics.json",
    min_class_count: int = 50,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict:
    df = load_complaints(data_path)
    df = normalize_product_labels(df)
    df = group_rare_classes(
        df,
        min_count=min_class_count,
        target_column=NORMALIZED_TARGET_COLUMN,
        output_column=GROUPED_TARGET_COLUMN,
    )

    x_train, x_test, y_train, y_test = train_test_split(
        df[TEXT_COLUMN],
        df[GROUPED_TARGET_COLUMN],
        test_size=test_size,
        random_state=random_state,
        stratify=df[GROUPED_TARGET_COLUMN],
    )

    pipeline = build_pipeline()
    pipeline.fit(x_train, y_train)
    predictions = pipeline.predict(x_test)

    metrics = {
        "model_type": "spacy_lemma_tfidf_linear_svc",
        "target_column": GROUPED_TARGET_COLUMN,
        "normalized_target_column": NORMALIZED_TARGET_COLUMN,
        "accuracy": accuracy_score(y_test, predictions),
        "classification_report": classification_report(
            y_test,
            predictions,
            output_dict=True,
            zero_division=0,
        ),
    }

    model_path = Path(model_path)
    report_path = Path(report_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    metrics["figures"] = save_training_figures(
        labels=df[GROUPED_TARGET_COLUMN],
        y_test=y_test,
        predictions=predictions,
        classification_report=metrics["classification_report"],
        figures_dir=report_path.parent / "figures",
    )

    joblib.dump(pipeline, model_path)
    report_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a TF-IDF complaint classifier.")
    parser.add_argument("--data-path", default=str(DATA_PATH))
    parser.add_argument("--model-path", default=str(MODELS_DIR / "complaint_classifier.joblib"))
    parser.add_argument("--report-path", default=str(REPORTS_DIR / "sklearn_metrics.json"))
    parser.add_argument("--min-class-count", type=int, default=50)
    args = parser.parse_args()

    metrics = train(
        data_path=args.data_path,
        model_path=args.model_path,
        report_path=args.report_path,
        min_class_count=args.min_class_count,
    )
    print(f"Accuracy: {metrics['accuracy']:.4f}")


if __name__ == "__main__":
    main()
