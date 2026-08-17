import argparse
import inspect
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from banking_complaints.config import (
    DATA_PATH,
    GROUPED_TARGET_COLUMN,
    MODELS_DIR,
    NORMALIZED_TARGET_COLUMN,
    REPORTS_DIR,
    TARGET_COLUMN,
    TEXT_COLUMN,
)

DEFAULT_MODEL_NAME = "distilbert-base-uncased"
DEFAULT_OUTPUT_DIR = MODELS_DIR / "bert_classifier"
DEFAULT_REPORT_PATH = REPORTS_DIR / "bert_metrics.json"
LABEL_METADATA_FILE = "label_metadata.json"


def encode_labels(labels: Iterable[Any]) -> tuple[list[int], dict[str, Any]]:
    raw_labels = [str(label) for label in labels]
    classes = sorted(set(raw_labels))
    label2id = {label: index for index, label in enumerate(classes)}
    id2label = {str(index): label for label, index in label2id.items()}

    metadata = {
        "classes": classes,
        "label2id": label2id,
        "id2label": id2label,
    }
    return [label2id[label] for label in raw_labels], metadata


def _load_torch():
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "BERT training requires optional dependencies. "
            "Install them with `python -m pip install -r requirements-bert.txt`."
        ) from exc
    return torch


class ComplaintTextDataset:
    def __init__(
        self,
        texts: Iterable[str],
        labels: Iterable[int],
        tokenizer: Any,
        max_length: int = 256,
    ) -> None:
        self.encodings = tokenizer(
            [str(text) for text in texts],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )
        self.labels = list(labels)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> dict[str, Any]:
        torch = _load_torch()
        item = {key: torch.tensor(values[index]) for key, values in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[index], dtype=torch.long)
        return item


def _training_arguments_kwargs(**kwargs: Any) -> dict[str, Any]:
    """Keep compatibility with Transformers versions that renamed eval strategy."""
    from transformers import TrainingArguments

    parameters = inspect.signature(TrainingArguments.__init__).parameters
    if "evaluation_strategy" in parameters:
        kwargs["evaluation_strategy"] = kwargs.pop("eval_strategy")
    return kwargs


def _compute_metrics(eval_pred: Any) -> dict[str, float]:
    import numpy as np
    from sklearn.metrics import accuracy_score, f1_score

    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "f1_macro": float(f1_score(labels, predictions, average="macro", zero_division=0)),
    }


def train(
    data_path: str | Path = DATA_PATH,
    model_name: str = DEFAULT_MODEL_NAME,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    report_path: str | Path = DEFAULT_REPORT_PATH,
    epochs: int = 2,
    max_length: int = 256,
    batch_size: int = 8,
    min_class_count: int = 50,
    test_size: float = 0.2,
    random_state: int = 42,
    normalize_labels: bool = True,
) -> dict[str, Any]:
    import numpy as np
    from sklearn.metrics import classification_report
    from sklearn.model_selection import train_test_split

    from banking_complaints.data import (
        group_rare_classes,
        load_complaints,
        normalize_product_labels,
    )

    _load_torch()
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
    )

    df = load_complaints(data_path)
    if normalize_labels:
        df = normalize_product_labels(df)
        target_column = NORMALIZED_TARGET_COLUMN
    else:
        target_column = TARGET_COLUMN
    df = group_rare_classes(
        df,
        min_count=min_class_count,
        target_column=target_column,
        output_column=GROUPED_TARGET_COLUMN,
    )
    labels, label_metadata = encode_labels(df[GROUPED_TARGET_COLUMN])
    df = df.assign(label=labels)

    stratify = df["label"] if df["label"].value_counts().min() >= 2 else None
    train_df, eval_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    train_dataset = ComplaintTextDataset(
        train_df[TEXT_COLUMN].tolist(),
        train_df["label"].tolist(),
        tokenizer,
        max_length=max_length,
    )
    eval_dataset = ComplaintTextDataset(
        eval_df[TEXT_COLUMN].tolist(),
        eval_df["label"].tolist(),
        tokenizer,
        max_length=max_length,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(label_metadata["classes"]),
        id2label={int(key): value for key, value in label_metadata["id2label"].items()},
        label2id=label_metadata["label2id"],
    )

    output_dir = Path(output_dir)
    report_path = Path(report_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        **_training_arguments_kwargs(
            output_dir=str(output_dir),
            eval_strategy="epoch",
            save_strategy="epoch",
            logging_dir=str(output_dir / "logs"),
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            report_to=[],
        )
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=_compute_metrics,
    )

    trainer.train()
    eval_metrics = trainer.evaluate()
    predictions = trainer.predict(eval_dataset)
    predicted_labels = np.argmax(predictions.predictions, axis=-1)

    class_names = label_metadata["classes"]
    metrics = {
        "model_name": model_name,
        "num_labels": len(class_names),
        "train_size": int(len(train_df)),
        "eval_size": int(len(eval_df)),
        "eval_metrics": {key: float(value) for key, value in eval_metrics.items()},
        "classification_report": classification_report(
            eval_df["label"].tolist(),
            predicted_labels,
            labels=list(range(len(class_names))),
            target_names=class_names,
            output_dict=True,
            zero_division=0,
        ),
    }

    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    (output_dir / LABEL_METADATA_FILE).write_text(
        json.dumps(label_metadata, indent=2),
        encoding="utf-8",
    )
    report_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune a BERT-style complaint classifier.")
    parser.add_argument("--data-path", default=str(DATA_PATH))
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--report-path", default=str(DEFAULT_REPORT_PATH))
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--min-class-count", type=int, default=50)
    parser.add_argument("--no-normalize-labels", action="store_true")
    args = parser.parse_args()

    metrics = train(
        data_path=args.data_path,
        model_name=args.model_name,
        output_dir=args.output_dir,
        report_path=args.report_path,
        epochs=args.epochs,
        max_length=args.max_length,
        batch_size=args.batch_size,
        min_class_count=args.min_class_count,
        normalize_labels=not args.no_normalize_labels,
    )
    print(f"Eval accuracy: {metrics['eval_metrics']['eval_accuracy']:.4f}")


if __name__ == "__main__":
    main()
