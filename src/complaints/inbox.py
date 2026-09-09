"""Demo source: a synthetic inbox of banking complaints, built from the frozen test set.

The model only understands banking complaints, so the demo feeds it banking
complaints. Each email is a real held-out complaint from test.parquet wrapped in
a synthetic sender, subject, and timestamp. The true label rides along so the
demo can say whether the routing was right. Nothing here touches the model.
"""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from complaints.clean import CLASS_COLUMN, CLEAN_TEXT_COLUMN
from complaints.config import RUNTIME_CONFIG_PATH, load_runtime_config
from complaints.ingest import ID_COLUMN
from complaints.predictor import Prediction, Predictor

REVIEW_FOLDER = "Needs a person"
SUBJECT_WORDS = 8
FIRST_NAMES = ["alex", "jordan", "maria", "sam", "priya", "diego", "chen", "fatima", "luis", "nora"]
DOMAINS = ["example.com", "example.net", "example.org"]


@dataclass(frozen=True)
class Email:
    id: str  # the Complaint ID, so the row can be traced back
    sender: str
    subject: str
    received_at: str  # ISO timestamp
    body: str
    true_product: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class RoutedEmail:
    email: Email
    prediction: Prediction
    folder: str  # the product, or REVIEW_FOLDER when confidence is under the threshold
    correct: bool  # predicted product == true label

    def to_dict(self) -> dict:
        return {
            "email": self.email.to_dict(),
            "prediction": self.prediction.to_dict(),
            "folder": self.folder,
            "correct": self.correct,
        }


def make_subject(body: str, words: int = SUBJECT_WORDS) -> str:
    """First few words of the complaint, capitalised, with a trailing ellipsis if cut."""
    parts = body.split()
    subject = " ".join(parts[:words]).strip(" .,;:")
    if not subject:
        return "(no subject)"
    subject = subject[0].upper() + subject[1:]
    return subject + ("..." if len(parts) > words else "")


def make_inbox(
    n: int = 20,
    seed: int = 42,
    test_path: str | Path | None = None,
    now: datetime | None = None,
) -> list[Email]:
    """Sample n test complaints and dress each one as an email. Deterministic per seed."""
    path = Path(test_path) if test_path else load_runtime_config(RUNTIME_CONFIG_PATH).test_data_path
    rows = pd.read_parquet(path).sample(n=n, random_state=seed).reset_index(drop=True)
    rng = random.Random(seed)
    now = now or datetime.now().replace(microsecond=0)

    emails = []
    for _, row in rows.iterrows():
        name = rng.choice(FIRST_NAMES)
        received = now - timedelta(minutes=rng.randint(5, 3 * 24 * 60))
        emails.append(
            Email(
                id=str(row[ID_COLUMN]),
                sender=f"{name}.{rng.randint(10, 99)}@{rng.choice(DOMAINS)}",
                subject=make_subject(row[CLEAN_TEXT_COLUMN]),
                received_at=received.isoformat(),
                body=str(row[CLEAN_TEXT_COLUMN]),
                true_product=str(row[CLASS_COLUMN]),
            )
        )
    return sorted(emails, key=lambda e: e.received_at, reverse=True)


def route_inbox(emails: list[Email], predictor: Predictor) -> list[RoutedEmail]:
    """Run every email through the predictor once, in a single batch."""
    predictions = predictor.predict_many([e.body for e in emails])
    return [
        RoutedEmail(
            email=e,
            prediction=p,
            folder=REVIEW_FOLDER if p.needs_review else p.product,
            correct=p.product == e.true_product,
        )
        for e, p in zip(emails, predictions, strict=True)
    ]


def summarize(routed: list[RoutedEmail]) -> dict:
    """Counts a viewer wants at a glance: per folder, automatic share, accuracy of what routed."""
    folders: dict[str, int] = {}
    for r in routed:
        folders[r.folder] = folders.get(r.folder, 0) + 1
    automatic = [r for r in routed if r.folder != REVIEW_FOLDER]
    return {
        "total": len(routed),
        "folders": dict(sorted(folders.items(), key=lambda kv: -kv[1])),
        "automatic": len(automatic),
        "automatic_correct": sum(r.correct for r in automatic),
        "review": len(routed) - len(automatic),
    }
