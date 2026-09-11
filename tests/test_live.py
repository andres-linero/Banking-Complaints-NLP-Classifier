from datetime import datetime

import pandas as pd
import pytest

from complaints.inbox import Email
from complaints.live import handle_email, handle_text, scoreboard
from complaints.predictor import Predictor
from complaints.train import build_pipeline

CONFIG = {
    "features": {"ngram_range": [1, 2], "min_df": 1},
    "classifier": {"C": 1.0, "class_weight": "balanced", "max_iter": 500},
}
NOW = datetime(2026, 9, 10, 12, 0, 0)


@pytest.fixture
def small_predictor() -> Predictor:
    n = 12
    df = pd.DataFrame(
        {
            "product": ["Mortgage"] * n + ["Credit card"] * n,
            "text": [f"my mortgage escrow payment and home loan servicer {i}" for i in range(n)]
            + [f"the credit card charge was disputed and the card frozen {i}" for i in range(n)],
        }
    )
    return Predictor(build_pipeline(CONFIG).fit(df["text"], df["product"]), review_threshold=0.5)


def _email(body: str, label: str) -> Email:
    return Email("C1", "alex.10@example.com", "Subject", NOW.isoformat(), body, label)


def test_handle_email_checks_the_true_label(small_predictor) -> None:
    record = handle_email(
        _email("mortgage escrow home loan servicer", "Mortgage"), small_predictor, NOW
    )

    assert record.source == "inbox"
    assert record.prediction.product == "Mortgage"
    assert record.correct is True
    assert record.handled_at == NOW.isoformat()
    assert record.steps()[0] == "Received from alex.10@example.com"
    assert record.steps()[-1] == "Matches the true label: Mortgage"


def test_handle_email_marks_wrong_routing(small_predictor) -> None:
    record = handle_email(
        _email("mortgage escrow home loan servicer", "Credit card"), small_predictor, NOW
    )

    assert record.correct is False
    assert record.steps()[-1] == "Wrong, the true label is Credit card"


def test_record_keeps_the_call_and_its_timing(small_predictor) -> None:
    record = handle_text("credit card charge disputed and frozen", small_predictor, NOW)

    assert record.elapsed_ms >= 0
    snippet = record.call_snippet()
    assert snippet.startswith('predictor.predict("credit card charge disputed and frozen")')
    assert '"product": "Credit card"' in snippet
    assert '"destination":' in snippet


def test_handle_text_has_no_label(small_predictor) -> None:
    record = handle_text("credit card charge disputed and frozen", small_predictor, NOW)

    assert record.source == "typed" and record.sender == "you"
    assert record.true_product is None and record.correct is None
    assert record.steps()[-1] == "No label to check"


def test_scoreboard_counts_only_labelled_automatic_records(small_predictor) -> None:
    records = [
        handle_email(
            _email("mortgage escrow home loan servicer", "Mortgage"), small_predictor, NOW
        ),
        handle_email(
            _email("mortgage escrow home loan servicer", "Credit card"), small_predictor, NOW
        ),
        handle_text("credit card charge disputed and frozen", small_predictor, NOW),
    ]
    board = scoreboard(records)

    assert board["arrived"] == 3
    assert board["automatic"] + board["review"] == 3
    assert board["checked"] <= board["automatic"]
    assert board["correct"] <= board["checked"]
