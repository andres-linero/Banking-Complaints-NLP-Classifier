from datetime import datetime

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from complaints import api
from complaints.inbox import REVIEW_FOLDER, make_inbox, make_subject, route_inbox, summarize
from complaints.predictor import Predictor
from complaints.train import build_pipeline

CONFIG = {
    "features": {"ngram_range": [1, 2], "min_df": 1},
    "classifier": {"C": 1.0, "class_weight": "balanced", "max_iter": 500},
}


def _frame(n: int = 12) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Complaint ID": [f"C{i}" for i in range(2 * n)],
            "product": ["Mortgage"] * n + ["Credit card"] * n,
            "text": [f"my mortgage escrow payment and home loan servicer {i}" for i in range(n)]
            + [f"the credit card charge was disputed and the card frozen {i}" for i in range(n)],
        }
    )


@pytest.fixture
def test_parquet(tmp_path):
    path = tmp_path / "test.parquet"
    _frame().to_parquet(path, index=False)
    return path


@pytest.fixture
def small_predictor() -> Predictor:
    df = _frame()
    return Predictor(build_pipeline(CONFIG).fit(df["text"], df["product"]), review_threshold=0.75)


def test_make_subject_cuts_and_capitalises() -> None:
    assert make_subject("the bank charged me a fee twice and nobody will help me at all") == (
        "The bank charged me a fee twice and..."
    )
    assert make_subject("short one") == "Short one"
    assert make_subject("   ") == "(no subject)"


def test_make_inbox_is_deterministic_and_carries_the_true_label(test_parquet) -> None:
    now = datetime(2026, 9, 9, 12, 0, 0)
    first = make_inbox(n=6, seed=1, test_path=test_parquet, now=now)
    second = make_inbox(n=6, seed=1, test_path=test_parquet, now=now)
    other = make_inbox(n=6, seed=2, test_path=test_parquet, now=now)

    assert [e.id for e in first] == [e.id for e in second]
    assert [e.id for e in first] != [e.id for e in other]
    assert len(first) == 6
    assert all(e.true_product in {"Mortgage", "Credit card"} for e in first)
    assert all("@example." in e.sender for e in first)
    assert [e.received_at for e in first] == sorted((e.received_at for e in first), reverse=True)


def test_make_inbox_none_means_every_row(test_parquet) -> None:
    emails = make_inbox(n=None, seed=1, test_path=test_parquet)
    assert len(emails) == 24
    assert len({e.id for e in emails}) == 24


def test_route_inbox_sets_folder_and_correct_flag(test_parquet, small_predictor) -> None:
    routed = route_inbox(make_inbox(n=8, seed=3, test_path=test_parquet), small_predictor)

    assert len(routed) == 8
    for r in routed:
        expected = REVIEW_FOLDER if r.prediction.needs_review else r.prediction.product
        assert r.folder == expected
        assert r.correct == (r.prediction.product == r.email.true_product)

    stats = summarize(routed)
    assert stats["total"] == 8
    assert stats["automatic"] + stats["review"] == 8
    assert sum(stats["folders"].values()) == 8


def test_api_inbox_endpoints(test_parquet, monkeypatch) -> None:
    monkeypatch.setattr(api, "make_inbox", lambda n, seed: make_inbox(n, seed, test_parquet))
    client = TestClient(api.app)

    emails = client.get("/inbox", params={"n": 5, "seed": 1}).json()
    assert len(emails) == 5
    assert set(emails[0]) == {"id", "sender", "subject", "received_at", "body", "true_product"}

    first = client.get("/inbox/next").json()
    second = client.get("/inbox/next").json()
    assert first["id"] != second["id"]
    assert client.get("/inbox", params={"n": 0}).status_code == 422


def test_frontend_escapes_html_in_email_fields() -> None:
    import importlib.util
    from pathlib import Path

    from complaints.live import handle_text

    spec = importlib.util.spec_from_file_location("frontend_app", Path("frontend/app.py"))
    app = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(app)

    predictor = Predictor(
        build_pipeline(CONFIG).fit(_frame()["text"], _frame()["product"]), review_threshold=0.5
    )
    record = handle_text("<img src=x onerror=alert(1)> credit card charge", predictor)
    record = record.__class__(**{**record.__dict__, "sender": "<script>alert(1)</script>"})

    rendered = app._record_header_html(record)
    assert "<script>" not in rendered and "&lt;script&gt;" in rendered
    assert "<img" not in app._department_card_html("<img>", "<b>x</b>")
