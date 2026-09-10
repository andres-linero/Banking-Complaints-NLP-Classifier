import joblib
import pandas as pd
import pytest
import yaml
from fastapi.testclient import TestClient

from complaints import api, mcp_server
from complaints import predictor as predictor_module
from complaints.config import load_serving_config
from complaints.predictor import Predictor
from complaints.train import build_pipeline

CONFIG = {
    "features": {"ngram_range": [1, 2], "min_df": 1},
    "classifier": {"C": 1.0, "class_weight": "balanced", "max_iter": 500},
}


def _fitted_pipeline():
    n = 12
    df = pd.DataFrame(
        {
            "product": ["Mortgage"] * n + ["Credit card"] * n,
            "text": [f"my mortgage escrow payment and home loan servicer {i}" for i in range(n)]
            + [f"the credit card charge was disputed and the card frozen {i}" for i in range(n)],
        }
    )
    return build_pipeline(CONFIG).fit(df["text"], df["product"])


@pytest.fixture
def small_predictor() -> Predictor:
    return Predictor(_fitted_pipeline(), review_threshold=0.75, model_name="tiny")


@pytest.fixture
def served(small_predictor, monkeypatch):
    """Point every door at the small predictor instead of the real model file."""
    monkeypatch.setattr(predictor_module, "get_predictor", lambda: small_predictor)
    monkeypatch.setattr(api, "get_predictor", lambda: small_predictor)
    monkeypatch.setattr(mcp_server, "get_predictor", lambda: small_predictor)
    return small_predictor


def test_predict_returns_product_confidence_and_review_flag(small_predictor) -> None:
    result = small_predictor.predict("mortgage escrow home loan servicer")

    assert result.product == "Mortgage"
    assert 0 < result.confidence <= 1
    assert result.needs_review == (result.confidence < 0.75)
    assert set(result.probabilities) == {"Credit card", "Mortgage"}
    assert abs(sum(result.probabilities.values()) - 1) < 1e-3


def test_predict_normalises_text_like_training(small_predictor) -> None:
    plain = small_predictor.predict("mortgage escrow on the account")
    masked = small_predictor.predict("mortgage escrow on XX/XX/XXXX the account XXXX")

    assert plain.product == masked.product


def test_predictor_load_reads_serving_yaml(tmp_path) -> None:
    model_path = tmp_path / "tiny.joblib"
    joblib.dump(_fitted_pipeline(), model_path)
    serving = tmp_path / "serving.yaml"
    serving.write_text(yaml.safe_dump({"model": "tiny", "review_threshold": 0.9}))

    loaded = Predictor.load(model_path=model_path, serving_config=serving)

    assert loaded.review_threshold == 0.9
    assert loaded.model_name == "tiny"


def test_serving_config_rejects_bad_threshold(tmp_path) -> None:
    path = tmp_path / "serving.yaml"
    path.write_text("review_threshold: 1.5\n")

    with pytest.raises(ValueError, match="between 0 and 1"):
        load_serving_config(path)


def test_serving_config_requires_threshold(tmp_path) -> None:
    """A missing or misspelled key must stop startup, not silently route everything."""
    path = tmp_path / "serving.yaml"
    path.write_text("model: baseline\nreview_treshold: 0.75\n")

    with pytest.raises(ValueError, match="must set review_threshold"):
        load_serving_config(path)


def test_routing_config_requires_every_class(tmp_path) -> None:
    from complaints.config import load_routing_config

    path = tmp_path / "routing.yaml"
    path.write_text("destinations:\n  Mortgage: m@x\nreview_queue: r@x\n")

    assert load_routing_config(path, classes=["Mortgage"])["destinations"] == {"Mortgage": "m@x"}
    with pytest.raises(ValueError, match="no destination for: \\['Credit card'\\]"):
        load_routing_config(path, classes=["Mortgage", "Credit card"])


def test_real_routing_yaml_covers_the_seven_classes() -> None:
    from complaints.config import load_routing_config

    routing = load_routing_config()
    assert set(routing["destinations"]) == {
        "Bank account",
        "Credit card",
        "Credit reporting",
        "Debt collection",
        "Loan",
        "Mortgage",
        "Student loan",
    }
    assert "@" in routing["review_queue"]


def test_prediction_carries_a_destination(small_predictor) -> None:
    result = small_predictor.predict("mortgage escrow home loan servicer")
    expected = (
        small_predictor.routing["review_queue"]
        if result.needs_review
        else small_predictor.routing["destinations"][result.product]
    )
    assert result.destination == expected


def test_real_serving_yaml_threshold_is_075() -> None:
    assert load_serving_config()["review_threshold"] == 0.75


def test_api_predict_and_health(served) -> None:
    client = TestClient(api.app)

    health = client.get("/health").json()
    assert health["status"] == "ok"
    assert health["review_threshold"] == 0.75
    assert health["classes"] == ["Credit card", "Mortgage"]

    response = client.post("/predict", json={"text": "credit card charge disputed"})
    assert response.status_code == 200
    body = response.json()
    assert body["product"] == "Credit card"
    assert set(body) == {"product", "confidence", "needs_review", "probabilities", "destination"}


def test_api_rejects_empty_and_oversized_text(served) -> None:
    client = TestClient(api.app)

    assert client.post("/predict", json={"text": "   "}).status_code == 422
    assert client.post("/predict", json={"text": "x" * 20001}).status_code == 413


def test_mcp_tools_use_the_predictor(served) -> None:
    result = mcp_server.classify_complaint("mortgage escrow home loan")
    assert result["product"] == "Mortgage"

    products = mcp_server.list_products()
    assert products["products"] == ["Credit card", "Mortgage"]
    assert set(products["destinations"]) == {"Credit card", "Mortgage"}
    assert products["review_threshold"] == 0.75

    with pytest.raises(ValueError, match="empty"):
        mcp_server.classify_complaint("  ")


def test_mcp_server_registers_both_tools() -> None:
    pytest.importorskip("mcp")
    server = mcp_server.build_server()
    names = {tool.name for tool in server._tool_manager.list_tools()}
    assert names == {"classify_complaint", "list_products"}
