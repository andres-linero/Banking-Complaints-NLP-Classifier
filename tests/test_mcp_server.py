from banking_complaints.mcp_server import summarize_metrics


def test_summarize_metrics_returns_agent_friendly_fields() -> None:
    metrics = {
        "model_type": "spacy_lemma_tfidf_linear_svc",
        "target_column": "Product Grouped",
        "accuracy": 0.78,
        "classification_report": {
            "Bank account / checking / savings": {
                "precision": 0.8,
                "recall": 0.7,
                "f1-score": 0.75,
                "support": 10,
            },
            "macro avg": {"f1-score": 0.72},
            "weighted avg": {"f1-score": 0.77},
        },
    }

    result = summarize_metrics(metrics)

    assert result["accuracy"] == 0.78
    assert result["macro_f1"] == 0.72
    assert result["weighted_f1"] == 0.77
    assert result["classes"] == ["Bank account issue"]
