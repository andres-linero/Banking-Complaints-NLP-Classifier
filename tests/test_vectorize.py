import pytest

from complaints.vectorize import (
    BASELINE_CONFIG_PATH,
    build_vectorizer,
    describe_vectors,
    load_model_config,
)

TEXTS = [
    "My mortgage payment was applied late and I was charged a fee",
    "The bank charged me an overdraft fee twice on redacted",
    "Somebody opened a credit card in my name without permission",
    "Mortgage escrow was miscalculated and my payment went up",
]


def test_baseline_config_has_both_sections() -> None:
    config = load_model_config(BASELINE_CONFIG_PATH)

    assert set(config) >= {"features", "classifier"}
    assert config["features"]["ngram_range"] == [1, 2]


def test_load_model_config_rejects_missing_section(tmp_path) -> None:
    path = tmp_path / "model.yaml"
    path.write_text("features:\n  min_df: 1\n")

    with pytest.raises(ValueError, match="classifier"):
        load_model_config(path)


def test_vectorizer_builds_unigrams_and_bigrams() -> None:
    vectorizer = build_vectorizer({"ngram_range": [1, 2], "min_df": 1})
    matrix = vectorizer.fit_transform(TEXTS)
    vocabulary = set(vectorizer.get_feature_names_out())

    assert matrix.shape[0] == len(TEXTS)
    assert "mortgage" in vocabulary
    assert "mortgage payment" in vocabulary
    assert "redacted" in vocabulary


def test_vectorizer_lowercases_and_reuses_training_vocabulary() -> None:
    vectorizer = build_vectorizer({"ngram_range": [1, 1], "min_df": 1, "lowercase": True})
    vectorizer.fit(TEXTS)

    unseen = vectorizer.transform(["MORTGAGE FEE and a brand new word zzzz"])

    assert unseen.shape[1] == len(vectorizer.get_feature_names_out())
    assert unseen[0, vectorizer.vocabulary_["mortgage"]] > 0
    assert "zzzz" not in vectorizer.vocabulary_


def test_min_df_drops_rare_terms() -> None:
    vectorizer = build_vectorizer({"ngram_range": [1, 1], "min_df": 2})
    vectorizer.fit(TEXTS)
    vocabulary = set(vectorizer.get_feature_names_out())

    assert "mortgage" in vocabulary  # appears in two texts
    assert "escrow" not in vocabulary  # appears in one


def test_describe_vectors_reports_shape_and_bigrams() -> None:
    vectorizer = build_vectorizer({"ngram_range": [1, 2], "min_df": 1})
    matrix = vectorizer.fit_transform(TEXTS)

    summary = describe_vectors(vectorizer, matrix)

    assert summary["rows"] == 4
    assert summary["vocabulary_size"] == matrix.shape[1]
    assert 0 < summary["bigrams"] < summary["vocabulary_size"]
    assert 0 < summary["density"] < 1
