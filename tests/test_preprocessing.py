from banking_complaints.preprocessing import sentiment_label


def test_sentiment_label_returns_expected_negative_label() -> None:
    text = "This bank made a terrible mistake and ignored my complaint."

    assert sentiment_label(text) == "negative"
