from sklearn.base import BaseEstimator, TransformerMixin
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


def sentiment_label(text: str) -> str:
    """Classify complaint sentiment with VADER."""
    analyzer = SentimentIntensityAnalyzer()
    score = analyzer.polarity_scores(str(text))["compound"]
    if score >= 0.05:
        return "positive"
    if score <= -0.05:
        return "negative"
    return "neutral"


class SpacyLemmaTransformer(BaseEstimator, TransformerMixin):
    """Lemmatize complaint text with spaCy for sklearn pipelines."""

    def __init__(self, model_name: str = "en_core_web_sm", batch_size: int = 64) -> None:
        self.model_name = model_name
        self.batch_size = batch_size

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return self._clean_texts([str(text) for text in X])

    def _load_model(self):
        import spacy

        try:
            return spacy.load(self.model_name, disable=["parser", "ner"])
        except OSError as exc:
            raise OSError(
                f"spaCy model '{self.model_name}' is not installed. "
                f"Install it with: python -m spacy download {self.model_name}"
            ) from exc

    def _clean_texts(self, texts: list[str]) -> list[str]:
        nlp = self._load_model()
        cleaned = []
        for doc in nlp.pipe(texts, batch_size=self.batch_size):
            tokens = [
                token.lemma_.lower()
                for token in doc
                if not token.is_stop
                and not token.is_punct
                and not token.like_num
                and token.is_alpha
            ]
            cleaned.append(" ".join(tokens))
        return cleaned
