import re
import string
from functools import lru_cache

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.base import BaseEstimator, TransformerMixin
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from banking_complaints.config import NLTK_DATA_DIR

NLTK_DATA_DIR.mkdir(parents=True, exist_ok=True)
if str(NLTK_DATA_DIR) not in nltk.data.path:
    nltk.data.path.insert(0, str(NLTK_DATA_DIR))

NLTK_PACKAGES = {
    "stopwords": "corpora/stopwords",
    "wordnet": "corpora/wordnet.zip",
    "punkt": "tokenizers/punkt",
    "punkt_tab": "tokenizers/punkt_tab",
}


@lru_cache(maxsize=1)
def ensure_nltk_data() -> None:
    """Download NLTK assets used by the preprocessing pipeline."""
    for package, resource in NLTK_PACKAGES.items():
        try:
            nltk.data.find(resource)
        except LookupError:
            nltk.download(package, download_dir=str(NLTK_DATA_DIR), quiet=True)


def clean_text(text: str) -> str:
    """Normalize complaint text for classical ML models."""
    ensure_nltk_data()
    text = str(text).lower()
    text = re.sub(r"\d+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()
    cleaned = [
        lemmatizer.lemmatize(token)
        for token in tokens
        if token.isalpha() and token not in stop_words
    ]
    return " ".join(cleaned)


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
