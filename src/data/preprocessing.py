import logging
import re
from pathlib import Path
from typing import Tuple

import joblib
import numpy as np
import pandas as pd

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

logger = logging.getLogger(__name__)


def _download_nltk_resources() -> None:
    for resource in ["stopwords", "wordnet", "omw-1.4", "punkt"]:
        try:
            nltk.download(resource, quiet=True)
        except Exception as exc:
            logger.warning("NLTK download failed for %s: %s", resource, exc)


_download_nltk_resources()


class TextPreprocessor:
    """Stateless text cleaner: lowercase → strip URLs/HTML → lemmatise → drop stopwords."""

    def __init__(self) -> None:
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words("english"))

    def clean_text(self, text: object) -> str:
        if not isinstance(text, str) or not text.strip():
            return ""

        text = text.lower()
        text = re.sub(r"https?://\S+|www\.\S+", "", text)   # URLs
        text = re.sub(r"<[^>]+>", "", text)                  # HTML tags
        text = re.sub(r"[^a-zA-Z\s]", "", text)              # punctuation & digits
        text = re.sub(r"\s+", " ", text).strip()

        tokens = [
            self.lemmatizer.lemmatize(t)
            for t in text.split()
            if t not in self.stop_words and len(t) > 2
        ]
        return " ".join(tokens)

    def process_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["text"] = df["text"].fillna("")
        df["title"] = df["title"].fillna("")
        df["combined_text"] = df["title"] + " " + df["text"]
        df["cleaned_text"] = df["combined_text"].apply(self.clean_text)
        df = df[df["cleaned_text"].str.len() > 0].reset_index(drop=True)
        return df


class TFIDFPipeline:
    """Thin wrapper around TfidfVectorizer with save/load helpers."""

    def __init__(
        self,
        max_features: int = 50_000,
        ngram_range: Tuple[int, int] = (1, 2),
    ) -> None:
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            sublinear_tf=True,
            min_df=2,
            max_df=0.95,
        )
        self.is_fitted = False

    def fit_transform(self, texts: pd.Series) -> np.ndarray:
        X = self.vectorizer.fit_transform(texts)
        self.is_fitted = True
        return X

    def transform(self, texts: object) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Vectoriser not fitted. Call fit_transform first.")
        if isinstance(texts, str):
            texts = [texts]
        return self.vectorizer.transform(texts)

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.vectorizer, path)
        logger.info("Vectoriser saved → %s", path)

    @classmethod
    def load(cls, path: Path) -> "TFIDFPipeline":
        instance = cls()
        instance.vectorizer = joblib.load(path)
        instance.is_fitted = True
        logger.info("Vectoriser loaded ← %s", path)
        return instance