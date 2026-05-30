import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import numpy as np

from src.data.preprocessing import TextPreprocessor, TFIDFPipeline
from src.data.validation import NewsArticleInput, PredictionResponse

logger = logging.getLogger(__name__)

MODELS_DIR = Path("models/saved")
METRICS_DIR = Path("models/metrics")
LABEL_MAP = {0: "FAKE", 1: "REAL"}


class FakeNewsPredictor:
    """
    Singleton inference engine.
    Loads the TF-IDF vectoriser and all serialised models once at startup.
    Thread-safe for FastAPI's async workers (read-only after initialisation).
    """

    _instance: Optional["FakeNewsPredictor"] = None

    def __new__(cls) -> "FakeNewsPredictor":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return
        self.preprocessor = TextPreprocessor()
        self.tfidf: Optional[TFIDFPipeline] = None
        self.models: Dict[str, object] = {}
        self.metrics: Dict[str, Dict] = {}
        self._initialized = True
        self._load_artifacts()

    # ── Startup ────────────────────────────────────────────────────────────

    def _load_artifacts(self) -> None:
        vec_path = MODELS_DIR / "tfidf_vectorizer.pkl"
        if not vec_path.exists():
            logger.warning("Vectoriser not found at %s. Run training first.", vec_path)
            return

        self.tfidf = TFIDFPipeline.load(vec_path)

        for pkl in sorted(MODELS_DIR.glob("*.pkl")):
            if pkl.name == "tfidf_vectorizer.pkl":
                continue
            try:
                self.models[pkl.stem] = joblib.load(pkl)
                logger.info("Loaded model: %s", pkl.stem)
            except Exception as exc:
                logger.error("Failed to load %s: %s", pkl.name, exc)

        metrics_path = METRICS_DIR / "all_metrics.json"
        if metrics_path.exists():
            with open(metrics_path) as fh:
                self.metrics = json.load(fh)

    # ── Public helpers ─────────────────────────────────────────────────────

    def is_ready(self) -> bool:
        return self.tfidf is not None and bool(self.models)

    def get_available_models(self) -> List[str]:
        return list(self.models.keys())

    # ── Inference ──────────────────────────────────────────────────────────

    def predict(
        self,
        article: NewsArticleInput,
        model_name: str = "logistic_regression",
    ) -> PredictionResponse:
        if not self.is_ready():
            raise RuntimeError("Models not loaded. Run the training pipeline first.")

        if model_name not in self.models:
            fallback = self.get_available_models()[0]
            logger.warning("Model '%s' not found; falling back to '%s'.", model_name, fallback)
            model_name = fallback

        combined = f"{article.title} {article.text}".strip()
        cleaned = self.preprocessor.clean_text(combined)

        warning: Optional[str] = None
        if len(cleaned.split()) < 10:
            warning = "Input text is very short — prediction confidence may be low."

        X = self.tfidf.transform(cleaned)
        model = self.models[model_name]

        prediction = int(model.predict(X)[0])

        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)[0]
            prob_fake, prob_real = float(proba[0]), float(proba[1])
        else:
            # LinearSVC: sigmoid of decision function
            score = float(model.decision_function(X)[0])
            prob_real = 1.0 / (1.0 + np.exp(-score))
            prob_fake = 1.0 - prob_real
            prediction = 1 if prob_real >= 0.5 else 0

        confidence = prob_real if prediction == 1 else prob_fake

        return PredictionResponse(
            label=LABEL_MAP[prediction],
            confidence=round(confidence, 4),
            probabilities={
                "FAKE": round(prob_fake, 4),
                "REAL": round(prob_real, 4),
            },
            model_used=model_name,
            input_text_length=len(article.text),
            warning=warning,
        )

    def predict_all_models(
        self, article: NewsArticleInput
    ) -> Dict[str, PredictionResponse]:
        results = {}
        for name in self.models:
            try:
                results[name] = self.predict(article, model_name=name)
            except Exception as exc:
                logger.error("Prediction failed for %s: %s", name, exc)
        return results

    def get_metrics(self) -> Dict:
        return self.metrics