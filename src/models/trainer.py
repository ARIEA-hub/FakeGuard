import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

from src.data.preprocessing import TextPreprocessor, TFIDFPipeline

logger = logging.getLogger(__name__)

MODELS_DIR = Path("models/saved")
METRICS_DIR = Path("models/metrics")

AVAILABLE_MODELS: Dict[str, Any] = {
    "logistic_regression": LogisticRegression(
        max_iter=1000, C=1.0, solver="lbfgs", random_state=42, n_jobs=-1
    ),
    "random_forest": RandomForestClassifier(
        n_estimators=200, max_depth=20, random_state=42, n_jobs=-1
    ),
    "gradient_boosting": GradientBoostingClassifier(
        n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42
    ),
    "linear_svc": LinearSVC(max_iter=2000, C=1.0, random_state=42),
}


class ModelTrainer:
    def __init__(self) -> None:
        self.preprocessor = TextPreprocessor()
        self.tfidf = TFIDFPipeline(max_features=50_000, ngram_range=(1, 2))
        self.trained_models: Dict[str, Any] = {}
        self.metrics: Dict[str, Dict] = {}
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        METRICS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Data preparation ───────────────────────────────────────────────────

    def prepare_data(
        self, df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        logger.info("Preprocessing text corpus…")
        df = self.preprocessor.process_dataframe(df)

        X = self.tfidf.fit_transform(df["cleaned_text"])
        y = df["label"].values

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, random_state=42, stratify=y
        )
        logger.info("Train=%d  Test=%d", X_train.shape[0], X_test.shape[0])
        return X_train, X_test, y_train, y_test

    # ── Per-model steps ────────────────────────────────────────────────────

    def train_model(self, name: str, X_train: np.ndarray, y_train: np.ndarray) -> Any:
        if name not in AVAILABLE_MODELS:
            raise ValueError(f"Unknown model '{name}'. Options: {list(AVAILABLE_MODELS)}")

        model = AVAILABLE_MODELS[name]
        logger.info("Training %s…", name)
        t0 = time.time()
        model.fit(X_train, y_train)
        logger.info("%s trained in %.2fs", name, time.time() - t0)
        self.trained_models[name] = model
        return model

    def evaluate_model(
        self,
        name: str,
        model: Any,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> Dict:
        y_pred = model.predict(X_test)

        if hasattr(model, "predict_proba"):
            y_score = model.predict_proba(X_test)[:, 1]
        else:
            # LinearSVC: calibrate with sigmoid
            y_score = model.decision_function(X_test)

        roc_auc = roc_auc_score(y_test, y_score)

        metrics = {
            "model_name": name,
            "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
            "precision": round(float(precision_score(y_test, y_pred, average="weighted")), 4),
            "recall": round(float(recall_score(y_test, y_pred, average="weighted")), 4),
            "f1_score": round(float(f1_score(y_test, y_pred, average="weighted")), 4),
            "roc_auc": round(float(roc_auc), 4),
            "classification_report": classification_report(
                y_test, y_pred, target_names=["FAKE", "REAL"]
            ),
        }
        self.metrics[name] = metrics

        # Persist numeric subset (exclude long string report)
        numeric = {k: v for k, v in metrics.items() if k != "classification_report"}
        with open(METRICS_DIR / f"{name}_metrics.json", "w") as fh:
            json.dump(numeric, fh, indent=2)

        logger.info(
            "%s → Acc=%.4f  F1=%.4f  AUC=%.4f",
            name,
            metrics["accuracy"],
            metrics["f1_score"],
            metrics["roc_auc"],
        )
        return metrics

    def save_model(self, name: str) -> None:
        if name not in self.trained_models:
            raise ValueError(f"Model '{name}' not trained yet.")
        joblib.dump(self.trained_models[name], MODELS_DIR / f"{name}.pkl")
        logger.info("Saved model → %s/%s.pkl", MODELS_DIR, name)

    # ── Orchestrator ───────────────────────────────────────────────────────

    def train_all_models(self, df: pd.DataFrame) -> Dict[str, Dict]:
        X_train, X_test, y_train, y_test = self.prepare_data(df)
        all_metrics: Dict[str, Dict] = {}

        for name in AVAILABLE_MODELS:
            try:
                model = self.train_model(name, X_train, y_train)
                metrics = self.evaluate_model(name, model, X_test, y_test)
                self.save_model(name)
                all_metrics[name] = metrics
            except Exception as exc:
                logger.error("Failed to train %s: %s", name, exc)

        self.tfidf.save(MODELS_DIR / "tfidf_vectorizer.pkl")

        summary = {
            k: {m: v for m, v in vals.items() if m != "classification_report"}
            for k, vals in all_metrics.items()
        }
        with open(METRICS_DIR / "all_metrics.json", "w") as fh:
            json.dump(summary, fh, indent=2)

        best = max(all_metrics, key=lambda k: all_metrics[k]["f1_score"])
        logger.info(
            "Best model: %s (F1=%.4f)", best, all_metrics[best]["f1_score"]
        )
        return all_metrics


def run_training() -> Dict[str, Dict]:
    """Importable training entry point used by scripts/run_training.py."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s"
    )
    from src.data.ingestion import create_sample_dataset, load_processed_data

    try:
        df = load_processed_data()
    except FileNotFoundError:
        logger.warning("No processed data found — using built-in sample dataset.")
        df = create_sample_dataset()

    trainer = ModelTrainer()
    return trainer.train_all_models(df)