import json
import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve

logger = logging.getLogger(__name__)
METRICS_DIR = Path("models/metrics")


class ModelEvaluator:
    """Utility class for post-training evaluation artefacts."""

    @staticmethod
    def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        return confusion_matrix(y_true, y_pred)

    @staticmethod
    def roc_data(y_true: np.ndarray, y_scores: np.ndarray) -> Dict:
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        return {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "thresholds": thresholds.tolist(),
            "auc": round(float(roc_auc_score(y_true, y_scores)), 4),
        }

    @staticmethod
    def save_report(
        model_name: str,
        metrics: Dict,
        conf_matrix: np.ndarray,
    ) -> None:
        METRICS_DIR.mkdir(parents=True, exist_ok=True)
        report = {
            "model_name": model_name,
            "metrics": {k: v for k, v in metrics.items() if k != "classification_report"},
            "confusion_matrix": conf_matrix.tolist(),
        }
        path = METRICS_DIR / f"{model_name}_full_report.json"
        with open(path, "w") as fh:
            json.dump(report, fh, indent=2)
        logger.info("Full evaluation report → %s", path)

    @staticmethod
    def load_all_metrics() -> Dict:
        path = METRICS_DIR / "all_metrics.json"
        if not path.exists():
            return {}
        with open(path) as fh:
            return json.load(fh)

    @staticmethod
    def best_model(metrics: Dict) -> str:
        if not metrics:
            return "logistic_regression"
        return max(metrics, key=lambda k: metrics[k].get("f1_score", 0))