import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch


@pytest.fixture()
def mock_predictor():
    with patch("src.app.main.predictor") as mock:
        from src.data.validation import PredictionResponse

        mock.is_ready.return_value = True
        mock.get_available_models.return_value = ["logistic_regression", "random_forest"]
        mock.predict.return_value = PredictionResponse(
            label="FAKE",
            confidence=0.87,
            probabilities={"FAKE": 0.87, "REAL": 0.13},
            model_used="logistic_regression",
            input_text_length=120,
        )
        mock.predict_all_models.return_value = {
            "logistic_regression": mock.predict.return_value,
            "random_forest": mock.predict.return_value,
        }
        mock.get_metrics.return_value = {
            "logistic_regression": {
                "accuracy": 0.987, "precision": 0.987,
                "recall": 0.987, "f1_score": 0.987, "roc_auc": 0.997,
            }
        }
        yield mock


def get_client():
    from src.app.main import app
    return TestClient(app)


def test_root_returns_200(mock_predictor):
    r = get_client().get("/")
    assert r.status_code == 200
    assert "service" in r.json()


def test_health_returns_200_when_ready(mock_predictor):
    r = get_client().get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "healthy"


def test_predict_valid_article(mock_predictor):
    r = get_client().post(
        "/predict",
        json={"title": "Shocking claim!", "text": "An extraordinary claim made without any verifiable evidence."},
        params={"model_name": "logistic_regression"},
    )
    assert r.status_code == 200
    data = r.json()
    assert data["label"] in ("FAKE", "REAL")
    assert 0 <= data["confidence"] <= 1
    assert "probabilities" in data


def test_predict_short_text_rejected(mock_predictor):
    r = get_client().post("/predict", json={"title": "Test", "text": "too short"})
    assert r.status_code == 422


def test_predict_all_models(mock_predictor):
    r = get_client().post(
        "/predict/all-models",
        json={"text": "Detailed news article about recent political developments in the region."},
    )
    assert r.status_code == 200
    assert isinstance(r.json(), dict)


def test_metrics_endpoint(mock_predictor):
    r = get_client().get("/metrics")
    assert r.status_code == 200
    data = r.json()
    assert "logistic_regression" in data


def test_models_endpoint(mock_predictor):
    r = get_client().get("/models")
    assert r.status_code == 200
    assert "available_models" in r.json()


def test_503_when_not_ready():
    with patch("src.app.main.predictor") as mock:
        mock.is_ready.return_value = False
        mock.get_available_models.return_value = []
        r = get_client().get("/health")
        assert r.status_code == 503