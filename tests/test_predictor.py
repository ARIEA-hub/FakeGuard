from src.data.validation import NewsArticleInput


class TestPredictorSingleton:
    def test_singleton_identity(self):
        from src.models.predictor import FakeNewsPredictor
        p1 = FakeNewsPredictor()
        p2 = FakeNewsPredictor()
        assert p1 is p2

    def test_not_ready_without_artifacts(self, tmp_path, monkeypatch):
        """Predictor should report not-ready when model dir is empty."""
        import src.models.predictor as pred_module

        monkeypatch.setattr(pred_module, "MODELS_DIR", tmp_path)
        monkeypatch.setattr(pred_module, "METRICS_DIR", tmp_path)

        # Reset singleton for this test
        pred_module.FakeNewsPredictor._instance = None
        predictor = pred_module.FakeNewsPredictor()
        assert not predictor.is_ready()
        assert predictor.get_available_models() == []

        # Restore singleton state
        pred_module.FakeNewsPredictor._instance = None

    def test_input_validation_guards_predictor(self):
        """Pydantic should reject short text before it ever reaches inference."""
        import pytest
        with pytest.raises(Exception):
            NewsArticleInput(title="Test", text="too short")