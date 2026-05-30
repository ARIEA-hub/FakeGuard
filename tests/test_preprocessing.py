import pandas as pd
import pytest

from src.data.preprocessing import TextPreprocessor, TFIDFPipeline
from src.data.validation import NewsArticleInput, URLInput


class TestTextPreprocessor:
    def setup_method(self):
        self.pp = TextPreprocessor()

    def test_basic_cleaning(self):
        result = self.pp.clean_text("Hello World! This is a TEST article about Politics.")
        assert isinstance(result, str)
        assert len(result) > 0
        assert "!" not in result
        assert result == result.lower()

    def test_url_removal(self):
        text = "See more at https://example.com and http://another.org/path"
        result = self.pp.clean_text(text)
        assert "https" not in result
        assert "example.com" not in result

    def test_empty_string_returns_empty(self):
        assert self.pp.clean_text("") == ""

    def test_non_string_input_returns_empty(self):
        assert self.pp.clean_text(None) == ""  # type: ignore[arg-type]
        assert self.pp.clean_text(42) == ""    # type: ignore[arg-type]

    def test_html_stripped(self):
        text = "<p>Breaking news: <strong>something happened</strong> today.</p>"
        result = self.pp.clean_text(text)
        assert "<p>" not in result
        assert "<strong>" not in result

    def test_process_dataframe_adds_columns(self):
        df = pd.DataFrame({
            "title": ["Federal Reserve raises rates"],
            "text": ["The Fed announced a quarter-point increase citing persistent inflation pressures."],
            "label": [1],
        })
        out = self.pp.process_dataframe(df)
        assert "cleaned_text" in out.columns
        assert "combined_text" in out.columns
        assert len(out) == 1

    def test_process_dataframe_drops_fully_empty(self):
        df = pd.DataFrame({
            "title": ["", "Real headline"],
            "text": ["", "Detailed and meaningful article content about economics."],
            "label": [0, 1],
        })
        out = self.pp.process_dataframe(df)
        assert len(out) == 1  # empty row should be dropped


class TestTFIDFPipeline:
    def test_fit_transform_shape(self):
        texts = pd.Series([
            "politics government president election campaign",
            "stock market economy financial quarterly report",
            "sports basketball championship team playoff win",
        ])
        pipe = TFIDFPipeline(max_features=500)
        X = pipe.fit_transform(texts)
        assert X.shape[0] == 3
        assert pipe.is_fitted

    def test_transform_single_string(self):
        texts = pd.Series(["politics government election", "economy stocks market"])
        pipe = TFIDFPipeline(max_features=500)
        pipe.fit_transform(texts)
        X = pipe.transform("breaking news political scandal")
        assert X.shape[0] == 1

    def test_transform_before_fit_raises(self):
        with pytest.raises(RuntimeError, match="not fitted"):
            TFIDFPipeline().transform("some text")


class TestValidation:
    def test_valid_article(self):
        a = NewsArticleInput(
            title="Breaking news today",
            text="This is a detailed news article with more than five words.",
        )
        assert a.title == "Breaking news today"

    def test_too_short_text_raises(self):
        with pytest.raises(Exception):
            NewsArticleInput(text="hi")

    def test_valid_url(self):
        u = URLInput(url="https://www.bbc.com/news/world-123456")
        assert "bbc.com" in u.url

    def test_invalid_url_raises(self):
        with pytest.raises(Exception):
            URLInput(url="not-a-url-at-all")

    def test_none_title_becomes_empty_string(self):
        a = NewsArticleInput(title=None, text="This is a complete news article body text.")
        assert a.title == ""