import re
from typing import Dict, List, Optional
from pydantic import BaseModel, Field, field_validator


class NewsArticleInput(BaseModel):
    title: Optional[str] = Field(default="", description="Article headline")
    text: str = Field(..., min_length=10, description="Article body text")

    @field_validator("text")
    @classmethod
    def text_must_be_meaningful(cls, v: str) -> str:
        v = v.strip()
        if len(v.split()) < 5:
            raise ValueError("Text must contain at least 5 words for meaningful analysis.")
        return v

    @field_validator("title")
    @classmethod
    def sanitize_title(cls, v: Optional[str]) -> str:
        return v.strip() if v else ""


class URLInput(BaseModel):
    url: str = Field(..., description="URL of the news article to scrape and analyse")

    @field_validator("url")
    @classmethod
    def validate_url(cls, v: str) -> str:
        pattern = re.compile(
            r"^https?://"
            r"(?:(?:[A-Z0-9](?:[A-Z0-9\-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|localhost|\d{1,3}(?:\.\d{1,3}){3})"
            r"(?::\d+)?(?:/?|[/?]\S+)$",
            re.IGNORECASE,
        )
        if not pattern.match(v):
            raise ValueError(f"Invalid URL format: {v}")
        return v


class PredictionResponse(BaseModel):
    label: str                          # "FAKE" or "REAL"
    confidence: float = Field(..., ge=0.0, le=1.0)
    probabilities: Dict[str, float]     # {"FAKE": 0.85, "REAL": 0.15}
    model_used: str
    input_text_length: int
    warning: Optional[str] = None


class BatchPredictionRequest(BaseModel):
    articles: List[NewsArticleInput] = Field(..., min_length=1, max_length=50)
    model_name: Optional[str] = "logistic_regression"


class ModelMetrics(BaseModel):
    model_name: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    roc_auc: float