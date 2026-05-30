import logging
import time
from contextlib import asynccontextmanager
from typing import Dict

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.data.validation import (
    BatchPredictionRequest,
    NewsArticleInput,
    PredictionResponse,
    URLInput,
)
from src.models.predictor import FakeNewsPredictor
from src.utils.helpers import scrape_article_from_url

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s"
)
logger = logging.getLogger(__name__)

predictor = FakeNewsPredictor()


@asynccontextmanager
async def lifespan(app: FastAPI):
    if predictor.is_ready():
        logger.info("Models ready: %s", predictor.get_available_models())
    else:
        logger.warning("Models NOT loaded — run training pipeline first.")
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title="FakeGuard — Fake News Detection API",
    description="ML-powered REST API to classify news articles as Real or Fake.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_process_time(request: Request, call_next):
    t0 = time.time()
    response = await call_next(request)
    response.headers["X-Process-Time"] = f"{time.time() - t0:.4f}s"
    return response


# ── Health ─────────────────────────────────────────────────────────────────────

@app.get("/", tags=["Health"])
def root():
    return {
        "service": "FakeGuard Fake News Detection API",
        "status": "operational" if predictor.is_ready() else "models_not_loaded",
        "available_models": predictor.get_available_models(),
    }


@app.get("/health", tags=["Health"])
def health_check():
    if not predictor.is_ready():
        raise HTTPException(status_code=503, detail="Models not trained/loaded yet.")
    return {"status": "healthy", "models": predictor.get_available_models()}


# ── Prediction ─────────────────────────────────────────────────────────────────

@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict_article(article: NewsArticleInput, model_name: str = "logistic_regression"):
    if not predictor.is_ready():
        raise HTTPException(status_code=503, detail="Models not trained. Run the training pipeline.")
    try:
        return predictor.predict(article, model_name=model_name)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.exception("Unhandled prediction error")
        raise HTTPException(status_code=500, detail="Internal prediction error.")


@app.post("/predict/all-models", tags=["Prediction"])
def predict_all_models(article: NewsArticleInput):
    if not predictor.is_ready():
        raise HTTPException(status_code=503, detail="Models not trained.")
    try:
        results = predictor.predict_all_models(article)
        return {name: result.model_dump() for name, result in results.items()}
    except Exception:
        logger.exception("Multi-model prediction error")
        raise HTTPException(status_code=500, detail="Prediction failed.")


@app.post("/predict/url", tags=["Prediction"])
def predict_from_url(payload: URLInput, model_name: str = "logistic_regression"):
    if not predictor.is_ready():
        raise HTTPException(status_code=503, detail="Models not trained.")

    title, text = scrape_article_from_url(payload.url)
    if not text:
        raise HTTPException(
            status_code=422,
            detail=(
                "Could not extract sufficient text from the provided URL. "
                "The page may be paywalled, bot-protected, or contain no parseable article content."
            ),
        )

    try:
        article = NewsArticleInput(title=title or "", text=text)
        result = predictor.predict(article, model_name=model_name)
        return {**result.model_dump(), "source_url": payload.url, "extracted_title": title}
    except Exception as exc:
        logger.exception("URL prediction error for %s", payload.url)
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/predict/batch", tags=["Prediction"])
def predict_batch(request: BatchPredictionRequest):
    if not predictor.is_ready():
        raise HTTPException(status_code=503, detail="Models not trained.")

    results = []
    for article in request.articles:
        try:
            result = predictor.predict(
                article,
                model_name=request.model_name or "logistic_regression",
            )
            results.append({"status": "success", "result": result.model_dump()})
        except Exception as exc:
            results.append({"status": "error", "detail": str(exc)})

    return {"results": results, "total": len(results)}


# ── Analytics ──────────────────────────────────────────────────────────────────

@app.get("/metrics", tags=["Analytics"])
def get_metrics():
    metrics = predictor.get_metrics()
    if not metrics:
        raise HTTPException(status_code=404, detail="No metrics found. Train the models first.")
    return metrics


@app.get("/models", tags=["Analytics"])
def list_models():
    return {
        "available_models": predictor.get_available_models(),
        "default_model": "logistic_regression",
    }


if __name__ == "__main__":
    uvicorn.run("src.app.main:app", host="0.0.0.0", port=8000, reload=True)