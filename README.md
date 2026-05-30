# 🔍 FakeGuard — AI-Powered Fake News Detection System

> A machine learning pipeline that classifies news articles as **Real** or **Fake** using
> an ensemble of four NLP models, a FastAPI inference backend, and an interactive Streamlit
> dashboard — with a built-in OOD (out-of-distribution) reliability guard.

[![Python](https://img.shields.io/badge/Python-3.13-blue?logo=python)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.135-009688?logo=fastapi)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.58-FF4B4B?logo=streamlit)](https://streamlit.io)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.8-f09237?logo=scikit-learn)](https://scikit-learn.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](#)

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Model Performance](#model-performance)
- [Tech Stack](#tech-stack)
- [How It Works](#how-it-works)
- [Quick Start](#quick-start-local)
- [Starting Both Services](#starting-both-services)
- [Docker Usage](#docker-usage)
- [Augmented Training](#augmented-training-fix-for-diverse-news)
- [API Reference](#api-reference)
- [Running Tests](#running-tests)
- [Project Structure](#project-structure)
- [⚠️ Limitations & Known Boundaries](#️-limitations--known-boundaries)
- [Troubleshooting](#troubleshooting)

---

## Overview

FakeGuard combines four machine learning classifiers with a TF-IDF text pipeline to identify
misinformation in news articles. It includes two layers of reliability:

**Primary layer — trained model ensemble.** Four classifiers vote on every article.
Predictions where all four models agree are highly reliable within the training distribution.

**Secondary layer — OOD ensemble guard.** When the models disagree significantly
(standard deviation of P(FAKE) across models > 0.15), the API flags the result as
`UNCERTAIN_OOD` rather than returning a silent, misleading confident prediction.

---

## Key Features

| Feature | Engineering decision |
|---|---|
| Four classifiers | Logistic Regression, Random Forest, Gradient Boosting, Linear SVC |
| TF-IDF (50k features, bigrams) | `sublinear_tf=True`, `min_df=2` — removes rare and noisy tokens |
| Singleton predictor | All four models loaded once at startup — no per-request disk I/O |
| LinearSVC calibration | Sigmoid over `decision_function` produces calibrated probabilities |
| OOD ensemble σ guard | Flags high-disagreement predictions as unreliable before they mislead users |
| Pydantic v2 validation | Rejects empty text, sub-5-word inputs, and malformed URLs before inference |
| Publisher signature stripping | Removes Reuters/AP/AFP datelines so models learn content, not wire-service style |
| URL scraping | BeautifulSoup with `<article>` → `<p>` fallback, 10-second timeout |
| Graceful 422 / 503 | Paywalled URLs and missing models return descriptive errors, never silent failures |
| Docker-compose | Both services with persistent model volumes and a healthcheck-gated startup |

---

## Model Performance

Evaluated on a **stratified 20% holdout** of the ISOT Fake News Dataset (44,898 articles).

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Linear SVC | 0.9977 | 0.9977 | 0.9977 | 0.9977 | 1.0000 |
| Gradient Boosting | 0.9963 | 0.9963 | 0.9963 | 0.9963 | 0.9991 |
| Logistic Regression | 0.9915 | 0.9915 | 0.9915 | 0.9915 | 0.9996 |
| Random Forest | 0.9901 | 0.9901 | 0.9901 | 0.9901 | 0.9994 |

> These figures apply **only within the training distribution** (Reuters-style English,
> 2015–2018 US political news). See [Limitations](#️-limitations--known-boundaries) for the
> full picture on what these numbers do and do not guarantee.

---

## Tech Stack

```
Data pipeline    Pandas · NLTK · scikit-learn TfidfVectorizer
Models           Logistic Regression · Random Forest · Gradient Boosting · LinearSVC
API              FastAPI 0.135 · Uvicorn · Pydantic v2
UI               Streamlit 1.58 · Plotly
Scraping         requests · BeautifulSoup4 · lxml
Augmentation     feedparser · GitHub open datasets (fake_or_real_news, LIAR)
Deploy           Docker · docker-compose
Launcher         start_all.py (Python stdlib, zero extra deps)
```

---

## How It Works

```
Raw article text
      │
      ▼
TextPreprocessor          lowercase → strip URLs/HTML → lemmatise → remove stopwords
      │
      ▼
TFIDFPipeline             50,000 features · bigrams · sublinear_tf
      │
      ▼
Four ML models            Logistic Regression (default) · Random Forest
                          Gradient Boosting · Linear SVC
      │
      ▼
OOD σ gate                stdev(P(FAKE) across models) > 0.15 → reliability = UNCERTAIN_OOD
      │
      ▼
PredictionResponse        label · confidence · probabilities · reliability · ensemble_sigma
```

The vectoriser and all four models are serialised together. Replacing one without the other
silently breaks predictions — both must always be retrained and deployed as a set.

---

## Quick Start (Local)

```bash
# 1. Clone and enter the repository
git clone https://github.com/yourname/fake-news-detector.git
cd fake-news-detector

# 2. Create and activate a virtual environment
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 4a. Download the ISOT dataset (requires Kaggle CLI)
python scripts/download_data.py

# 4b. Or place Fake.csv and True.csv in data/raw/ manually
#     https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset

# 4c. Quick smoke test — no dataset required
python scripts/run_training.py --sample

# 5. Train all four models
python scripts/run_training.py
```

---

## Starting Both Services

> ⚠️ **Important:** FakeGuard runs as two processes that must be alive at the same time.
> The Streamlit dashboard calls the FastAPI backend — starting one without the other
> causes "API offline" errors. Do not press Ctrl+C on the API before opening the dashboard.

### Option A — one command (recommended)

```bash
python start_all.py
```

Starts both services, streams their logs to one terminal, and shuts both down cleanly
on a single Ctrl+C.

### Option B — two separate terminals

**Terminal 1** (keep this window open):
```bash
uvicorn src.app.main:app --host 0.0.0.0 --port 8000
```

**Terminal 2** (new window):
```bash
streamlit run src/app/streamlit_app.py --server.port 8501
```

### Option C — Windows batch launcher

Double-click `start.bat`. Opens two separate Command Prompt windows, one per service.

### Service URLs

| Service | URL |
|---|---|
| Streamlit dashboard | http://localhost:8501 |
| FastAPI backend | http://localhost:8000 |
| Interactive API docs | http://localhost:8000/docs |

---

## Docker Usage

```bash
# Build and start both services
docker-compose up --build

# Train models inside the running API container
docker-compose exec api python scripts/run_training.py

# Run the test suite inside the container
docker-compose exec api pytest tests/ -v

# Stream API logs
docker-compose logs -f api
```

---

## Augmented Training (Fix for Diverse News)

The default ISOT-trained models are biased toward Reuters wire-service style (US politics,
2015–2018). To fix this and improve recall on international and multi-domain real news,
use the augmented training pipeline:

```bash
# Auto-downloads two open datasets from GitHub (no Kaggle account needed)
python scripts/run_augmented_training.py --no-rss

# Full run — also scrapes live RSS feeds (BBC, Guardian, Al Jazeera, Times of India, etc.)
# Run on your local machine where these URLs are accessible
python scripts/run_augmented_training.py

# With WELFake for maximum diversity (72k articles — download from Kaggle first)
python scripts/run_augmented_training.py --welfake data/raw/WELFake_Dataset.csv

# Fast integration test (no ISOT required, ~6,000 articles)
python scripts/run_augmented_training.py --no-rss --max-articles 6000
```

The augmented pipeline:
- Blends five sources: ISOT, fake_or_real_news, LIAR dataset, WELFake (optional), RSS feeds (optional)
- Strips Reuters/AP/AFP datelines so models learn semantic content instead of publisher formatting
- Holds out 15% of non-ISOT real articles as an OOD validation slice
- Prints an OOD recall table — target is ≥ 80% recall on the diverse real-news holdout

After augmented training, restart both services:
```bash
python start_all.py
```

---

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Liveness check — 503 if models are not loaded |
| POST | `/predict` | Classify a single article. Select model via `?model_name=` |
| POST | `/predict/all-models` | Run all four models and return a comparison |
| POST | `/predict/url` | Scrape a URL and classify the extracted text |
| POST | `/predict/batch` | Classify up to 50 articles in one request |
| GET | `/metrics` | Per-model evaluation metrics from the last training run |
| GET | `/models` | List currently loaded model names |

### Example request

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"title": "Fed raises rates by 25 basis points",
       "text": "The Federal Reserve announced Wednesday it raised its benchmark interest rate by a quarter percentage point citing continued progress toward its 2 percent inflation target."}' \
  -G --data-urlencode "model_name=logistic_regression"
```

### Example response (post-augmentation)

```json
{
  "label": "REAL",
  "confidence": 0.9312,
  "probabilities": { "FAKE": 0.0688, "REAL": 0.9312 },
  "model_used": "logistic_regression",
  "input_text_length": 187,
  "warning": null,
  "reliability": "HIGH",
  "ensemble_sigma": 0.0341,
  "ood_status": null
}
```

### OOD response (when models disagree)

When `reliability` is `UNCERTAIN_OOD`, treat the prediction as advisory only:

```json
{
  "label": "FAKE",
  "confidence": 0.7782,
  "reliability": "UNCERTAIN_OOD",
  "ensemble_sigma": 0.1745,
  "ood_status": "Uncertain — Input pattern deviates from historical baseline training structures. This article may be from a source, language style, or topic domain not well-represented in the training corpus. Treat with caution and apply independent verification."
}
```

### Reliability tiers

| Tier | σ range | Meaning |
|---|---|---|
| `HIGH` | σ ≤ 0.10 | All models agree closely. Reliable prediction. |
| `MEDIUM` | σ ≤ 0.15 | Minor disagreement. Result is likely reliable. |
| `LOW` | σ ≤ 0.20 | Notable disagreement. Treat with caution. |
| `UNCERTAIN_OOD` | σ > 0.15 + gate | Article may be outside the training distribution. |

---

## Running Tests

```bash
# Full suite (no trained model files required)
pytest tests/ -v --cov=src --cov-report=term-missing

# OOD-specific tests only (includes σ math, gate logic, augmenter unit tests)
pytest tests/test_ood_handling.py -v -k "not requires_models"

# Including live model tests (requires trained .pkl files)
pytest tests/test_ood_handling.py -v
```

---

## Project Structure

```
fake-news-detector/
├── src/
│   ├── data/
│   │   ├── data_augmenter.py     multi-source ingestion + publisher signature stripping
│   │   ├── ingestion.py          ISOT loader + sample dataset generator
│   │   ├── preprocessing.py      TextPreprocessor + TFIDFPipeline
│   │   └── validation.py         Pydantic v2 schemas (includes OOD response fields)
│   ├── models/
│   │   ├── evaluator.py          confusion matrix + ROC helpers
│   │   ├── predictor.py          singleton inference engine + OOD σ gate
│   │   └── trainer.py            training loop for all four models
│   ├── app/
│   │   ├── main.py               FastAPI routes
│   │   └── streamlit_app.py      4-tab Streamlit dashboard
│   └── utils/
│       └── helpers.py            URL scraper + rate limiter
├── scripts/
│   ├── download_data.py          Kaggle CLI dataset downloader
│   ├── run_training.py           standard ISOT training entry point
│   └── run_augmented_training.py diverse multi-source training with OOD evaluation
├── tests/
│   ├── conftest.py
│   ├── test_api.py
│   ├── test_ood_handling.py      OOD gate + augmenter tests (15 tests)
│   ├── test_predictor.py
│   └── test_preprocessing.py
├── data/
│   ├── raw/                      ← Fake.csv + True.csv (gitignored)
│   └── processed/                ← generated CSVs (gitignored)
├── models/
│   ├── saved/                    ← .pkl files (gitignored)
│   └── metrics/                  ← JSON metric files
├── start_all.py                  cross-platform launcher ← use this
├── start.bat                     Windows double-click launcher
├── start.sh                      Linux / macOS launcher
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## ⚠️ Limitations & Known Boundaries

This section is required reading before using FakeGuard in any real-world context.

### Training data scope

FakeGuard was trained on the [ISOT Fake News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset),
which has a specific and narrow scope. Understanding what it contains determines
what the system can and cannot reliably classify.

**What the training data covers:**

| Category | Details |
|---|---|
| Real news source | Exclusively Reuters wire service |
| Fake news sources | 21 known fake news websites (PoliticsUS, US News Flash, etc.) |
| Time period | 2015 – 2018 only |
| Primary topic | US domestic politics (Trump, Clinton, Congress, elections) |
| Language | English only |
| Article style | Reuters AP-style formal wire copy |

---

### Domains where the model is reliable

The system performs at 99%+ F1 on content that matches the training distribution:

- US political news written in formal AP / Reuters style
- Conspiracy-theory-style fake news (anonymous sources, unverifiable claims, emotional language)
- English-language articles from 2015–2018 on US government, elections, and policy

---

### Domains where reliability is reduced or unknown

The following categories are **outside the training distribution**. The model may
return high-confidence predictions on these inputs, but those numbers do not reflect
real-world accuracy. The OOD σ guard will flag many of these — but not all.

**By news source:**

| Source type | Why it's problematic |
|---|---|
| BBC, Guardian, Al Jazeera | Different sentence structures, vocabulary, and style from Reuters wire copy |
| Indian outlets (Times of India, The Hindu, NDTV) | Different English register, Indian-English idioms, local context |
| Australian, Canadian, South African outlets | Regional news styles, different political vocabulary |
| Tabloids (Daily Mail, NY Post) | Sensational but often factual — different style signals than fake news sites |
| Tech blogs and trade press | Domain-specific vocabulary underrepresented in training data |
| Academic or scientific news | Formal but factual — may read differently from Reuters political copy |
| Financial and business news (non-Reuters) | Different stylistic markers not learned during training |

**By topic domain:**

| Topic | Why it's problematic |
|---|---|
| Sports news | Rarely appeared in training data; statistical/score reporting style is unfamiliar |
| Entertainment and celebrity news | Informal tone can pattern-match to fake news stylistics |
| Science and health news | Technical vocabulary has low TF-IDF weight in the trained vectoriser |
| International politics (non-US) | Very few training examples for EU, Asia, Middle East, African politics |
| Economics and finance | Moderate coverage only through US Treasury / Fed-related Reuters articles |
| Local and regional news | No representation in training data |
| Opinion and editorial | Subjective writing style may resemble fake news patterns |
| Satire | Satirical language is semantically similar to conspiracy content |

**By time period:**

| Period | Why it's problematic |
|---|---|
| 2019 and later | Vocabulary drift: COVID-19, Ukraine, AI boom, new political figures — all unseen during training |
| Before 2015 | Limited representation in training data |

**By language:**

| Language | Status |
|---|---|
| English | Supported (within distribution constraints above) |
| Hindi, Tamil, Telugu, Bengali | Not supported |
| Urdu, Arabic, French, Spanish | Not supported |
| Any non-English language | Not supported — results meaningless |

---

### What high confidence does not mean

A prediction of 99% FAKE does **not** mean the article is fake. It means the article's
n-gram pattern in TF-IDF space is 99% similar to articles in the fake-news cluster of the
training set. An article from a credible Indian newspaper about cricket will pattern-match
to fake news because neither Indian papers nor cricket were in the training data.

The OOD σ guard catches many of these cases, but it is not infallible. If all four models
happen to agree on the wrong answer (because the input superficially resembles training
fake-news patterns), the guard will not fire and the wrong prediction will be returned with
high apparent confidence.

---

### What this system is and is not

| FakeGuard IS | FakeGuard IS NOT |
|---|---|
| A research and educational demonstration of ML-based text classification | A production fact-checking tool suitable for journalism or legal decisions |
| Reliable for classifying Reuters-style US political content from 2015–2018 | A general-purpose detector for all news in all domains |
| Useful for comparing ML model performance on a benchmark NLP task | A replacement for human editorial judgment or professional fact-checkers |
| A starting point to build a domain-specific classifier for your use case | Authoritative on articles outside its training distribution |

---

### Improving coverage with augmented training

The `run_augmented_training.py` script adds three additional data sources
(fake_or_real_news, LIAR dataset, and optional RSS feeds and WELFake) to partially
address the Reuters-only bias. After augmented training:

- OOD real-news recall improves from ~30–50% to approximately **80–89%** on a diverse holdout
- The system gains exposure to Guardian, PolitiFact, CNN, and Politico styles
- International and sports coverage remains limited without RSS feed data
- The OOD σ guard remains active as a permanent safety net

Even with augmented training, this system should not be used as a definitive source of truth.
It is a classification model trained on labelled data, not a reasoning system with factual knowledge.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| "API offline" in sidebar | FastAPI not running | Use `python start_all.py` — both must run simultaneously |
| "API docs" page not found | API was stopped | Restart the API and keep its window open |
| uvicorn window appears to hang | Web servers block intentionally — this is correct | Open a separate terminal for Streamlit |
| Models not found (503) | Training not run yet | `python scripts/run_training.py --sample` |
| Real article classified as FAKE | Distribution shift (OOD input) | Check `reliability` field — if `UNCERTAIN_OOD`, the article is outside training scope |
| High confidence wrong prediction | OOD input that bypassed the gate | Run augmented training; treat high-confidence results on non-Reuters content with scepticism |
| Port already in use | Previous process still running | `netstat -ano \| findstr :8000` then kill that PID |
| `--sample` overwrites production models | `run_training.py --sample` run in wrong environment | Set `ENVIRONMENT=production` to block destructive commands |
