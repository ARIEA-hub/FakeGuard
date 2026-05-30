# FakeGuard — AI-Powered Fake News Detection System

> A production-ready machine learning pipeline for classifying news articles as **Real** or **Fake**.
> Includes dataset ingestion, NLP preprocessing, model training, a FastAPI backend, and a Streamlit dashboard.

[![Python](https://img.shields.io/badge/Python-3.13-blue?logo=python)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.135-009688?logo=fastapi)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.58-FF4B4B?logo=streamlit)](https://streamlit.io)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.8-f09237?logo=scikit-learn)](https://scikit-learn.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](#)

---

## Overview

FakeGuard is built to make fake news detection practical and reproducible. The repository combines:

- NLP preprocessing with `nltk` and TF-IDF
- Four classification models trained on text embeddings
- A FastAPI inference service with Pydantic validation
- A Streamlit dashboard for model comparison, metrics, and URL scraping
- Docker support for local reproduction and deployment

---

## Key Features

| Feature | Why it matters |
|---|---|
| Four classifiers | Logistic Regression, Random Forest, Gradient Boosting, Linear SVC for model comparison |
| TF-IDF tokenization | 50k vocabulary with bigrams reduces noise and captures phrase patterns |
| Singleton inference | Loads models once at startup for fast API responses |
| Input validation | Pydantic enforces text length and URL format before inference |
| URL scraping | BeautifulSoup extracts article text from live URLs |
| Metrics dashboard | Compare model performance and inspect predictions visually |
| Docker-ready | API + UI can launch together with one command |

---

## Tech Stack

- Python 3.11+ / 3.13
- FastAPI for the backend
- Streamlit for the UI
- scikit-learn for modeling
- NLTK + TF-IDF for text preprocessing
- Docker and docker-compose for local deployment

---

## Quick Start (Local)

```bash
cd fake-news-detector
python -m venv .venv
.venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Train with sample data

```bash
python scripts/run_training.py --sample
```

### Train with the full ISOT dataset

```bash
python scripts/download_data.py
python scripts/run_training.py
```

### Start the backend

```bash
uvicorn src.app.main:app --reload --host 0.0.0.0 --port 8000
```

### Start the dashboard

```bash
streamlit run src/app/streamlit_app.py --server.port 8501
```

Open:

- http://localhost:8501 for the dashboard
- http://localhost:8000/docs for FastAPI docs

---

## Docker Usage

```bash
docker-compose up --build
```

To train models inside the API container:

```bash
docker-compose exec api python scripts/run_training.py
```

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Service health and model loading state |
| POST | `/predict` | Classify a single article |
| POST | `/predict/all-models` | Compare predictions from all available models |
| POST | `/predict/url` | Scrape a URL and classify the extracted text |
| POST | `/predict/batch` | Classify up to 50 articles in one request |
| GET | `/metrics` | Return evaluation metrics for trained models |
| GET | `/models` | List models that are available for inference |

---

## Running Tests

```bash
pytest tests/ -v --cov=src --cov-report=term-missing
```

---

## Project Structure

```text
fake-news-detector/
├── data/
│   ├── processed/
│   └── raw/
├── models/
│   ├── metrics/
│   └── saved/
├── scripts/
│   ├── download_data.py
│   └── run_training.py
├── src/
│   ├── app/
│   │   ├── main.py
│   │   └── streamlit_app.py
│   ├── data/
│   │   ├── ingestion.py
│   │   ├── preprocessing.py
│   │   └── validation.py
│   ├── models/
│   │   ├── evaluator.py
│   │   ├── predictor.py
│   │   └── trainer.py
│   └── utils/
│       └── helpers.py
├── tests/
│   ├── test_api.py
│   ├── test_predictor.py
│   └── test_preprocessing.py
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

## Notes

- The repository is designed to be easy to run locally.
- If you are using Windows, use `.
venv\Scripts\activate` to activate the virtual environment.
- If the Kaggle dataset download fails, copy `Fake.csv` and `True.csv` manually into `data/raw/`.
