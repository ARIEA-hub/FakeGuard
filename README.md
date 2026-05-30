# FakeGuard — AI-Powered Fake News Detection System

> A production-ready machine learning pipeline that classifies news articles as
> **Real** or **Fake** — with a multi-model comparison engine, live URL scraping,
> a REST API, and an interactive analytics dashboard.

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue?logo=python)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-backend-009688?logo=fastapi)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-dashboard-FF4B4B?logo=streamlit)](https://streamlit.io)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-f09237?logo=scikit-learn)](https://scikit-learn.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](#)

---

## How the system works (two-process architecture)

FakeGuard runs as **two independent services that must both be running at the same time**:

| Service | What it does | Port |
|---|---|---|
| **FastAPI** (`uvicorn`) | Serves the ML API — loads all 4 models into memory | 8000 |
| **Streamlit** | Provides the browser dashboard — calls the API | 8501 |

> ⚠️ **Common mistake**: Starting Streamlit after stopping uvicorn will always show
> "API offline" because the two processes need to be alive simultaneously.

---

## Quick start (recommended — one command)

```bash
# From the project root:
python start_all.py
```

This launches both services in the same terminal, streams their logs, and
shuts both down cleanly when you press **Ctrl+C once**.

Open in browser:
- Dashboard → **http://localhost:8501**
- API docs  → **http://localhost:8000/docs**

---

## Alternative: two separate terminal windows

If you prefer separate log streams (easier to read each service's output):

### Terminal 1 — start the API (keep this window open)
```bash
cd C:\Users\A\Desktop\fake-news-detector   # adjust path
.venv\Scripts\activate                     # activate your virtual env
uvicorn src.app.main:app --host 0.0.0.0 --port 8000
```
The terminal will show log lines and stay running — **this is correct**.
Do **not** press Ctrl+C. Open a second terminal for the next step.

### Terminal 2 — start the dashboard (keep this window open too)
```bash
cd C:\Users\A\Desktop\fake-news-detector   # same project root
.venv\Scripts\activate                     # same virtual env
streamlit run src/app/streamlit_app.py --server.port 8501
```

Both windows must remain open. The dashboard at http://localhost:8501 will
show ✅ API online in the sidebar once the API is healthy.

---

## Windows batch launcher (double-click alternative)

```bash
# Double-click start.bat in Windows Explorer, or run from CMD:
start.bat
```

This opens two separate Command Prompt windows — one for the API, one for
the dashboard — and prints the URLs. Close those two windows to stop both
services.

---

## Full setup from scratch

```bash
# 1. Clone the repository
git clone https://github.com/yourname/fake-news-detector.git
cd fake-news-detector

# 2. Create and activate a virtual environment
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

# 3. Install all dependencies
pip install -r requirements.txt

# 4a. Train on the ISOT dataset (requires Fake.csv + True.csv in data/raw/)
python scripts/run_training.py

# 4b. Quick smoke-test with the built-in 8-article sample (no CSV needed)
python scripts/run_training.py --sample

# 5. Start both services (choose any method above)
python start_all.py
```

---

## Dataset setup (ISOT Fake News Dataset)

The full training dataset is **not included** in the repository (221 MB).

**Option A — Kaggle CLI (automated)**
```bash
pip install kaggle
# Place ~/.kaggle/kaggle.json from kaggle.com/settings → API
python scripts/download_data.py
```

**Option B — Manual download**
1. Visit https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset
2. Download `Fake.csv` and `True.csv`
3. Move them into `data/raw/`

**Option C — Skip it** — run `python scripts/run_training.py --sample` to
train on the 8-article built-in dataset for a functional demo without the CSV.

---

## Docker (single-command alternative)

```bash
# Build images and start both services
docker-compose up --build

# Run training inside the API container
docker-compose exec api python scripts/run_training.py
```

Services:
- API       → http://localhost:8000
- Dashboard → http://localhost:8501

---

## Key features

| Feature | Engineering choice |
|---|---|
| 4 ML classifiers | Logistic Regression, Random Forest, Gradient Boosting, Linear SVC |
| TF-IDF (50k bigrams) | `sublinear_tf=True`, `min_df=2` — removes rare/noisy tokens |
| Singleton predictor | All 4 models loaded once at startup — no per-request disk I/O |
| LinearSVC calibration | Sigmoid over `decision_function` produces calibrated probabilities |
| URL scraping | BeautifulSoup with `<article>` → `<p>` fallback, 10 s timeout |
| Pydantic v2 validation | Rejects empty text, sub-5-word inputs, and malformed URLs before inference |
| Graceful 422/503 | Paywalled URLs and missing models return descriptive errors, never silent failures |
| Docker-compose | Both services with persistent model volumes, healthcheck-gated startup |

---

## API reference

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Liveness check — 503 if models not loaded |
| POST | `/predict` | Classify article text, select model via `?model_name=` |
| POST | `/predict/all-models` | Run all 4 models and return comparison |
| POST | `/predict/url` | Scrape URL and classify |
| POST | `/predict/batch` | Classify up to 50 articles at once |
| GET | `/metrics` | Per-model evaluation metrics from training |
| GET | `/models` | List loaded model names |

Interactive docs: **http://localhost:8000/docs** (only reachable while API is running)

---

## Training script options

```bash
# Full ISOT dataset (default paths)
python scripts/run_training.py

# Custom CSV paths
python scripts/run_training.py --fake data/raw/Fake.csv --true data/raw/True.csv

# Built-in sample (no CSV required)
python scripts/run_training.py --sample

# Limit rows for faster iteration
python scripts/run_training.py --limit 5000
```

---

## Running tests

```bash
pytest tests/ -v --cov=src --cov-report=term-missing
```

All tests use a mocked predictor so they pass without trained model artefacts
present. No GPU or model files required for CI.

---

## Project structure

```
fake-news-detector/
├── src/
│   ├── data/        ingestion · preprocessing · validation
│   ├── models/      trainer · evaluator · predictor
│   ├── app/         FastAPI backend (main.py) · Streamlit UI (streamlit_app.py)
│   └── utils/       URL scraper · helpers
├── scripts/
│   ├── run_training.py     CLI training entry point
│   └── download_data.py    Kaggle dataset downloader
├── tests/
│   ├── conftest.py
│   ├── test_api.py
│   ├── test_predictor.py
│   └── test_preprocessing.py
├── data/
│   ├── raw/           ← put Fake.csv + True.csv here (gitignored)
│   └── processed/     ← generated (gitignored)
├── models/
│   ├── saved/         ← .pkl files (gitignored)
│   └── metrics/       ← JSON metric files
├── start_all.py       ← cross-platform launcher  ★ use this
├── start.bat          ← Windows batch launcher
├── start.sh           ← Linux / macOS launcher
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
├── .env.example
└── README.md
```

---

## Tech stack

```
Data       Pandas · NLTK · scikit-learn TfidfVectorizer
Models     Logistic Regression · Random Forest · Gradient Boosting · LinearSVC
API        FastAPI · Uvicorn · Pydantic v2
UI         Streamlit · Plotly
Scraping   requests · BeautifulSoup4 · lxml
Deploy     Docker · docker-compose
Launcher   start_all.py (subprocess, zero extra deps)
```

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| "API offline" in sidebar | FastAPI not running | Open a second terminal and start the API |
| "API docs" page not found | API was stopped | Restart API, keep its window open |
| uvicorn seems to hang | It's a server — it blocks intentionally | Leave it running, open a new terminal |
| Models not found (503) | Training not run yet | `python scripts/run_training.py --sample` |
| Port already in use | Previous process still alive | `netstat -ano \| findstr :8000` then kill that PID |
