FROM python:3.11-slim

WORKDIR /app

# System deps for lxml
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-fetch NLTK data at build time (avoids runtime download)
RUN python -c "\
import nltk; \
nltk.download('stopwords', quiet=True); \
nltk.download('wordnet', quiet=True); \
nltk.download('omw-1.4', quiet=True); \
nltk.download('punkt', quiet=True)"

COPY . .

# Ensure output dirs exist inside image
RUN mkdir -p data/raw data/processed models/saved models/metrics

EXPOSE 8000 8501

# Default command — override in docker-compose
CMD ["uvicorn", "src.app.main:app", "--host", "0.0.0.0", "--port", "8000"]