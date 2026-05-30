import logging
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

RAW_DATA_DIR = Path("data/raw")
PROCESSED_DATA_DIR = Path("data/processed")


def load_isot_dataset(fake_path: Path, true_path: Path) -> pd.DataFrame:
    """Load ISOT Fake-and-Real News dataset (two separate CSVs)."""
    fake_df = pd.read_csv(fake_path)
    true_df = pd.read_csv(true_path)

    fake_df["label"] = 0  # FAKE
    true_df["label"] = 1  # REAL

    df = pd.concat([fake_df, true_df], ignore_index=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    for col in ("title", "text"):
        if col not in df.columns:
            df[col] = ""

    logger.info(
        "Loaded %d articles: %d fake / %d real",
        len(df),
        len(fake_df),
        len(true_df),
    )
    return df[["title", "text", "label"]]


def load_single_csv(
    path: Path,
    text_col: str = "text",
    title_col: str = "title",
    label_col: str = "label",
) -> pd.DataFrame:
    """Flexible loader for a single CSV with configurable column names."""
    df = pd.read_csv(path)

    rename = {c: k for k, c in [("text", text_col), ("title", title_col), ("label", label_col)] if c in df.columns and c != k}
    df = df.rename(columns=rename)

    missing = [c for c in ("text", "label") if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Available: {list(df.columns)}")

    if "title" not in df.columns:
        df["title"] = ""

    if df["label"].dtype == object:
        df["label"] = df["label"].str.lower().map({"fake": 0, "real": 1, "0": 0, "1": 1})

    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)
    return df[["title", "text", "label"]]


def create_sample_dataset() -> pd.DataFrame:
    """Minimal labelled dataset for smoke-testing when no real data is present."""
    rows = [
        {"title": "SHOCKING: Scientists discover moon made of cheese",
         "text": "Anonymous sources inside NASA have reportedly confirmed that the lunar surface is composed entirely of aged cheddar, overturning decades of accepted geology.",
         "label": 0},
        {"title": "Government secretly replaced tap water with mind-control serum",
         "text": "An unnamed whistleblower with alleged government ties claims fluoride is a cover story for a neurological agent added to municipal water supplies.",
         "label": 0},
        {"title": "Drinking bleach cures all diseases, experts say",
         "text": "A group of unnamed alternative medicine practitioners are urging followers to ingest household bleach as a universal cure, claiming it purges toxins.",
         "label": 0},
        {"title": "Vaccines contain microchips designed to track citizens",
         "text": "Viral social media posts claim that COVID-19 vaccines contain RFID chips activated by 5G towers, a claim debunked by every reputable health body.",
         "label": 0},
        {"title": "Federal Reserve raises interest rates by 25 basis points",
         "text": "The Federal Reserve announced Wednesday it raised its benchmark interest rate by a quarter percentage point, citing continued progress toward its 2% inflation target.",
         "label": 1},
        {"title": "WHO publishes updated guidance on antibiotic resistance",
         "text": "The World Health Organization released comprehensive updated guidelines on antibiotic stewardship, urging hospitals to audit prescribing practices to combat resistant strains.",
         "label": 1},
        {"title": "Stock markets close higher after strong jobs report",
         "text": "US equity indices gained ground on Friday after the Labour Department reported 256,000 new jobs in December, far exceeding analyst expectations of 155,000.",
         "label": 1},
        {"title": "Scientists confirm new species of deep-sea fish in Pacific",
         "text": "Marine biologists working with NOAA have formally described a previously unknown species of snailfish found at 8,300 metres in the Mariana Trench.",
         "label": 1},
    ]
    return pd.DataFrame(rows)


def save_processed_data(df: pd.DataFrame, filename: str = "processed_data.csv") -> Path:
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = PROCESSED_DATA_DIR / filename
    df.to_csv(path, index=False)
    logger.info("Processed data saved → %s", path)
    return path


def load_processed_data(filename: str = "processed_data.csv") -> pd.DataFrame:
    path = PROCESSED_DATA_DIR / filename
    if not path.exists():
        raise FileNotFoundError(
            f"Processed data not found at {path}. Run the ingestion step first."
        )
    return pd.read_csv(path)