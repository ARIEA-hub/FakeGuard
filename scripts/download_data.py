#!/usr/bin/env python3
"""
Download the ISOT Fake News Dataset from Kaggle.

Prerequisites
-------------
1. pip install kaggle
2. Place ~/.kaggle/kaggle.json (API credentials from kaggle.com/settings)

Dataset: https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset
"""
import logging
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

RAW_DIR = Path("data/raw")


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    log.info("Downloading ISOT Fake News Dataset from Kaggle…")
    result = subprocess.run(
        [
            "kaggle", "datasets", "download",
            "-d", "clmentbisaillon/fake-and-real-news-dataset",
            "--unzip", "-p", str(RAW_DIR),
        ],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        log.error("Kaggle CLI failed:\n%s", result.stderr)
        print(
            "\nManual download instructions:\n"
            "1. Visit: https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset\n"
            "2. Download Fake.csv and True.csv\n"
            f"3. Move them into: {RAW_DIR.resolve()}/\n"
        )
        sys.exit(1)

    for fname in ("Fake.csv", "True.csv"):
        p = RAW_DIR / fname
        if p.exists():
            log.info("✅ %s  (%.1f MB)", fname, p.stat().st_size / 1_048_576)
        else:
            log.error("❌ %s not found after download — check Kaggle response.", fname)


if __name__ == "__main__":
    main()