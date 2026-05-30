import argparse
import logging
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.data.ingestion import (
    create_sample_dataset,
    load_isot_dataset,
    load_processed_data,
    save_processed_data,
)
from src.models.trainer import run_training

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train FakeGuard ML models")
    parser.add_argument("--fake", default="data/raw/Fake.csv", help="Path to Fake.csv")
    parser.add_argument("--true", default="data/raw/True.csv", help="Path to True.csv")
    parser.add_argument("--sample", action="store_true", help="Use built-in sample data")
    parser.add_argument("--limit", type=int, default=None, help="Limit dataset size for fast testing")
    args = parser.parse_args()

    if args.sample:
        log.info("Using built-in sample dataset (8 articles — smoke test only).")
        df = create_sample_dataset()
        save_processed_data(df)

    else:
        fake_path, true_path = Path(args.fake), Path(args.true)
        if fake_path.exists() and true_path.exists():
            log.info("Loading ISOT dataset from %s + %s", fake_path, true_path)
            df = load_isot_dataset(fake_path, true_path)

            if args.limit and args.limit < len(df):
                log.info("Limiting dataset down to %d rows for speed.", args.limit)
                df = df.head(args.limit)

            save_processed_data(df)
        else:
            log.warning("CSV files not found at %s / %s.", fake_path, true_path)
            try:
                df = load_processed_data()
                log.info("Using existing processed data (%d rows).", len(df))
            except FileNotFoundError:
                log.info("Falling back to built-in sample dataset.")
                df = create_sample_dataset()
                save_processed_data(df)

    log.info("Training on %d articles…", len(df))
    metrics = run_training()

    print("\n" + "=" * 65)
    print(f"{'MODEL':<28} {'ACC':>6} {'F1':>7} {'AUC':>7}")
    print("=" * 65)
    for model, m in metrics.items():
        print(f"{model:<28} {m['accuracy']:>6.4f} {m['f1_score']:>7.4f} {m['roc_auc']:>7.4f}")
    print("=" * 65)

    best = max(metrics, key=lambda k: metrics[k]["f1_score"])
    print(f"\n✅ Best model: {best} (F1 = {metrics[best]['f1_score']:.4f})\n")


if __name__ == "__main__":
    main()
