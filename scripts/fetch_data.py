"""Download OpenML creditcard (data id 1597) to data/raw/ and write the committed 5k sample."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.datasets import fetch_openml

from fraudscore.data import make_sample

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw", type=Path, default=ROOT / "data" / "raw" / "creditcard.parquet")
    p.add_argument(
        "--sample", type=Path, default=ROOT / "data" / "sample" / "creditcard_sample.parquet"
    )
    p.add_argument("--rows", type=int, default=5000)
    args = p.parse_args()

    args.raw.parent.mkdir(parents=True, exist_ok=True)
    if not args.raw.exists():
        bunch = fetch_openml(data_id=1597, as_frame=True, data_home=str(args.raw.parent / "openml"))
        df = bunch.frame.copy()
        df["Class"] = df["Class"].astype(int)
        if "Time" not in df.columns:
            # OpenML 1597 drops the Kaggle `Time` column but keeps the rows in their original
            # chronological order, so the row position is the ordinal time used for the split.
            df.insert(0, "Time", range(len(df)))
        df.to_parquet(args.raw, index=False)
    df = pd.read_parquet(args.raw)
    print(f"raw rows={len(df)} frauds={int(df['Class'].sum())}")
    sample = make_sample(df, args.rows)
    args.sample.parent.mkdir(parents=True, exist_ok=True)
    sample.to_parquet(args.sample, index=False)
    print(f"sample rows={len(sample)} frauds={int(sample['Class'].sum())} -> {args.sample}")


if __name__ == "__main__":
    main()
