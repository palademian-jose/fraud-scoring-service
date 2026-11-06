"""Dataset loading, the stratified sample and the time-ordered split."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

TARGET = "Class"
TIME = "Time"
FEATURES = [f"V{i}" for i in range(1, 29)] + ["Amount"]
"""Model features. `Time` is excluded: it is an elapsed-seconds counter, so using it as a
feature would only let the model memorise the training window (a leakage trap)."""
ALL_COLUMNS = [TIME, *FEATURES, TARGET]


def make_sample(
    df: pd.DataFrame, rows: int = 5000, fraud_rate: float = 0.05, seed: int = 42
) -> pd.DataFrame:
    """Class-stratified sample with a fixed, enriched fraud rate, sorted by Time.

    A proportional 5,000-row draw would hold only ~9 frauds, too few to train or evaluate
    anything, so the sample fixes the fraud share at `fraud_rate` (5%) and draws each class
    independently. Metrics measured on it are not comparable with full-data metrics.
    Every row keeps its original Time so the time-ordered split still works.
    """
    n_pos = min(round(rows * fraud_rate), int(df[TARGET].sum()))
    pos = df[df[TARGET] == 1].sample(n=n_pos, random_state=seed)
    neg = df[df[TARGET] == 0].sample(n=rows - n_pos, random_state=seed)
    out = pd.concat([pos, neg]).sort_values(TIME, kind="stable").reset_index(drop=True)
    return out[ALL_COLUMNS]


def load_frame(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    missing = set(ALL_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    df = df[ALL_COLUMNS].copy()
    df[TARGET] = df[TARGET].astype(int)
    return df.sort_values(TIME, kind="stable").reset_index(drop=True)


@dataclass(frozen=True)
class Split:
    train: pd.DataFrame
    valid: pd.DataFrame
    test: pd.DataFrame

    def sizes(self) -> dict[str, int]:
        return {"train": len(self.train), "valid": len(self.valid), "test": len(self.test)}


def time_split(df: pd.DataFrame, train: float = 0.6, valid: float = 0.2) -> Split:
    """Split by position after sorting on Time: 60/20/20 by default, no shuffling.

    Boundaries fall on row counts, then are nudged so no timestamp is shared across two splits
    (rows with an identical Time must stay together).
    """
    if not 0 < train < 1 or not 0 < valid < 1 or train + valid >= 1:
        raise ValueError("train and valid fractions must leave room for a test split")
    df = df.sort_values(TIME, kind="stable").reset_index(drop=True)
    n = len(df)
    t = df[TIME].to_numpy()

    def boundary(i: int) -> int:
        while 0 < i < n and t[i] == t[i - 1]:
            i += 1
        return i

    i1 = boundary(int(n * train))
    i2 = boundary(max(i1 + 1, int(n * (train + valid))))
    return Split(df.iloc[:i1].copy(), df.iloc[i1:i2].copy(), df.iloc[i2:].copy())
