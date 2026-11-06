"""Labelled synthetic transaction stream for drift demos.

The stream is a seeded bootstrap of a reference frame with small Gaussian jitter, so labels and
feature correlations resemble the reference. `shift=True` then moves a few features and
rescales Amount, the way a new merchant mix or a changed upstream feature pipeline would.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from fraudscore.data import ALL_COLUMNS, FEATURES, TARGET, TIME

SHIFTS = {"V4": 1.5, "V10": -1.2, "V14": -1.5, "V17": 1.0}  # in units of reference std
AMOUNT_SCALE = 2.5


def make_stream(
    reference: pd.DataFrame, rows: int, *, shift: bool = False, seed: int = 7
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(reference), rows)
    out = reference.iloc[idx][ALL_COLUMNS].reset_index(drop=True).copy()
    std = reference[FEATURES].std().to_numpy()
    out[FEATURES] = out[FEATURES].to_numpy() + rng.normal(0, 0.05, (rows, len(FEATURES))) * std
    out["Amount"] = out["Amount"].clip(lower=0)
    if shift:
        for name, units in SHIFTS.items():
            out[name] = out[name] + units * reference[name].std()
        out["Amount"] = out["Amount"] * AMOUNT_SCALE
    out[TIME] = np.arange(rows)
    out[TARGET] = out[TARGET].astype(int)
    return out
