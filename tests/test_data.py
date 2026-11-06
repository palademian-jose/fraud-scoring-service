from __future__ import annotations

import pandas as pd
import pytest

from fraudscore.data import ALL_COLUMNS, FEATURES, TARGET, TIME, make_sample, time_split


def test_sample_shape_and_columns(frame: pd.DataFrame) -> None:
    assert len(frame) == 5000
    assert list(frame.columns) == ALL_COLUMNS
    assert frame[TARGET].sum() == 250
    assert frame[TIME].is_monotonic_increasing


def test_time_not_a_feature() -> None:
    assert TIME not in FEATURES and TARGET not in FEATURES


def test_split_is_chronological_and_disjoint(frame: pd.DataFrame) -> None:
    s = time_split(frame)
    assert s.train[TIME].max() < s.valid[TIME].min()
    assert s.valid[TIME].max() < s.test[TIME].min()
    assert sum(s.sizes().values()) == len(frame)
    assert s.sizes() == {"train": 3000, "valid": 1000, "test": 1000}


def test_split_keeps_equal_timestamps_together() -> None:
    df = pd.DataFrame({TIME: [1, 1, 2, 2, 3, 3, 4, 4, 5, 5], TARGET: [0, 1] * 5})
    s = time_split(df)
    assert set(s.train[TIME]).isdisjoint(s.valid[TIME])
    assert set(s.valid[TIME]).isdisjoint(s.test[TIME])


def test_split_rejects_bad_fractions(frame: pd.DataFrame) -> None:
    with pytest.raises(ValueError):
        time_split(frame, train=0.8, valid=0.3)


def test_make_sample_is_deterministic(frame: pd.DataFrame) -> None:
    a = make_sample(frame, rows=1000, fraud_rate=0.1, seed=1)
    b = make_sample(frame, rows=1000, fraud_rate=0.1, seed=1)
    pd.testing.assert_frame_equal(a, b)
    assert a[TARGET].sum() == 100
