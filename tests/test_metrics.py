from __future__ import annotations

import numpy as np
import pytest

from fraudscore.metrics import bootstrap_ci, precision_recall_at_fraction, ranking_metrics
from fraudscore.threshold import (
    PlattCalibrator,
    choose_threshold,
    cost_at,
    reliability_curve,
)


def test_precision_recall_at_fraction() -> None:
    y = np.array([1, 0, 1, 0, 0, 0, 0, 0, 0, 0])
    score = np.array([0.9, 0.8, 0.7, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    p, r = precision_recall_at_fraction(y, score, 0.2)
    assert p == 0.5 and r == 0.5


def test_perfect_ranking() -> None:
    y = np.array([0, 0, 0, 1, 1] * 20)
    m = ranking_metrics(y, y.astype(float))
    assert m.pr_auc == 1.0 and m.roc_auc == 1.0


def test_bootstrap_ci_is_deterministic_and_brackets_point() -> None:
    rng = np.random.default_rng(0)
    y = (rng.random(2000) < 0.05).astype(int)
    s = y * 0.5 + rng.random(2000)
    a = bootstrap_ci(y, s, resamples=100, seed=3)
    b = bootstrap_ci(y, s, resamples=100, seed=3)
    assert a == b and a[0] < a[1]
    point = ranking_metrics(y, s).pr_auc
    assert a[0] <= point <= a[1]


def test_threshold_minimises_cost() -> None:
    y = np.array([1, 1, 0, 0, 0, 0])
    p = np.array([0.9, 0.6, 0.55, 0.3, 0.2, 0.1])
    t = choose_threshold(y, p, cost_fp=1, cost_fn=10)
    assert cost_at(y, p, t, 1, 10).fn == 0
    best = min(cost_at(y, p, c, 1, 10).total_cost for c in [*p, 1.1])
    assert cost_at(y, p, t, 1, 10).total_cost == best


def test_threshold_never_alerts_when_fn_is_free() -> None:
    y = np.array([1, 0, 0, 0])
    p = np.array([0.4, 0.5, 0.3, 0.2])
    t = choose_threshold(y, p, cost_fp=5, cost_fn=0)
    assert cost_at(y, p, t, 5, 0).fp == 0


def test_higher_fn_cost_lowers_threshold() -> None:
    rng = np.random.default_rng(1)
    y = (rng.random(5000) < 0.05).astype(int)
    p = np.clip(y * 0.4 + rng.random(5000) * 0.6, 0, 1)
    assert choose_threshold(y, p, 1, 100) <= choose_threshold(y, p, 1, 5)


def test_platt_calibration_reduces_brier() -> None:
    from sklearn.metrics import brier_score_loss

    rng = np.random.default_rng(2)
    margin = rng.normal(0, 2, 4000)
    y = (rng.random(4000) < 1 / (1 + np.exp(-(margin * 0.5 - 3)))).astype(int)
    raw = 1 / (1 + np.exp(-margin))
    cal = PlattCalibrator().fit(margin, y).predict(margin)
    assert brier_score_loss(y, cal) < brier_score_loss(y, raw)


def test_reliability_curve_shapes() -> None:
    rng = np.random.default_rng(3)
    p = rng.random(1000)
    y = (rng.random(1000) < p).astype(int)
    mp, obs, cnt = reliability_curve(y, p, bins=5)
    assert len(mp) == len(obs) == len(cnt) <= 5
    assert cnt.sum() == 1000
    assert pytest.approx(obs, abs=0.15) == mp
