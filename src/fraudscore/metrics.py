"""Ranking, alert-queue and calibration metrics."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


def precision_recall_at_fraction(
    y: np.ndarray, score: np.ndarray, fraction: float
) -> tuple[float, float]:
    """Precision and recall when only the top `fraction` of scores raises an alert."""
    n = len(y)
    k = max(1, math.ceil(n * fraction))
    top = np.argsort(-np.asarray(score), kind="stable")[:k]
    hits = float(np.asarray(y)[top].sum())
    total = float(np.sum(y))
    return hits / k, (hits / total if total else 0.0)


def bootstrap_ci(
    y: np.ndarray,
    score: np.ndarray,
    *,
    resamples: int = 1000,
    seed: int = 42,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Percentile bootstrap CI on PR-AUC. Resamples with no positive are skipped."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    score = np.asarray(score)
    n = len(y)
    vals: list[float] = []
    for _ in range(resamples):
        idx = rng.integers(0, n, n)
        yb = y[idx]
        if yb.sum() == 0:
            continue
        vals.append(average_precision_score(yb, score[idx]))
    lo, hi = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


@dataclass
class RankingMetrics:
    n: int
    n_fraud: int
    pr_auc: float
    roc_auc: float
    precision_at_1pct: float
    recall_at_1pct: float
    precision_at_01pct: float
    recall_at_01pct: float
    brier: float | None = None
    pr_auc_ci_low: float | None = None
    pr_auc_ci_high: float | None = None

    def to_dict(self) -> dict[str, float | int | None]:
        return asdict(self)


def ranking_metrics(
    y: np.ndarray,
    score: np.ndarray,
    *,
    proba: np.ndarray | None = None,
    ci_resamples: int = 0,
    seed: int = 42,
) -> RankingMetrics:
    """PR-AUC is the primary metric; ROC-AUC is reported but never optimised.

    `proba` is a calibrated probability used for the Brier score only.
    """
    y = np.asarray(y)
    p1, r1 = precision_recall_at_fraction(y, score, 0.01)
    p01, r01 = precision_recall_at_fraction(y, score, 0.001)
    m = RankingMetrics(
        n=len(y),
        n_fraud=int(y.sum()),
        pr_auc=float(average_precision_score(y, score)),
        roc_auc=float(roc_auc_score(y, score)),
        precision_at_1pct=p1,
        recall_at_1pct=r1,
        precision_at_01pct=p01,
        recall_at_01pct=r01,
    )
    if proba is not None:
        m.brier = float(brier_score_loss(y, proba))
    if ci_resamples:
        m.pr_auc_ci_low, m.pr_auc_ci_high = bootstrap_ci(
            y, score, resamples=ci_resamples, seed=seed
        )
    return m
