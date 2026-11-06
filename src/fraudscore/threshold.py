"""Cost-weighted alert threshold and probability calibration."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression


@dataclass(frozen=True)
class CostReport:
    threshold: float
    cost_fp: float
    cost_fn: float
    tp: int
    fp: int
    fn: int
    tn: int
    total_cost: float
    cost_per_10k: float
    alert_rate: float

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def cost_at(
    y: np.ndarray, proba: np.ndarray, threshold: float, cost_fp: float, cost_fn: float
) -> CostReport:
    y = np.asarray(y).astype(bool)
    alert = np.asarray(proba) >= threshold
    tp = int((alert & y).sum())
    fp = int((alert & ~y).sum())
    fn = int((~alert & y).sum())
    tn = int((~alert & ~y).sum())
    total = fp * cost_fp + fn * cost_fn
    n = len(y)
    return CostReport(
        threshold=float(threshold),
        cost_fp=cost_fp,
        cost_fn=cost_fn,
        tp=tp,
        fp=fp,
        fn=fn,
        tn=tn,
        total_cost=float(total),
        cost_per_10k=float(total / n * 10_000) if n else 0.0,
        alert_rate=float(alert.mean()) if n else 0.0,
    )


def choose_threshold(y: np.ndarray, proba: np.ndarray, cost_fp: float, cost_fn: float) -> float:
    """Threshold minimising `fp * cost_fp + fn * cost_fn` on the given (validation) data.

    Candidates are the distinct predicted probabilities (alert when proba >= t). When even the
    best cut costs more than never alerting, the result is just above the maximum score. Ties
    on cost go to the higher threshold, which raises fewer alerts.
    """
    y = np.asarray(y).astype(bool)
    proba = np.asarray(proba)
    order = np.argsort(-proba, kind="stable")
    ys = y[order]
    ps = proba[order]
    tp = np.cumsum(ys)
    fp = np.cumsum(~ys)
    fn = ys.sum() - tp
    cost = fp * cost_fp + fn * cost_fn
    # Cut only between distinct scores: index i means "alert on the first i+1 rows".
    idx = np.flatnonzero(np.r_[ps[1:] != ps[:-1], True])
    # Walk from the highest threshold (fewest alerts) so argmin keeps the first minimum.
    rev = idx[::-1]
    best_i = int(rev[int(np.argmin(cost[rev]))])
    if ys.sum() * cost_fn <= cost[best_i]:
        return float(np.nextafter(ps[0], np.inf))
    return float(ps[best_i])


class PlattCalibrator:
    """Sigmoid calibration of a raw model margin (log-odds), fitted on the validation split."""

    def __init__(self) -> None:
        self._lr = LogisticRegression(C=1e6, max_iter=1000)

    def fit(self, margin: np.ndarray, y: np.ndarray) -> PlattCalibrator:
        self._lr.fit(np.asarray(margin, dtype=float).reshape(-1, 1), np.asarray(y).astype(int))
        return self

    def predict(self, margin: np.ndarray) -> np.ndarray:
        return self._lr.predict_proba(np.asarray(margin, dtype=float).reshape(-1, 1))[:, 1]


def reliability_curve(
    y: np.ndarray, proba: np.ndarray, bins: int = 10
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Quantile-binned reliability: (mean predicted, observed rate, count) per bin.

    Quantile bins are used because fraud scores are heavily skewed toward zero.
    """
    y = np.asarray(y)
    proba = np.asarray(proba)
    edges = np.unique(np.quantile(proba, np.linspace(0, 1, bins + 1)))
    ids = np.clip(np.searchsorted(edges, proba, side="right") - 1, 0, len(edges) - 2)
    mp, obs, cnt = [], [], []
    for b in range(len(edges) - 1):
        m = ids == b
        if m.any():
            mp.append(proba[m].mean())
            obs.append(y[m].mean())
            cnt.append(int(m.sum()))
    return np.array(mp), np.array(obs), np.array(cnt)
