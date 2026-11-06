"""Drift monitoring: two-sample KS and PSI per feature and on the score distribution."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import yaml
from pydantic import BaseModel, Field
from scipy.stats import ks_2samp

Status = Literal["OK", "WARN", "ALERT"]


class KSConfig(BaseModel):
    pvalue: float = 0.01
    statistic: float = 0.10


class PSIConfig(BaseModel):
    bins: int = Field(default=10, ge=2)
    warn: float = 0.10
    alert: float = 0.25


class MonitoringConfig(BaseModel):
    ks: KSConfig = KSConfig()
    psi: PSIConfig = PSIConfig()
    min_features_alert: int = Field(default=3, ge=1)
    min_rows: int = Field(default=200, ge=1)


def load_monitoring_config(path: Path) -> MonitoringConfig:
    return MonitoringConfig.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")) or {})


def psi(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    """Population stability index with quantile bins taken from the reference.

    Bin edges are the reference quantiles (outer edges open), so each reference bin holds about
    1/bins of the data. Empty bins are floored at 1e-4 to keep the logarithm finite.
    """
    ref = np.asarray(reference, dtype=float)
    cur = np.asarray(current, dtype=float)
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)[1:-1]))
    r = np.bincount(np.searchsorted(edges, ref, side="right"), minlength=len(edges) + 1)
    c = np.bincount(np.searchsorted(edges, cur, side="right"), minlength=len(edges) + 1)
    rp = np.clip(r / r.sum(), 1e-4, None)
    cp = np.clip(c / c.sum(), 1e-4, None)
    return float(np.sum((cp - rp) * np.log(cp / rp)))


@dataclass
class FeatureDrift:
    name: str
    ks_stat: float
    ks_p: float
    psi: float
    status: Status
    mean_ref: float
    mean_cur: float


@dataclass
class DriftResult:
    date: str
    n_reference: int
    n_current: int
    features: list[FeatureDrift] = field(default_factory=list)
    score: FeatureDrift | None = None
    verdict: Status = "OK"
    config: MonitoringConfig = field(default_factory=MonitoringConfig)

    @property
    def n_alert(self) -> int:
        return sum(f.status == "ALERT" for f in self.features)

    @property
    def n_warn(self) -> int:
        return sum(f.status == "WARN" for f in self.features)


def _one(name: str, ref: np.ndarray, cur: np.ndarray, cfg: MonitoringConfig) -> FeatureDrift:
    ks = ks_2samp(ref, cur)
    p = psi(ref, cur, cfg.psi.bins)
    ks_hit = ks.pvalue < cfg.ks.pvalue and ks.statistic >= cfg.ks.statistic
    if ks_hit or p >= cfg.psi.alert:
        status: Status = "ALERT"
    elif p >= cfg.psi.warn:
        status = "WARN"
    else:
        status = "OK"
    return FeatureDrift(
        name=name,
        ks_stat=float(ks.statistic),
        ks_p=float(ks.pvalue),
        psi=p,
        status=status,
        mean_ref=float(np.mean(ref)),
        mean_cur=float(np.mean(cur)),
    )


def detect_drift(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    features: list[str],
    cfg: MonitoringConfig,
    *,
    reference_scores: np.ndarray | None = None,
    current_scores: np.ndarray | None = None,
    when: date | None = None,
) -> DriftResult:
    if len(current) < cfg.min_rows:
        raise ValueError(f"batch has {len(current)} rows; need at least {cfg.min_rows}")
    res = DriftResult(
        date=(when or date.today()).isoformat(),
        n_reference=len(reference),
        n_current=len(current),
        config=cfg,
    )
    for f in features:
        res.features.append(_one(f, reference[f].to_numpy(), current[f].to_numpy(), cfg))
    res.features.sort(key=lambda d: (-d.psi, d.name))
    if reference_scores is not None and current_scores is not None:
        res.score = _one("score", reference_scores, current_scores, cfg)
    score_alert = res.score is not None and res.score.status == "ALERT"
    if score_alert or res.n_alert >= cfg.min_features_alert:
        res.verdict = "ALERT"
    elif res.n_alert or res.n_warn or (res.score and res.score.status == "WARN"):
        res.verdict = "WARN"
    return res


def render_drift_report(res: DriftResult, source: str) -> str:
    c = res.config
    lines = [
        f"# Drift report {res.date}",
        "",
        f"**Verdict: {res.verdict}**",
        "",
        f"- Batch: `{source}`, {res.n_current:,} rows against a training reference of "
        f"{res.n_reference:,} rows.",
        f"- Features: {res.n_alert} ALERT, {res.n_warn} WARN, "
        f"{len(res.features) - res.n_alert - res.n_warn} OK.",
        f"- Rule: ALERT when KS p < {c.ks.pvalue} and KS statistic >= {c.ks.statistic}, "
        f"or PSI >= {c.psi.alert}; WARN when PSI >= {c.psi.warn}. Batch verdict is ALERT "
        f"when the score distribution alerts or {c.min_features_alert} or more features alert "
        "(thresholds in `monitoring.yaml`).",
        "",
        "| Series | KS stat | KS p-value | PSI | Mean ref | Mean batch | Status |",
        "|---|---|---|---|---|---|---|",
    ]
    rows = ([res.score] if res.score else []) + [f for f in res.features if f.status != "OK"]
    rows += [f for f in res.features if f.status == "OK"][:3]
    for f in rows:
        lines.append(
            f"| {f.name} | {f.ks_stat:.3f} | {f.ks_p:.2e} | {f.psi:.3f} | "
            f"{f.mean_ref:.3f} | {f.mean_cur:.3f} | {f.status} |"
        )
    lines += ["", "Features are sorted by PSI. Only the first three OK features are listed.", ""]
    return "\n".join(lines)
