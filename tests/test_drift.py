from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from fraudscore.data import FEATURES, Split
from fraudscore.drift import (
    MonitoringConfig,
    detect_drift,
    load_monitoring_config,
    psi,
    render_drift_report,
)
from fraudscore.models import ModelBundle
from fraudscore.settings import ROOT
from fraudscore.synth import make_stream


def test_psi_zero_for_same_distribution_and_large_for_shift() -> None:
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, 20_000)
    assert psi(a, rng.normal(0, 1, 20_000)) < 0.02
    assert psi(a, rng.normal(1.5, 1, 20_000)) > 0.25


def test_psi_handles_constant_reference() -> None:
    assert psi(np.zeros(100), np.zeros(100)) == pytest.approx(0.0)


def test_monitoring_yaml_loads() -> None:
    cfg = load_monitoring_config(ROOT / "monitoring.yaml")
    assert cfg.psi.alert > cfg.psi.warn and cfg.min_rows >= 1


def test_no_shift_is_not_an_alert(champion: ModelBundle, split: Split) -> None:
    stream = make_stream(split.train, 2000, shift=False)
    res = detect_drift(
        champion.reference,
        stream,
        FEATURES,
        MonitoringConfig(),
        reference_scores=champion.reference_scores,
        current_scores=champion.proba(stream),
    )
    assert res.verdict != "ALERT"


def test_shifted_stream_raises_alert(champion: ModelBundle, split: Split) -> None:
    stream = make_stream(split.train, 2000, shift=True)
    res = detect_drift(
        champion.reference,
        stream,
        FEATURES,
        MonitoringConfig(),
        reference_scores=champion.reference_scores,
        current_scores=champion.proba(stream),
        when=date(2026, 11, 7),
    )
    assert res.verdict == "ALERT"
    drifted = {f.name for f in res.features if f.status == "ALERT"}
    assert {"V14", "V4", "Amount"} <= drifted
    report = render_drift_report(res, "stream_shifted.parquet")
    assert "# Drift report 2026-11-07" in report and "Verdict: ALERT" in report


def test_small_batch_rejected(champion: ModelBundle, frame: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="need at least"):
        detect_drift(champion.reference, frame.head(10), FEATURES, MonitoringConfig())


def test_stream_is_deterministic_and_labelled(frame: pd.DataFrame) -> None:
    a = make_stream(frame, 500, seed=3)
    b = make_stream(frame, 500, seed=3)
    pd.testing.assert_frame_equal(a, b)
    assert set(a["Class"].unique()) <= {0, 1}
