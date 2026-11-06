from __future__ import annotations

import numpy as np

from fraudscore.data import FEATURES, TARGET, Split
from fraudscore.explain import global_importance, reason_codes, shap_matrix
from fraudscore.models import ModelBundle


def test_shap_values_sum_to_margin(champion: ModelBundle, split: Split) -> None:
    rows = split.test.head(50)
    values = shap_matrix(champion, rows)
    base = float(np.ravel(champion._explainer.expected_value)[-1])
    np.testing.assert_allclose(values.sum(axis=1) + base, champion.margin(rows), atol=1e-4)


def test_reason_codes_top_five_sorted(champion: ModelBundle, split: Split) -> None:
    rows = split.test[split.test[TARGET] == 1].head(4)
    codes = reason_codes(champion, rows)
    assert len(codes) == 4
    for row in codes:
        assert len(row) == 5
        mags = [abs(c["contribution"]) for c in row]
        assert mags == sorted(mags, reverse=True)
        assert all(c["feature"] in FEATURES for c in row)
        assert {c["direction"] for c in row} <= {"increases_risk", "decreases_risk"}


def test_flagged_fraud_has_risk_increasing_top_reason(champion: ModelBundle, split: Split) -> None:
    fraud = split.test[split.test[TARGET] == 1]
    top = champion.proba(fraud).argmax()
    first = reason_codes(champion, fraud.iloc[[top]])[0][0]
    assert first["direction"] == "increases_risk"


def test_global_importance_sorted(champion: ModelBundle) -> None:
    imp = global_importance(champion)
    assert list(imp.columns) == ["feature", "mean_abs_shap"]
    assert imp["mean_abs_shap"].is_monotonic_decreasing
    assert len(imp) == len(FEATURES)
