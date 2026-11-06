"""End-to-end training: split, baseline, champion, challenger, metrics and figures."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve

from fraudscore.data import TARGET, Split, load_frame, time_split
from fraudscore.explain import global_importance, reason_codes
from fraudscore.metrics import ranking_metrics
from fraudscore.models import ModelBundle, fit_bundle, save_bundle
from fraudscore.plots import pr_curve_plot, reliability_plot, shap_bar
from fraudscore.settings import Settings
from fraudscore.threshold import cost_at, reliability_curve

ROLES = ("baseline", "champion", "challenger")


def describe_split(split: Split) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, part in (("train", split.train), ("valid", split.valid), ("test", split.test)):
        out[name] = {
            "rows": len(part),
            "frauds": int(part[TARGET].sum()),
            "fraud_rate": float(part[TARGET].mean()),
            "time_min": int(part["Time"].min()),
            "time_max": int(part["Time"].max()),
        }
    return out


def evaluate_bundle(
    bundle: ModelBundle, split: Split, settings: Settings, *, ci: bool
) -> dict[str, Any]:
    """Metrics by split, plus expected cost on validation and test at the chosen threshold."""
    result: dict[str, Any] = {
        "version": bundle.version,
        "threshold": bundle.threshold,
        "trained_on": bundle.trained_on,
    }
    for name, part in (("train", split.train), ("valid", split.valid), ("test", split.test)):
        y = part[TARGET].to_numpy()
        margin = bundle.margin(part)
        proba = bundle.calibrator.predict(margin)
        m = ranking_metrics(
            y,
            margin,
            proba=proba,
            ci_resamples=settings.bootstrap_resamples if (ci and name == "test") else 0,
            seed=settings.seed,
        )
        result[name] = m.to_dict()
        result[name]["cost"] = cost_at(
            y, proba, bundle.threshold, settings.cost_fp, settings.cost_fn
        ).to_dict()
    return result


def run_training(
    data_path: Path,
    models_dir: Path,
    reports_dir: Path,
    settings: Settings,
    *,
    plots: bool = True,
) -> dict[str, Any]:
    df = load_frame(data_path)
    split = time_split(df)
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    bundles: dict[str, ModelBundle] = {}
    results: dict[str, Any] = {}
    for role in ROLES:
        bundle, _ = fit_bundle(role, split, settings)
        bundles[role] = bundle
        results[role] = evaluate_bundle(bundle, split, settings, ci=True)
        save_bundle(bundle, models_dir / f"{role}.joblib")

    champion = bundles["champion"]
    out: dict[str, Any] = {
        "generated": datetime.now(UTC).strftime("%Y-%m-%d"),
        "data": {
            "file": data_path.name,
            "rows": len(df),
            "frauds": int(df[TARGET].sum()),
            "split": describe_split(split),
        },
        "assumptions": {
            "cost_fp": settings.cost_fp,
            "cost_fn": settings.cost_fn,
            "seed": settings.seed,
            "bootstrap_resamples": settings.bootstrap_resamples,
        },
        "models": results,
    }

    # Global importance from the 2,000-row background sample, and example reason codes.
    imp = global_importance(champion)
    out["shap_top10"] = imp.head(10).to_dict(orient="records")
    test = split.test
    frauds = test[test[TARGET] == 1].head(3)
    legit = test[test[TARGET] == 0].head(2)
    examples = pd.concat([frauds, legit])
    codes = reason_codes(champion, examples)
    out["reason_code_examples"] = [
        {
            "label": int(row[TARGET]),
            "score": float(champion.proba(examples.iloc[[i]])[0]),
            "reasons": codes[i],
        }
        for i, (_, row) in enumerate(examples.iterrows())
    ]

    if plots:
        shap_bar(imp, reports_dir / "shap_importance.png")
        y = test[TARGET].to_numpy()
        curves = {
            role: reliability_curve(y, bundles[role].proba(test))
            for role in ("baseline", "champion")
        }
        reliability_plot(curves, reports_dir / "calibration.png", "Reliability on the test split")
        pr = {}
        for role in ("baseline", "champion"):
            prec, rec, _ = precision_recall_curve(y, bundles[role].margin(test))
            pr[f"{role} (AP {results[role]['test']['pr_auc']:.3f})"] = (rec, prec)
        pr_curve_plot(pr, reports_dir / "pr_curve.png", "Precision-recall on the test split")

    (reports_dir / "metrics.json").write_text(
        json.dumps(
            out, indent=2, default=lambda o: o.item() if isinstance(o, np.generic) else str(o)
        )
        + "\n",
        encoding="utf-8",
    )
    return out


def load_split(data_path: Path) -> Split:
    return time_split(load_frame(data_path))
