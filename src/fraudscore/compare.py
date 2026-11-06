"""Champion vs challenger on the test split, written as eval-result JSON for `evalkit gate`."""

from __future__ import annotations

import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from fraudscore.data import TARGET, Split
from fraudscore.metrics import ranking_metrics
from fraudscore.models import ModelBundle
from fraudscore.settings import ROOT, Settings
from fraudscore.threshold import cost_at

WINDOWS = 10


def git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def _window_cases(bundle: ModelBundle, test: pd.DataFrame, settings: Settings) -> list[dict]:
    """Ten consecutive time windows of the test split, so a gate can see where a model fails."""
    cases = []
    for i, idx in enumerate(np.array_split(np.arange(len(test)), WINDOWS)):
        part = test.iloc[idx]
        y = part[TARGET].to_numpy()
        c = cost_at(y, bundle.proba(part), bundle.threshold, settings.cost_fp, settings.cost_fn)
        recall = c.tp / (c.tp + c.fn) if (c.tp + c.fn) else None
        scores: dict[str, float] = {"cost_per_10k": c.cost_per_10k, "frauds": float(y.sum())}
        if recall is not None:
            scores["recall"] = recall
        cases.append(
            {
                "id": f"window-{i:02d}",
                "scores": scores,
                "passed": recall is None or recall >= 0.5,
                "latency_ms": 0.0,
                "cost_usd": 0.0,
            }
        )
    return cases


def eval_result(
    role: str, bundle: ModelBundle, split: Split, settings: Settings, data_name: str
) -> dict[str, Any]:
    test = split.test
    y = test[TARGET].to_numpy()
    margin = bundle.margin(test)
    proba = bundle.calibrator.predict(margin)
    m = ranking_metrics(y, margin, proba=proba, ci_resamples=0, seed=settings.seed)
    cost = cost_at(y, proba, bundle.threshold, settings.cost_fp, settings.cost_fn)
    metrics = {
        "pr_auc": m.pr_auc,
        "roc_auc": m.roc_auc,
        "precision_at_1pct": m.precision_at_1pct,
        "recall_at_1pct": m.recall_at_1pct,
        "precision_at_01pct": m.precision_at_01pct,
        "recall_at_01pct": m.recall_at_01pct,
        "brier": m.brier,
        "cost_per_10k": cost.cost_per_10k,
        "alert_rate": cost.alert_rate,
    }
    return {
        "run_id": f"{role}-{bundle.version}",
        "commit": git_commit(),
        "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
        "provider": "none",
        "model": bundle.version,
        "metrics": {k: float(v) for k, v in metrics.items() if v is not None},
        "cases": _window_cases(bundle, test, settings),
        "meta": {
            "role": role,
            "data": data_name,
            "split": "test",
            "n_test": len(test),
            "n_fraud_test": int(y.sum()),
            "threshold": bundle.threshold,
            "cost_fp": settings.cost_fp,
            "cost_fn": settings.cost_fn,
        },
    }


def write_eval(result: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


def promotion_gate(champion_json: Path, challenger_json: Path, config: Path) -> tuple[bool, str]:
    """Run the evalkit gate with the champion as baseline and the challenger as candidate."""
    from evalkit.gate import evaluate_gate, load_gate_config
    from evalkit.io import load_eval_result

    report = evaluate_gate(
        load_eval_result(champion_json), load_eval_result(challenger_json), load_gate_config(config)
    )
    return report.ok, report.to_markdown()
