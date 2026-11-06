from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner

from fraudscore.cli import app
from fraudscore.data import FEATURES, Split
from fraudscore.settings import ROOT, Settings, get_settings
from fraudscore.synth import make_stream

runner = CliRunner()


@pytest.fixture(scope="module")
def trained(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("run")
    get_settings.cache_clear()
    res = runner.invoke(
        app,
        [
            "train",
            "--data",
            str(Settings().sample_path),
            "--models-dir",
            str(root / "models"),
            "--reports-dir",
            str(root / "reports"),
        ],
    )
    assert res.exit_code == 0, res.output
    return root


def test_train_outputs(trained: Path) -> None:
    for name in ("baseline", "champion", "challenger"):
        assert (trained / "models" / f"{name}.joblib").exists()
    metrics = json.loads((trained / "reports" / "metrics.json").read_text())
    assert metrics["models"]["champion"]["test"]["pr_auc_ci_low"] is not None
    assert metrics["data"]["split"]["train"]["rows"] == 3000


def test_score_file(trained: Path, frame: pd.DataFrame) -> None:
    src = trained / "in.parquet"
    frame.head(40).to_parquet(src)
    out = trained / "out.csv"
    res = runner.invoke(
        app,
        [
            "score-file",
            str(src),
            "--output",
            str(out),
            "--model",
            str(trained / "models/champion.joblib"),
        ],
    )
    assert res.exit_code == 0, res.output
    scored = pd.read_csv(out)
    assert {"score", "decision", "reasons", "model_version"} <= set(scored.columns)
    assert len(json.loads(scored["reasons"][0])) == 5


def test_score_file_rejects_missing_columns(trained: Path) -> None:
    src = trained / "bad.csv"
    pd.DataFrame({"V1": [0.1]}).to_csv(src, index=False)
    res = runner.invoke(
        app,
        [
            "score-file",
            str(src),
            "--output",
            str(trained / "x.csv"),
            "--model",
            str(trained / "models/champion.joblib"),
        ],
    )
    assert res.exit_code != 0


def test_drift_command_writes_dated_report(trained: Path, split: Split) -> None:
    batch = trained / "shifted.parquet"
    make_stream(split.train, 1500, shift=True).to_parquet(batch)
    res = runner.invoke(
        app,
        [
            "drift",
            str(batch),
            "--model",
            str(trained / "models/champion.joblib"),
            "--config",
            str(ROOT / "monitoring.yaml"),
            "--out-dir",
            str(trained / "reports"),
            "--fail-on-alert",
        ],
    )
    assert res.exit_code == 2, res.output
    reports = list((trained / "reports").glob("drift-*.md"))
    assert len(reports) == 1 and "ALERT" in reports[0].read_text()


def test_compare_writes_eval_json_and_verdict(trained: Path) -> None:
    out = trained / "cmp"
    res = runner.invoke(
        app,
        [
            "compare",
            "--champion",
            str(trained / "models/champion.joblib"),
            "--challenger",
            str(trained / "models/challenger.joblib"),
            "--data",
            str(Settings().sample_path),
            "--out-dir",
            str(out),
            "--config",
            str(ROOT / "evals.yaml"),
        ],
    )
    assert res.exit_code == 0, res.output
    result = json.loads((out / "eval-champion.json").read_text())
    assert {"run_id", "commit", "timestamp", "provider", "model", "metrics", "cases"} <= set(result)
    assert result["metrics"]["pr_auc"] > 0.5 and len(result["cases"]) == 10
    assert "Decision:" in (out / "compare.md").read_text()
    from evalkit.io import load_eval_result

    assert load_eval_result(out / "eval-challenger.json").metrics["pr_auc"] > 0


def test_evaluate_prints_json(trained: Path) -> None:
    res = runner.invoke(
        app,
        [
            "evaluate",
            "--model",
            str(trained / "models/champion.joblib"),
            "--data",
            str(Settings().sample_path),
        ],
    )
    assert res.exit_code == 0, res.output
    assert json.loads(res.output)["test"]["pr_auc"] > 0.5
    assert FEATURES  # features exported
