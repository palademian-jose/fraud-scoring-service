from __future__ import annotations

import json
from pathlib import Path

import pytest

from fraudscore.results import END, START, inject, render_results
from fraudscore.settings import Settings
from fraudscore.train import run_training


def test_results_table_numbers_come_from_metrics(tmp_path: Path, settings: Settings) -> None:
    out = run_training(settings.sample_path, tmp_path / "m", tmp_path / "r", settings, plots=False)
    bench = {"with_reason_codes": {"p95_ms": 12.3}}
    table = render_results(json.loads((tmp_path / "r" / "metrics.json").read_text()), bench)
    champ = out["models"]["champion"]["test"]
    assert f"{champ['pr_auc']:.3f} [" in table
    assert f"{champ['recall_at_1pct']:.3f}" in table
    assert "12.3 ms" in table
    assert table.count("\n| ") >= 11


def test_inject_replaces_only_marked_block() -> None:
    readme = f"before\n{START}\nold\n{END}\nafter\n"
    new = inject(readme, "NEW\n")
    assert new == f"before\n{START}\nNEW\n{END}\nafter\n"
    with pytest.raises(ValueError):
        inject("no markers", "x")
