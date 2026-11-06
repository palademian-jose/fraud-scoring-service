from __future__ import annotations

import json
from pathlib import Path

from fraudscore.modelcard import render_model_card
from fraudscore.settings import Settings
from fraudscore.train import run_training


def test_model_card_from_a_real_run(tmp_path: Path, settings: Settings) -> None:
    out = run_training(settings.sample_path, tmp_path / "m", tmp_path / "r", settings)
    assert (tmp_path / "r" / "shap_importance.png").stat().st_size > 1000
    assert (tmp_path / "r" / "calibration.png").exists()
    metrics = json.loads((tmp_path / "r" / "metrics.json").read_text())
    card = render_model_card(metrics)
    for heading in ("## Model details", "## Intended use", "## Metrics by split", "## Limitations"):
        assert heading in card
    assert "## Ethical considerations" in card and "PCA" in card
    assert f"{out['models']['champion']['test']['pr_auc']:.4f}" in card
