from __future__ import annotations

import numpy as np

from fraudscore.data import FEATURES, TARGET, Split
from fraudscore.metrics import ranking_metrics
from fraudscore.models import ModelBundle, fit_bundle, load_bundle, save_bundle
from fraudscore.settings import Settings


def test_champion_beats_chance_on_held_out_time(champion: ModelBundle, split: Split) -> None:
    y = split.test[TARGET].to_numpy()
    m = ranking_metrics(y, champion.margin(split.test), proba=champion.proba(split.test))
    assert m.pr_auc > 0.5
    assert m.roc_auc > 0.9
    assert 0 <= m.brier < 0.05


def test_probabilities_are_monotone_in_margin(champion: ModelBundle, split: Split) -> None:
    margin = champion.margin(split.test)
    proba = champion.proba(split.test)
    order = np.argsort(margin)
    assert (np.diff(proba[order]) >= -1e-12).all()


def test_training_is_deterministic(split: Split, settings: Settings, champion: ModelBundle) -> None:
    again, _ = fit_bundle("champion", split, settings)
    assert again.version == champion.version
    np.testing.assert_allclose(again.margin(split.test), champion.margin(split.test))
    assert again.threshold == champion.threshold


def test_bundle_roundtrip(champion: ModelBundle, split: Split, tmp_path) -> None:
    path = tmp_path / "c.joblib"
    save_bundle(champion, path)
    loaded = load_bundle(path)
    np.testing.assert_allclose(loaded.proba(split.test), champion.proba(split.test))
    assert loaded.features == FEATURES


def test_baseline_and_challenger_train(split: Split, settings: Settings) -> None:
    for role in ("baseline", "challenger"):
        b, _ = fit_bundle(role, split, settings)
        assert b.version.startswith(role)
        assert b.proba(split.test).shape == (len(split.test),)


def test_background_and_reference_are_sampled_from_train(champion: ModelBundle) -> None:
    assert len(champion.background) <= 2000
    assert list(champion.reference.columns) == FEATURES
    assert len(champion.reference_scores) == len(champion.reference)
