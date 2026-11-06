"""Model definitions, the serialisable bundle and the training routine."""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from fraudscore.data import FEATURES, TARGET, Split
from fraudscore.settings import Settings
from fraudscore.threshold import PlattCalibrator, choose_threshold

REFERENCE_ROWS = 20_000


def champion_params(pos_weight: float, seed: int) -> dict[str, Any]:
    """LightGBM with a damped positive-class weight (square root of the class ratio)."""
    return {
        "n_estimators": 600,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_child_samples": 20,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.8,
        "reg_lambda": 1.0,
        "scale_pos_weight": math.sqrt(pos_weight),
        "random_state": seed,
        "n_jobs": 4,
        "verbose": -1,
        "deterministic": True,
        "force_row_wise": True,
    }


def challenger_params(pos_weight: float, seed: int) -> dict[str, Any]:
    """A deliberately different candidate: shallower trees, no reweighting, stronger shrinkage."""
    p = champion_params(pos_weight, seed)
    p.update(
        num_leaves=15,
        learning_rate=0.03,
        n_estimators=800,
        min_child_samples=40,
        scale_pos_weight=1.0,
        colsample_bytree=0.6,
    )
    return p


@dataclass
class ModelBundle:
    """Everything needed to score a transaction: model, calibrator, threshold, references."""

    version: str
    kind: str
    model: Any
    calibrator: PlattCalibrator
    threshold: float
    features: list[str]
    cost_fp: float
    cost_fn: float
    params: dict[str, Any]
    background: pd.DataFrame  # rows used for SHAP global importance
    reference: pd.DataFrame  # training feature sample for drift
    reference_scores: np.ndarray  # calibrated probabilities on `reference`
    trained_on: dict[str, Any] = field(default_factory=dict)
    _explainer: Any = field(default=None, repr=False, compare=False)

    def margin(self, X: pd.DataFrame) -> np.ndarray:
        X = X[self.features]
        if self.kind == "lightgbm":
            return np.asarray(self.model.predict(X, raw_score=True))
        return np.asarray(self.model.decision_function(X))

    def proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.calibrator.predict(self.margin(X))

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state["_explainer"] = None
        return state


def save_bundle(bundle: ModelBundle, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path, compress=3)


def load_bundle(path: Path) -> ModelBundle:
    # The bundle is a pickle: load only files this repo produced (models/ is gitignored).
    bundle = joblib.load(path)
    if not isinstance(bundle, ModelBundle):
        raise TypeError(f"{path} does not contain a ModelBundle")
    return bundle


def fit_baseline(split: Split, seed: int) -> Any:
    """Standardised logistic regression, class-balanced. The bar the champion must clear."""
    pipe = make_pipeline(
        StandardScaler(),
        LogisticRegression(class_weight="balanced", max_iter=2000, C=0.1, random_state=seed),
    )
    pipe.fit(split.train[FEATURES], split.train[TARGET])
    return pipe


def _pos_weight(y: pd.Series) -> float:
    pos = int(y.sum())
    return (len(y) - pos) / max(pos, 1)


def fit_bundle(role: str, split: Split, settings: Settings) -> tuple[ModelBundle, dict[str, Any]]:
    """Train one LightGBM bundle (`champion` or `challenger`) or the `baseline` logistic model.

    Early stopping and calibration use the validation split only; the test split is untouched.
    """
    seed = settings.seed
    started = time.perf_counter()
    if role == "baseline":
        kind, params = "logreg", {"C": 0.1, "class_weight": "balanced"}
        model = fit_baseline(split, seed)
    else:
        kind = "lightgbm"
        pw = _pos_weight(split.train[TARGET])
        params = champion_params(pw, seed) if role == "champion" else challenger_params(pw, seed)
        model = LGBMClassifier(**params)
        model.fit(
            split.train[FEATURES],
            split.train[TARGET],
            eval_X=split.valid[FEATURES],
            eval_y=split.valid[TARGET],
            eval_metric="average_precision",
            callbacks=[early_stopping(50, verbose=False)],
        )
    train_seconds = time.perf_counter() - started

    stub = ModelBundle(
        version="",
        kind=kind,
        model=model,
        calibrator=PlattCalibrator(),
        threshold=0.5,
        features=list(FEATURES),
        cost_fp=settings.cost_fp,
        cost_fn=settings.cost_fn,
        params=params,
        background=pd.DataFrame(),
        reference=pd.DataFrame(),
        reference_scores=np.array([]),
    )
    stub.calibrator.fit(stub.margin(split.valid), split.valid[TARGET].to_numpy())
    valid_proba = stub.proba(split.valid)
    stub.threshold = choose_threshold(
        split.valid[TARGET].to_numpy(), valid_proba, settings.cost_fp, settings.cost_fn
    )

    rng = np.random.default_rng(seed)
    ref_idx = rng.choice(
        len(split.train), size=min(REFERENCE_ROWS, len(split.train)), replace=False
    )
    stub.reference = split.train.iloc[np.sort(ref_idx)][FEATURES].reset_index(drop=True)
    stub.reference_scores = stub.proba(stub.reference)
    bg_rows = min(settings.shap_background_rows, len(split.train))
    bg_idx = rng.choice(len(split.train), size=bg_rows, replace=False)
    stub.background = split.train.iloc[np.sort(bg_idx)][FEATURES].reset_index(drop=True)

    digest = hashlib.sha1(
        json.dumps(
            {"role": role, "params": params, "n": len(split.train), "seed": seed},
            sort_keys=True,
            default=str,
        ).encode()
    ).hexdigest()[:8]
    stub.version = f"{role}-{digest}"
    best_iter = getattr(model, "best_iteration_", None)
    stub.trained_on = {
        "n_train": len(split.train),
        "n_valid": len(split.valid),
        "n_test": len(split.test),
        "n_fraud_train": int(split.train[TARGET].sum()),
        "best_iteration": int(best_iter) if best_iter else None,
        "train_seconds": round(train_seconds, 2),
        "seed": seed,
    }
    return stub, {"train_seconds": train_seconds}
