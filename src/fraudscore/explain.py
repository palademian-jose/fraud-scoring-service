"""SHAP explanations: global importance and per-decision reason codes."""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
import shap

from fraudscore.models import ModelBundle

TOP_K = 5


def get_explainer(bundle: ModelBundle) -> Any:
    """TreeExplainer (path-dependent, so scoring one row stays in the millisecond range)."""
    if bundle.kind != "lightgbm":
        raise ValueError("SHAP reason codes need a tree model")
    if bundle._explainer is None:
        bundle._explainer = shap.TreeExplainer(bundle.model)
    return bundle._explainer


def shap_matrix(bundle: ModelBundle, X: pd.DataFrame) -> np.ndarray:
    """SHAP values in log-odds units, shape (rows, features)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        values = get_explainer(bundle).shap_values(X[bundle.features])
    if isinstance(values, list):  # older shap returns one array per class
        values = values[-1]
    values = np.asarray(values)
    if values.ndim == 3:
        values = values[..., -1]
    return values


def global_importance(bundle: ModelBundle, X: pd.DataFrame | None = None) -> pd.DataFrame:
    """Mean absolute SHAP value per feature over the 2,000-row background sample."""
    X = bundle.background if X is None else X
    values = shap_matrix(bundle, X)
    imp = pd.DataFrame(
        {"feature": bundle.features, "mean_abs_shap": np.abs(values).mean(axis=0)}
    ).sort_values("mean_abs_shap", ascending=False, kind="stable")
    return imp.reset_index(drop=True)


def reason_codes(
    bundle: ModelBundle, X: pd.DataFrame, k: int = TOP_K
) -> list[list[dict[str, float | str]]]:
    """Top-k features by absolute SHAP contribution for each row.

    `contribution` is in log-odds; positive means the feature pushed the score toward fraud.
    """
    values = shap_matrix(bundle, X)
    out: list[list[dict[str, float | str]]] = []
    for row_idx in range(values.shape[0]):
        order = np.argsort(-np.abs(values[row_idx]), kind="stable")[:k]
        codes: list[dict[str, float | str]] = []
        for j in order:
            c = float(values[row_idx, j])
            codes.append(
                {
                    "feature": bundle.features[j],
                    "value": float(X.iloc[row_idx][bundle.features[j]]),
                    "contribution": round(c, 4),
                    "direction": "increases_risk" if c > 0 else "decreases_risk",
                }
            )
        out.append(codes)
    return out
