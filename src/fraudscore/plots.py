"""PNG figures written under reports/ (matplotlib, headless backend)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def shap_bar(importance: pd.DataFrame, path: Path, top: int = 15) -> None:
    d = importance.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.barh(d["feature"], d["mean_abs_shap"], color="#2b6cb0")
    ax.set_xlabel("mean |SHAP value| (log-odds), 2,000-row background sample")
    ax.set_title("Champion: global feature importance")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def reliability_plot(
    curves: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]], path: Path, title: str
) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 5))
    lim = max([1e-3] + [float(max(c[0].max(), c[1].max())) for c in curves.values()]) * 1.05
    ax.plot([0, lim], [0, lim], "k--", lw=1, label="perfect")
    for name, (pred, obs, _n) in curves.items():
        ax.plot(pred, obs, marker="o", ms=4, label=name)
    ax.set_xlabel("mean predicted probability (quantile bins)")
    ax.set_ylabel("observed fraud rate")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def pr_curve_plot(curves: dict[str, tuple[np.ndarray, np.ndarray]], path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 5))
    for name, (rec, prec) in curves.items():
        ax.plot(rec, prec, label=name)
    ax.set_xlabel("recall")
    ax.set_ylabel("precision")
    ax.set_ylim(0, 1.02)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
