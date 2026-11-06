"""Render the README results table from reports/metrics.json and reports/bench.json."""

from __future__ import annotations

from typing import Any

START = "<!-- results:start -->"
END = "<!-- results:end -->"


def _ci(m: dict[str, Any]) -> str:
    return f"{m['pr_auc']:.3f} [{m['pr_auc_ci_low']:.3f}, {m['pr_auc_ci_high']:.3f}]"


def render_results(metrics: dict[str, Any], bench: dict[str, Any] | None) -> str:
    models = metrics["models"]
    roles = ["baseline", "champion", "challenger"]
    head = {"baseline": "Baseline (logistic regression)", "champion": "Champion (LightGBM)"}
    head["challenger"] = "Challenger (LightGBM, no reweighting)"
    a = metrics["assumptions"]
    test = metrics["data"]["split"]["test"]

    def row(label: str, fn) -> str:
        return f"| {label} | " + " | ".join(fn(models[r]) for r in roles) + " |"

    p95 = "not measured"
    if bench:
        p95 = f"{bench['with_reason_codes']['p95_ms']:.1f} ms (with reason codes)"
    champ = models["champion"]
    lines = [
        "| Metric (test split) | " + " | ".join(head[r] for r in roles) + " |",
        "|---|---|---|---|",
        row("PR-AUC, 95% bootstrap CI", lambda m: _ci(m["test"])),
        row("ROC-AUC (reported, not optimised)", lambda m: f"{m['test']['roc_auc']:.3f}"),
        row("Precision @ top 1%", lambda m: f"{m['test']['precision_at_1pct']:.3f}"),
        row("Recall @ top 1%", lambda m: f"{m['test']['recall_at_1pct']:.3f}"),
        row("Precision @ top 0.1%", lambda m: f"{m['test']['precision_at_01pct']:.3f}"),
        row("Recall @ top 0.1%", lambda m: f"{m['test']['recall_at_01pct']:.3f}"),
        row("Chosen threshold (calibrated probability)", lambda m: f"{m['threshold']:.4f}"),
        row(
            f"Expected cost per 10k transactions (FP {a['cost_fp']:g}, FN {a['cost_fn']:g})",
            lambda m: f"{m['test']['cost']['cost_per_10k']:.1f}",
        ),
        row("Brier score (calibrated)", lambda m: f"{m['test']['brier']:.6f}"),
        row("Train time (s, CPU)", lambda m: f"{m['trained_on']['train_seconds']:.1f}"),
        f"| p95 `/score` latency | n/a | {p95} | n/a |",
        "",
        f"Data: `{metrics['data']['file']}`, {metrics['data']['rows']:,} rows, "
        f"{metrics['data']['frauds']} frauds. Time-ordered 60/20/20 split; the test split has "
        f"{test['rows']:,} rows and {test['frauds']} frauds ({test['fraud_rate']:.3%}). "
        f"Seed {a['seed']}, {a['bootstrap_resamples']} bootstrap resamples. "
        f"Champion version `{champ['version']}`. Generated {metrics['generated']} by "
        "`scripts/results_table.py` from `reports/metrics.json`"
        + (" and `reports/bench.json`." if bench else "."),
    ]
    return "\n".join(lines) + "\n"


def inject(readme: str, table: str) -> str:
    if START not in readme or END not in readme:
        raise ValueError("README has no results markers")
    head, rest = readme.split(START, 1)
    _, tail = rest.split(END, 1)
    return f"{head}{START}\n{table}{END}{tail}"
