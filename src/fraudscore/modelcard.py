"""Render `reports/model_card.md` (Google model-card layout) from `reports/metrics.json`."""

from __future__ import annotations

from typing import Any


def _f(x: float | None, nd: int = 4) -> str:
    return "n/a" if x is None else f"{x:.{nd}f}"


def _split_row(name: str, m: dict[str, Any]) -> str:
    return (
        f"| {name} | {m['n']:,} | {m['n_fraud']} | {_f(m['pr_auc'])} | {_f(m['roc_auc'])} | "
        f"{_f(m['precision_at_1pct'])} | {_f(m['recall_at_1pct'])} | "
        f"{_f(m['precision_at_01pct'])} | {_f(m['recall_at_01pct'])} | {_f(m['brier'], 6)} |"
    )


def render_model_card(metrics: dict[str, Any]) -> str:
    champ = metrics["models"]["champion"]
    base = metrics["models"]["baseline"]
    data = metrics["data"]
    a = metrics["assumptions"]
    test = champ["test"]
    ci = f"[{_f(test['pr_auc_ci_low'], 3)}, {_f(test['pr_auc_ci_high'], 3)}]"
    cost = test["cost"]
    sp = data["split"]

    lines = [
        "# Model card: fraud-scoring-service champion",
        "",
        f"Generated {metrics['generated']} by `scripts/model_card.py` from `reports/metrics.json`. "
        "Every number below is read from that file.",
        "",
        "## Model details",
        "",
        f"- Version: `{champ['version']}`",
        "- Type: LightGBM gradient-boosted trees, binary classifier, positive-class weight set to "
        "the square root of the train class ratio, early stopping on validation PR-AUC.",
        "- Calibration: Platt (sigmoid) scaling of the raw margin, fitted on the validation split.",
        f"- Alert threshold: {champ['threshold']:.4f} on the calibrated probability, chosen on "
        f"validation to minimise `{a['cost_fp']:g} x false positives + {a['cost_fn']:g} x "
        "false negatives`.",
        f"- Seed: {a['seed']}. Training rows: {champ['trained_on']['n_train']:,} "
        f"({champ['trained_on']['n_fraud_train']} frauds), best iteration "
        f"{champ['trained_on']['best_iteration']}, train time "
        f"{champ['trained_on']['train_seconds']} s.",
        "- Baseline for comparison: standardised, class-balanced logistic regression.",
        "",
        "## Intended use",
        "",
        "- Primary use: rank card transactions by fraud risk so analysts review the top of the "
        "queue, and return the top five SHAP reason codes with each decision.",
        "- Intended users: a fraud-operations team and the engineers who run the scoring service.",
        "- Out of scope: automatic account closure, credit decisions, any use on populations "
        "other than the two-day European card sample below, or on features other than the "
        "PCA components V1-V28 and Amount.",
        "",
        "## Data",
        "",
        "- Source: OpenML `creditcard` (data id 1597), the ULB/Worldline card-fraud dataset "
        "(ODbL, see `data/SOURCES.md`).",
        f"- File: `{data['file']}`, {data['rows']:,} transactions, {data['frauds']} frauds.",
        "- Features: V1-V28 (PCA output, original features withheld) and Amount. "
        "`Time` is used only to order the split and is not a model input.",
        "- Split: time-ordered 60/20/20 (no shuffling), so the test split is strictly later "
        "than training.",
        "",
        "| Split | Rows | Frauds | Fraud rate |",
        "|---|---|---|---|",
    ]
    for name in ("train", "valid", "test"):
        row = sp[name]
        lines.append(f"| {name} | {row['rows']:,} | {row['frauds']} | {row['fraud_rate']:.4%} |")
    lines += [
        "",
        "## Metrics by split (champion)",
        "",
        "PR-AUC is the primary metric. ROC-AUC is reported but not optimised. "
        "Precision and recall are at the top 1% and 0.1% of scores. Brier uses the calibrated "
        "probability.",
        "",
        "| Split | Rows | Frauds | PR-AUC | ROC-AUC | P@1% | R@1% | P@0.1% | R@0.1% | Brier |",
        "|---|---|---|---|---|---|---|---|---|---|",
        _split_row("train", champ["train"]),
        _split_row("valid", champ["valid"]),
        _split_row("test", champ["test"]),
        "",
        f"Test PR-AUC with 95% bootstrap interval ({a['bootstrap_resamples']} resamples): "
        f"{_f(test['pr_auc'])} {ci}.",
        "",
        "Baseline (logistic regression) on the test split: "
        f"PR-AUC {_f(base['test']['pr_auc'])}, ROC-AUC {_f(base['test']['roc_auc'])}, "
        f"R@1% {_f(base['test']['recall_at_1pct'])}.",
        "",
        "## Quantitative analysis: cost at the chosen threshold (test split)",
        "",
        f"- Alerts raised: {cost['tp'] + cost['fp']} of {test['n']:,} "
        f"({cost['alert_rate']:.3%}); true positives {cost['tp']}, false positives {cost['fp']}, "
        f"missed frauds {cost['fn']}.",
        f"- Expected cost per 10,000 transactions: {cost['cost_per_10k']:.1f} "
        f"(cost_fp {a['cost_fp']:g}, cost_fn {a['cost_fn']:g}).",
        "- The cost assumptions are configuration, not facts about any real bank. Change "
        "`COST_FP` and `COST_FN` and retrain to move the threshold.",
        "",
        "Figures: `reports/shap_importance.png`, `reports/calibration.png`, "
        "`reports/pr_curve.png`.",
        "",
        "## Limitations",
        "",
        f"- Few positives: the test split holds {test['n_fraud']} frauds, so metrics at the top "
        "0.1% rest on a handful of cases. Read the bootstrap interval, not the point estimate.",
        "- Two days of transactions from one issuer population. Fraud patterns change; the "
        "model has not been tested on other periods, merchants or regions.",
        "- Time-ordered evaluation removes look-ahead leakage but not label delay: "
        "real chargebacks arrive days or weeks after the transaction, so recent labels are "
        "incomplete in production.",
        "- Calibration is fitted on a validation split with few positives and can drift; "
        "re-check the reliability curve after every retrain.",
        "- Drift monitoring (KS, PSI) flags input and score shifts but cannot see label shift.",
        "",
        "## Ethical considerations",
        "",
        "- Features V1-V28 are PCA components. The original attributes were withheld for "
        "confidentiality, so reason codes name components, not human-readable causes, and "
        "the model cannot be audited for use of protected attributes or their proxies.",
        "- A false positive blocks or delays a legitimate customer. The alert threshold trades "
        "that harm against fraud loss; the trade is explicit in the cost settings and should "
        "be reviewed by a human owner, not tuned silently.",
        "- No demographic data is present, so no fairness analysis was possible. Do not "
        "deploy without one on real customer data.",
        "- Scores are decision support: an analyst reviews alerts and the customer has a "
        "route to contest a block.",
        "",
        "## Caveats and recommendations",
        "",
        "- Retrain on a rolling window and re-run `fraudscore compare` and the gate before "
        "promoting a challenger.",
        "- Run `fraudscore drift` on every scoring batch; investigate before retraining.",
        "",
    ]
    return "\n".join(lines)
