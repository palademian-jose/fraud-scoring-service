# Champion vs challenger (2026-11-07)

Data: `creditcard.parquet`, test split. Baseline = champion `champion-20a4a933`, candidate = challenger `challenger-58049eaf`. Thresholds from `evals.yaml`.

**Decision: KEEP CHAMPION**

**GATE FAILED (1 violation(s))** — baseline `champion-20a4a933` -> candidate `challenger-58049eaf`

| Metric | Baseline | Candidate | Δ | Threshold | Status |
|---|---|---|---|---|---|
| pr_auc | 0.783 | 0.761 | -0.022 | min 0.7, max_drop 0.02 | FAIL — dropped 0.022 > max_drop 0.02 |
| recall_at_1pct | 0.840 | 0.867 | +0.027 | max_drop 0.05 | PASS |
| cost_per_10k | 444.2 | 454.7 | +10.533 | max_increase 50 | PASS |
