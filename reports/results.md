| Metric (test split) | Baseline (logistic regression) | Champion (LightGBM) | Challenger (LightGBM, no reweighting) |
|---|---|---|---|
| PR-AUC, 95% bootstrap CI | 0.741 [0.626, 0.851] | 0.783 [0.692, 0.864] | 0.761 [0.660, 0.843] |
| ROC-AUC (reported, not optimised) | 0.982 | 0.962 | 0.985 |
| Precision @ top 1% | 0.114 | 0.111 | 0.114 |
| Recall @ top 1% | 0.867 | 0.840 | 0.867 |
| Precision @ top 0.1% | 0.895 | 0.895 | 0.842 |
| Recall @ top 0.1% | 0.680 | 0.680 | 0.640 |
| Chosen threshold (calibrated probability) | 0.0154 | 0.0435 | 0.0202 |
| Expected cost per 10k transactions (FP 5, FN 150) | 469.6 | 444.2 | 454.7 |
| Brier score (calibrated) | 0.000465 | 0.000468 | 0.000506 |
| Train time (s, CPU) | 0.6 | 6.1 | 2.1 |
| p95 `/score` latency | n/a | 20.8 ms (with reason codes) | n/a |

Data: `creditcard.parquet`, 284,807 rows, 492 frauds. Time-ordered 60/20/20 split; the test split has 56,962 rows and 75 frauds (0.132%). Seed 42, 1000 bootstrap resamples. Champion version `champion-20a4a933`. Generated 2026-11-06 by `scripts/results_table.py` from `reports/metrics.json` and `reports/bench.json`.
