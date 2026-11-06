# Model card: fraud-scoring-service champion

Generated 2026-11-06 by `scripts/model_card.py` from `reports/metrics.json`. Every number below is read from that file.

## Model details

- Version: `champion-20a4a933`
- Type: LightGBM gradient-boosted trees, binary classifier, positive-class weight set to the square root of the train class ratio, early stopping on validation PR-AUC.
- Calibration: Platt (sigmoid) scaling of the raw margin, fitted on the validation split.
- Alert threshold: 0.0435 on the calibrated probability, chosen on validation to minimise `5 x false positives + 150 x false negatives`.
- Seed: 42. Training rows: 170,884 (360 frauds), best iteration 229, train time 6.06 s.
- Baseline for comparison: standardised, class-balanced logistic regression.

## Intended use

- Primary use: rank card transactions by fraud risk so analysts review the top of the queue, and return the top five SHAP reason codes with each decision.
- Intended users: a fraud-operations team and the engineers who run the scoring service.
- Out of scope: automatic account closure, credit decisions, any use on populations other than the two-day European card sample below, or on features other than the PCA components V1-V28 and Amount.

## Data

- Source: OpenML `creditcard` (data id 1597), the ULB/Worldline card-fraud dataset (ODbL, see `data/SOURCES.md`).
- File: `creditcard.parquet`, 284,807 transactions, 492 frauds.
- Features: V1-V28 (PCA output, original features withheld) and Amount. `Time` is used only to order the split and is not a model input.
- Split: time-ordered 60/20/20 (no shuffling), so the test split is strictly later than training.

| Split | Rows | Frauds | Fraud rate |
|---|---|---|---|
| train | 170,884 | 360 | 0.2107% |
| valid | 56,961 | 57 | 0.1001% |
| test | 56,962 | 75 | 0.1317% |

## Metrics by split (champion)

PR-AUC is the primary metric. ROC-AUC is reported but not optimised. Precision and recall are at the top 1% and 0.1% of scores. Brier uses the calibrated probability.

| Split | Rows | Frauds | PR-AUC | ROC-AUC | P@1% | R@1% | P@0.1% | R@0.1% | Brier |
|---|---|---|---|---|---|---|---|---|---|
| train | 170,884 | 360 | 1.0000 | 1.0000 | 0.2106 | 1.0000 | 1.0000 | 0.4750 | 0.000010 |
| valid | 56,961 | 57 | 0.7735 | 0.9782 | 0.0825 | 0.8246 | 0.7544 | 0.7544 | 0.000324 |
| test | 56,962 | 75 | 0.7829 | 0.9621 | 0.1105 | 0.8400 | 0.8947 | 0.6800 | 0.000468 |

Test PR-AUC with 95% bootstrap interval (1000 resamples): 0.7829 [0.692, 0.864].

Baseline (logistic regression) on the test split: PR-AUC 0.7415, ROC-AUC 0.9820, R@1% 0.8667.

## Quantitative analysis: cost at the chosen threshold (test split)

- Alerts raised: 116 of 56,962 (0.204%); true positives 60, false positives 56, missed frauds 15.
- Expected cost per 10,000 transactions: 444.2 (cost_fp 5, cost_fn 150).
- The cost assumptions are configuration, not facts about any real bank. Change `COST_FP` and `COST_FN` and retrain to move the threshold.

Figures: `reports/shap_importance.png`, `reports/calibration.png`, `reports/pr_curve.png`.

## Limitations

- Few positives: the test split holds 75 frauds, so metrics at the top 0.1% rest on a handful of cases. Read the bootstrap interval, not the point estimate.
- Two days of transactions from one issuer population. Fraud patterns change; the model has not been tested on other periods, merchants or regions.
- Time-ordered evaluation removes look-ahead leakage but not label delay: real chargebacks arrive days or weeks after the transaction, so recent labels are incomplete in production.
- Calibration is fitted on a validation split with few positives and can drift; re-check the reliability curve after every retrain.
- Drift monitoring (KS, PSI) flags input and score shifts but cannot see label shift.

## Ethical considerations

- Features V1-V28 are PCA components. The original attributes were withheld for confidentiality, so reason codes name components, not human-readable causes, and the model cannot be audited for use of protected attributes or their proxies.
- A false positive blocks or delays a legitimate customer. The alert threshold trades that harm against fraud loss; the trade is explicit in the cost settings and should be reviewed by a human owner, not tuned silently.
- No demographic data is present, so no fairness analysis was possible. Do not deploy without one on real customer data.
- Scores are decision support: an analyst reviews alerts and the customer has a route to contest a block.

## Caveats and recommendations

- Retrain on a rolling window and re-run `fraudscore compare` and the gate before promoting a challenger.
- Run `fraudscore drift` on every scoring batch; investigate before retraining.
