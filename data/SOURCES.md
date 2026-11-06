# Data sources

## Credit card fraud (ULB / Worldline), via OpenML

- Name: `creditcard`, OpenML data id 1597 (version 1)
- URL: https://www.openml.org/d/1597
- Original source: Dal Pozzolo, Caelen, Johnson, Bontempi. "Calibrating Probability with
  Undersampling for Unbalanced Classification", IEEE SSCI 2015. Collected by Worldline and the
  Machine Learning Group of ULB (Universite Libre de Bruxelles).
- Licence: Open Database License (ODbL) v1.0, database contents under the Database Contents
  License (DbCL) v1.0, as stated by the dataset's publisher on Kaggle/OpenML.
- Downloaded: 2026-11-07 with `sklearn.datasets.fetch_openml(data_id=1597, as_frame=True)`
  (`scripts/fetch_data.py`), stored in `data/raw/` (gitignored, never executed).
- Content: 284,807 transactions over two days, 492 frauds (0.173%), features `V1`-`V28`
  (PCA output, original features withheld), `Amount`, label `Class`.

### Known deviation: no `Time` column

The OpenML copy does not include the Kaggle `Time` column. It keeps the rows in their original
chronological order, so `scripts/fetch_data.py` inserts `Time` as the zero-based row position.
That is an ordinal clock, not seconds. The time-ordered split, drift windows and sample sorting
rely only on the order, never on absolute seconds. `Time` is never a model feature.

### Committed sample

`data/sample/creditcard_sample.parquet`: 5,000 rows, drawn independently per class with seed 42
(250 frauds, 4,750 non-frauds), sorted by `Time`. The fraud rate is enriched to 5% (the full
data is 0.173%) because a proportional draw would hold about 9 frauds, too few to train or
evaluate. Sample metrics are for CI and tests only and are not comparable with full-data metrics.

## Synthetic stream

`scripts/synth.py` generates a labelled synthetic stream (Gaussian features, seeded) for drift
demos. It is generated, not downloaded, and carries no licence constraint.
