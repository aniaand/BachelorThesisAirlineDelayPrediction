# US Flight Delay Prediction: Methodology

## 1. Research design

The study tests whether external signals (weather and structured news events) improve the prediction of US flight delays beyond historical flight data alone. The target is `dep_del15` (departure delay of at least 15 minutes, binary, BTS definition).

Four nested information layers:

- **Layer A**: BTS On-Time Performance only
- **Layer B**: BTS + Meteostat weather
- **Layer C**: BTS + GDELT news events
- **Layer D**: BTS + weather + GDELT

Five models, each run on every layer (4 × 5 grid):

- **Logistic regression (LR)**: linear baseline
- **Random Forest (Full)**: trained on the full 2024 training data
- **Random Forest (RF)**: trained on the 1M-row subsample
- **TabPFN-3.5 zero-shot**: tabular foundation model, in-context inference without weight updates
- **TabPFN-3.5 fine-tuned (FT)**: same model with weights fine-tuned on the subsample

Running every model on every layer separates the effect of additional information from the effect of the modeling approach.

## 2. Data

| Step | Flights | Airports |
|---|---|---|
| Raw BTS data, 2024–2025 | 14,080,680 | 358 |
| After airport filter | 12,226,431 | 114 |
| Excluded: cancelled / diverted | −167,656 / −30,079 | |
| Analysis sample | 12,028,696 | 114 |
| Training set (2024) / holdout (2025) | 6,062,137 / 5,966,559 | 114 |
| Training subsample (2024, stratified) | 1,000,000 | 114 |

- **Airport filter**: airports are kept if their *minimum* daily flight count is at least 10 (114 of 358 airports, 86.8% of rows).
- **Exclusions**: cancelled and diverted flights are removed, since their delay outcome is undefined. This may introduce selection bias (see Section 9).
- **Pre-departure information only**: actual times, taxi times and the delay-cause breakdown are dropped.
- **Processing/engineering separation**: the processing and merging scripts do no encoding or modeling transformations; all encodings happen in `feature_engineering.py`.

**Weather (Meteostat)**: hourly observations from the nearest station, aggregated into 12-hour trailing windows before the scheduled departure hour (mean temperature, wind speed, cloud cover; summed precipitation). Remaining missing values (0.9–1.2%) are filled with the 2024 median, also applied to 2025. Honolulu's station only reports from July 2025, so HNL weather is largely imputed; a robustness check excluding HNL leaves the conclusions unchanged.

**News events (GDELT)**: events within 50 km of each airport for CAMEO root codes 14 (PROTEST), 17 (COERCE), 18 (ASSAULT) and 20 (MASS VIOLENCE), aggregated to daily airport-level counts per category plus a total, then summed over 1-, 3- and 7-day windows before the flight date (same-day counts excluded). This gives 15 features. Counts measure media coverage, not the number of real-world events.

## 3. Feature encoding

- **Carrier**: one-hot (15 airlines).
- **Origin and destination airports**: smoothed target encoding (m = 10), computed from 2024 rows only. Training rows are encoded out-of-fold with five folds, so a flight's own outcome never enters its encoding; holdout rows use statistics from the full training year.
- **Month, day of week, departure hour**: sin/cos encoding.

| Layer | Feature columns |
|---|---|
| A | 25 (calendar 6, airports 2, carrier 15, elapsed time and distance 2) |
| B | 29 (A + 4 weather) |
| C | 40 (A + 15 GDELT) |
| D | 44 (A + 4 weather + 15 GDELT) |

## 4. Train/test split and subsampling

- **Temporal split**: all of 2024 for training, all of 2025 as the out-of-sample holdout.
- **Subsample**: about 1M rows of 2024 data, stratified jointly on the delay outcome and origin airport, used by TabPFN (row limit) and by the subsampled Random Forest.
- **Holdout never subsampled**: every model is scored on the identical, complete 2025 holdout (5.97M flights).
- **TabPFN validation split**: 15% of the subsample (150,000 rows) is held out for threshold tuning and early stopping, leaving 850,000 rows.

## 5. Models and tuning

**Logistic regression**: linear baseline trained on the full 2024 data with the same encoded features. Features are standardised (`StandardScaler`), followed by `LogisticRegression` with L2 regularisation (`C=1.0`), the `lbfgs` solver (`max_iter=2000`) and `class_weight="balanced"`. Coefficients are therefore comparable within a layer (change in log-odds per standard deviation of a feature).

**Random Forest**: grid search over `max_depth` × `min_samples_leaf` × `n_estimators` on a 100,000-row sample of Layer D, scored by OOB AUC-ROC. Final configuration for all layers and both tiers: depth 30, minimum leaf size 10, 300 trees, `max_features="sqrt"`, `class_weight="balanced"`. The full-data Random Forest uses `max_samples=0.3` for memory and runtime reasons.

**TabPFN-3.5 zero-shot**: number of estimators and context rows probed on 30,000 Layer D flights. All configurations except the smallest lie within about 0.002 AUC-ROC of each other, and a larger context (40,000 rows) did not help. **2 estimators with 20,000 context rows** were retained as a runtime compromise.

**TabPFN-3.5 fine-tuned**: learning-rate search on a 100,000-row Layer D sample. Final runs: learning rate 3e-5, up to 20 epochs, early stopping on validation AUC-ROC (patience 8), 2 estimators and 20,000 context rows as in zero-shot.

## 6. Decision threshold

Thresholds are tuned with the **F-beta score at β = 1.2**, weighting recall slightly above precision, on probabilities that never touch the test set. The tuned threshold is then applied once to the 2025 holdout. This changes only the decision cutoff, so AUC-ROC and AUC-PR are unaffected.

- **Random Forest**: out-of-bag probabilities on 2024 data.
- **Logistic regression**: out-of-fold probabilities from 5-fold stratified cross-validation on 2024 data; the model is then refit on all of 2024.
- **TabPFN**: the 15% validation split.

β = 1.2 was fixed in advance. A sensitivity analysis afterwards (β = 1, 1.2, 1.5, 2) shows it is within 0.001 of the highest F1 in every layer.

## 7. Evaluation

**Metrics**: AUC-ROC (primary), AUC-PR, F1-score, MCC, precision and recall, all on the full 2025 holdout.

**Statistical comparison**: pairwise AUC-ROC differences are tested with DeLong's test for correlated ROC curves, within two families (layer vs. layer per model, and model vs. model per layer), with Bonferroni-corrected p-values and 95% confidence intervals.

With almost six million test flights, nearly every difference is statistically significant, so each comparison is classified using a **minimum meaningful difference of 0.005 AUC-ROC**:

- **meaningful**: the corrected interval lies entirely beyond ±0.005
- **negligible**: the interval lies entirely within ±0.005
- **inconclusive**: the interval crosses ±0.005

## 8. Robustness checks

- **Honolulu**: trained models re-evaluated on the holdout without HNL (no retraining). Layer and model gains remain unchanged.
- **Target encoding**: the subsampled Random Forest is retrained with and without the airport encodings, comparing the gap between OOB and holdout AUC-ROC. The gap is essentially unchanged, which gives no indication of leakage, but the check cannot rule it out conclusively.

## 9. Methodological limitations

- **Dependence between flights**: DeLong's test assumes independent observations, but flights on the same day and airport share conditions, so its confidence intervals are likely too narrow. The 0.005 threshold addresses practical relevance but does not correct this; a cluster-based bootstrap (resampling days or airport-days) would be the more rigorous approach.
- **Selection bias**: excluding cancelled and diverted flights removes many of the most disrupted days, so the contribution of weather and news features is likely understated.
- **Unequal data exposure**: zero-shot TabPFN conditions on at most about 40,000 training flights per prediction, while the subsampled Random Forest is trained on 1M rows. The comparison reflects each model family under a practical compute budget, not identical exposure to the training data.
- **Thin baseline**: Layer A contains no recent delay state (e.g. previous-day delay rate or inbound aircraft delay).
- **Single holdout year**: results may partly reflect conditions specific to 2025.
- **Environment**: the Layer C fine-tuning run used a different package and Python version than the other layers.

## 10. Computational setup

- **Local** (Python 3.12, conda environment `thesis`): data pipeline, logistic regression, both Random Forest tiers, result analysis.
- **Kaggle GPU**: all TabPFN notebooks (zero-shot about 1h45 per layer; fine-tuning about 8.5–10 hours per layer).
- **Outputs**: summary CSVs, per-flight test probabilities and checkpoints in `data/final/modeling_results/`. `results.ipynb` builds the comparison tables, DeLong tests and figures.