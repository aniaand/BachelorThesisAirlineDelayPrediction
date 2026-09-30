# US Flight Delay Prediction — Modeling Documentation

## 1. Research design

The study tests whether adding external signals — weather and structured news events — improves prediction of US flight delays (`dep_del15`, departure delay ≥15 min) beyond historical flight data alone. Three information layers are compared:

- **Layer A** — BTS On-Time Performance only
- **Layer B** — BTS + Meteostat weather
- **Layer C** — BTS + GDELT news events
- **Layer D** — BTS + weather + GDELT (all layers combined)

Two model families are evaluated on all four layers: **Random Forest** (classical ML) and **TabPFN v3.5** (tabular foundation model, zero-shot in-context inference).

## 2. Data cleaning

- **Airport filtering**: airports are kept based on their *minimum* (not average) daily flight count, with a floor of 10 flights/day. This retains 114 of 358 airports — 86.8% of all flight rows — while dropping airports with sparse, unreliable daily traffic.
- **Processing/engineering separation**: cleaning and merging scripts (`process_bts.py`, `process_meteostat.py`, `process_gdelt.py`) do no imputation, transformation, or feature derivation. All encodings and derived features (target encoding, cyclical encoding, lags/rolling aggregates) are computed later, in `feature_engineering.py`.

- **GDELT-specific cleaning**: restricted to four relevant CAMEO root codes, aggregated to airport-level daily counts, and queried with `_PARTITIONTIME` filtering to keep BigQuery scans cost-safe.

## Feature encoding

Three encoding strategies are used for categorical and cyclical features, each chosen for a different reason:

- **One-hot encoded carriers** (`carrier_*`): `reporting_airline` has only ~15 distinct values with no natural ordering, so one-hot encoding is cheap and avoids inventing a false ranking between airlines.

- **Target-encoded origin/destination airports** (`origin_te`, `dest_te`): with 114 airports, one-hot encoding would add 114+ sparse columns per side. Instead, each airport is replaced with a smoothed average of the target (`dep_del15`) computed from that airport's *own 2024 training rows only*, K-fold out-of-fold for training rows to avoid self-leakage, with light smoothing toward the global rate for low-count airports. This keeps the feature space compact while still capturing that some airports are structurally more delay-prone than others.

- **Sin/cos cyclical encoding** (`hour_sin/cos`, `dow_sin/cos`, `month_sin/cos`): calendar and time-of-day features are cyclical, not linear Mapping each value onto a circle via sine and cosine pairs preserves the true "distance" between, say, 11pm and midnight, without exploding the column count the way one-hot would.

## 3. GDELT categories and meaning

News events are pulled from GDELT's BigQuery table, restricted to events within 50km of a retained airport. Four CAMEO root codes are kept:

| Code | Category | Share of matched events |
|---|---|---|
| 17 | COERCE | ~67% |
| 18 | ASSAULT | ~20% |
| 14 | PROTEST | ~12% |
| 20 | MASS VIOLENCE | <1% |

Important framing: GDELT counts reflect **media coverage**, not the number of discrete real-world events — a higher count means more/louder reporting near an airport, not necessarily more incidents. This is treated as a known limitation, not a flaw to hide.

## 4. Weather events and aggregation

Hourly weather is pulled from Meteostat (`ms.hourly()`, station lookup via `ms.stations.nearby`) for each retained airport. Weather variables are aggregated into rolling windows ending at scheduled departure time — the modeling features use **12-hour trailing means** (e.g. `temp_mean_12h`, `wspd_mean_12h`, `cldc_mean_12h`, plus a precipitation equivalent), giving the model a same-day pre-departure weather picture rather than raw hourly readings.

## 5. Train/test split and why

- **Temporal split**: full 2024 as training, full 2025 as the out-of-sample holdout. This is framed as *temporal integrity* rather than an arbitrary ratio — the model is trained only on the past and evaluated only on the future, matching how it would actually be deployed and avoiding any leakage from 2025 into training.
- **Subsampling**: TabPFN v3.5 has a joint row×feature budget it cannot exceed, so a ~1M-row subsample of the **2024 training data only** is used for it. The subsample is stratified jointly on the delay outcome (`dep_del15`) and origin airport, so both the true class balance and the relative traffic volume of each of the 114 retained airports are preserved rather than left to chance. Critically, **the full 2025 test set (~5.97M rows) is never subsampled** — every model, RF or TabPFN, full-data or subsampled, is scored against the identical, complete 2025 holdout. This keeps comparisons across models and layers apples-to-apples. Classical ML (RF) is run both on the full training data and on the same subsample, so the subsample-vs-full comparison isolates the effect of data volume from the effect of model architecture.

## 6. Columns per dataset

Every layer file also carries `flight_id`, `year`, and the target `dep_del15` (excluded from features).

| Layer | Total cols | Feature cols | Feature families |
|---|---|---|---|
| A — `layer_a_bts` | 28 | 25 | Cyclical time encodings (`hour_sin/cos`, `month_sin/cos`, `dow_sin/cos`), target-encoded `origin_te`/`dest_te`, `distance`, `crs_elapsed_time`, one-hot `carrier_*` (15 airlines) |
| B — `layer_b_bts_meteostat` | 32 | 29 | Layer A + 4 weather features: `temp_mean_12h`, `prcp_sum_12h`, `wspd_mean_12h`, `cldc_mean_12h` |
| C — `layer_c_bts_gdelt` | 43 | 40 | Layer A + 15 GDELT features: `_lag1d`/`_lag3d`/`_lag7d` rolling counts for `total`, `coerce`, `assault`, `protest`, `mass_violence` (5 categories × 3 lag windows). Same-day counts are deliberately excluded — see note below |
| D — `layer_d_bts_meteostat_gdelt` | 47 | 44 | Layer A + weather (4) + GDELT (15) |

Row counts: full layers = 12,028,696 rows (6,062,137 train / 5,966,559 test). Subsampled layers = 6,966,559 rows (1,000,000 subsampled train / 5,966,559 full test, unchanged).


## 7. Threshold and beta — prioritizing recall

Missing a real delay is treated as more costly than a false alarm, so both model families tune their decision threshold with an **F-beta score, beta = 1.2**, which weights recall above precision. In both cases, `tune_threshold()` runs `precision_recall_curve` on a set of "safe" (never-test-touching) probabilities, computes F-beta at every candidate threshold, and keeps the argmax — that threshold is then frozen and applied once to the 2025 test probabilities. Beta itself is fixed, not tuned; a sensitivity sweep across beta ∈ {1, 1.2, 1.5, 2} was run post-hoc on both models to show the precision/recall trade-off at each setting.

Where the two model families differ is where those "safe" probabilities come from, and how the model itself is trained:

### Random Forest

RF uses out-of-bag (OOB) probabilities from the bagged trees — unbiased, in-sample estimates that require no held-out split, since each tree only votes on rows it didn't see during its own bootstrap sample. `tune_threshold()` runs directly on these OOB probabilities.

### TabPFN

TabPFN v3.5 is run **zero-shot**: Because there's no OOB equivalent for an in-context model, a stratified 15% validation split is carved out of the 2024 training data instead, and `tune_threshold()` runs on validation probabilities in its place.

## 8. Hyperparameter tuning

**Random Forest**: a full grid search (`max_depth` × `min_samples_leaf` × `n_estimators`, 4×4×4 = 64 combinations) was run on a 100k-row stratified sample drawn from the 1M-row subsample — small enough to make the full grid tractable given time available. Each combination's OOB score, fit time, and OOB-per-minute efficiency were logged. The absolute-best OOB score came from `max_depth=None, min_samples_leaf=1, n_estimators=300` (OOB ≈ 0.753), but the config carried into production was `max_depth=30, min_samples_leaf=10, n_estimators=300` — nearly as strong, much faster to fit, and safer against overfitting at full data scale. `max_features="sqrt"` and `class_weight="balanced"` were fixed throughout. Aditionally, full dataset Random Forest uses `max_samples=0.3` due to computational and memory limitations.

**TabPFN v3.5**: the model itself isn't hyperparameter-tuned in the classical sense; the levers that matter are the ensemble size (`n_estimators`) and the in-context sample budget (`SUBSAMPLE_SAMPLES`), both of which trade accuracy against GPU time. These were set on a small sample given the Kaggle GPU/session time available, landing on `n_estimators=2` and `SUBSAMPLE_SAMPLES=20,000` (prediction done in 20k-row batches to manage memory).

## 9. Results

### Random Forest — subsampled layers (1M train / full 2025 test), beta=1.2

| Layer | F1 | Precision | Recall | AUC-ROC | AUC-PR | MCC | Threshold | OOB score | Fit time |
|---|---|---|---|---|---|---|---|---|---|
| A — BTS | 0.422 | 0.313 | 0.647 | 0.673 | 0.357 | 0.208 | 0.446 | 0.686 | 166s |
| B — +weather | 0.431 | 0.331 | 0.616 | 0.686 | 0.370 | 0.226 | 0.454 | 0.708 | 194s |
| C — +GDELT | 0.424 | 0.326 | 0.605 | 0.677 | 0.357 | 0.215 | 0.453 | 0.717 | 221s |
| D — all | 0.430 | 0.337 | 0.593 | 0.686 | 0.367 | 0.226 | 0.455 | 0.723 | 241s |

### TabPFN v3.5 — subsampled layers, beta=1.2 (zero-shot)

| Layer | F1 | Precision | Recall | AUC-ROC | AUC-PR | MCC | Threshold | Train time |
|---|---|---|---|---|---|---|---|---|
| A — BTS | 0.422 | 0.308 | 0.668 | 0.674 | 0.359 | 0.206 | 0.188 | 6,271s (~1h 45m) |
| B — +weather | 0.430 | 0.315 | 0.678 | 0.684 | 0.369 | 0.221 | 0.187 | 6,413s (~1h 47m) |
| C — +GDELT | 0.422 | 0.306 | 0.678 | 0.673 | 0.355 | 0.205 | 0.187 | 6,819s (~1h 54m) |
| D — all | 0.429 | 0.321 | 0.646 | 0.682 | 0.364 | 0.219 | 0.196 | 6,969s (~1h 56m) |

### Random Forest — full dataset (6.06M train / full 2025 test), beta=1.2

| Layer | F1 | Precision | Recall | AUC-ROC | AUC-PR | MCC | Threshold | OOB score | Fit time |
|---|---|---|---|---|---|---|---|---|---|
| A — BTS | 0.425 | 0.320 | 0.635 | 0.678 | 0.362 | 0.214 | 0.422 | 0.719 | 1,436s (~24 min) |
| B — +weather | 0.432 | 0.339 | 0.592 | 0.689 | 0.373 | 0.230 | 0.426 | 0.749 | 1,801s (~30 min) |
| C — +GDELT | 0.424 | 0.332 | 0.587 | 0.680 | 0.359 | 0.218 | 0.421 | 0.758 | 2,066s (~34 min) |
| D — all | 0.431 | 0.342 | 0.584 | 0.689 | 0.370 | 0.230 | 0.418 | 0.763 | 2,268s (~38 min) |


Weather (Layer B) gives the most consistent lift across all three models; GDELT's contribution is smaller and layer-dependent.

**Recall/precision discrepancy between RF and TabPFN**: at a glance, TabPFN's threshold sits at a very different point on its own probability scale than RF's (roughly 0.16–0.17 vs. RF's 0.45–0.46 in earlier runs), and it consistently lands with higher recall / lower precision than RF at matched F1. This is a calibration artifact, not a quality difference: RF is trained with `class_weight="balanced"`, which pushes its predicted probabilities toward 0.5 for the positive class, while TabPFN's zero-shot probabilities are never rebalanced and sit closer to the true ~20% delay prevalence — so a much lower cutoff is needed to reach a comparable operating point. Since both models are tuned to the same F-beta objective (beta=1.2), they converge to similar F1 but resolve the precision/recall trade-off differently: TabPFN leans further into recall, RF stays closer to balanced. Given the thesis explicitly prioritizes recall, this is not a weakness of the TabPFN result — if anything it's closer to the stated objective.

### Feature importance (RF, subsampled layers, impurity-based)

`hour_sin` is the single strongest feature in every layer. Weather features (`temp_mean_12h`, `wspd_mean_12h`, `cldc_mean_12h`) enter the top 5 as soon as they're available, ahead of schedule-only features like `distance`/`crs_elapsed_time`. GDELT features only appear in the top 10 for Layer C (not Layer D), and even there rank below both the weather and schedule features — consistent with GDELT contributing a smaller, layer-dependent signal.

| Rank | Layer A | Layer B | Layer C | Layer D |
|---|---|---|---|---|
| 1 | hour_sin (.196) | hour_sin (.166) | hour_sin (.151) | hour_sin (.139) |
| 2 | origin_te (.137) | temp_mean_12h (.099) | origin_te (.058) | temp_mean_12h (.058) |
| 3 | dest_te (.123) | origin_te (.090) | dest_te (.056) | origin_te (.050) |
| 4 | distance (.107) | wspd_mean_12h (.082) | month_cos (.047) | dest_te (.050) |
| 5 | crs_elapsed_time (.105) | dest_te (.080) | crs_elapsed_time (.046) | wspd_mean_12h (.044) |

## 10. Statistical significance — marginal differences, tested with DeLong's test


Across every metric and every layer, the three models sit within a few thousandths to a few hundredths of one another — RF, TabPFN, and RF (Full) are all clustered tightly rather than one model clearly dominating. Despite how small these differences are, DeLong's test flags essentially all of the pairwise AUC-ROC comparisons — both RF vs. TabPFN and layer vs. layer within each model — as statistically significant, including comparisons where the raw AUC-ROC difference is under 0.01. However, the significance might be influenced by the scale of data.

