# US Flight Delay Prediction — Modeling Documentation

## 1. Research design

The study tests whether adding external signals (weather and structured news events) improves prediction of US flight delays (`dep_del15`, departure delay ≥15 min) beyond historical flight data alone. Four information layers are compared:

- **Layer A**: BTS On-Time Performance only
- **Layer B**: BTS + Meteostat weather
- **Layer C**: BTS + GDELT news events
- **Layer D**: BTS + weather + GDELT (all layers combined)

Two model families are evaluated on all four layers, in four variants:

- **Random Forest (RF)**, classical ML, trained on the 1M-row subsample
- **Random Forest (Full)**, trained on the full 2024 training data
- **TabPFN v3.5 zero-shot**, a tabular foundation model doing in-context inference without weight updates
- **TabPFN v3.5 fine-tuned (FT)**, the same model with its weights fine-tuned on the subsample

## 2. Data cleaning

- **Airport filtering**: airports are kept based on their *minimum* (not average) daily flight count, with a floor of 10 flights/day. This retains 114 of 358 airports (86.8% of all flight rows) and drops airports with sparse, unreliable daily traffic.
- **Processing/engineering separation**: the cleaning and merging scripts (`process_bts.py`, `process_meteostat.py`, `process_gdelt.py`) do no imputation, transformation or feature derivation. All encodings and derived features (target encoding, cyclical encoding, lags/rolling aggregates) are computed later, in `feature_engineering.py`.
- **GDELT-specific cleaning**: restricted to four relevant CAMEO root codes, aggregated to airport-level daily counts, and queried with `_PARTITIONTIME` filtering to keep BigQuery scans cost-safe.

## 3. Feature encoding

Three encoding strategies are used for categorical and cyclical features, each chosen for a different reason:

- **One-hot encoded carriers** (`carrier_*`): `reporting_airline` has only ~15 distinct values with no natural ordering. One-hot encoding is cheap and avoids inventing a false ranking between airlines.
- **Target-encoded origin/destination airports** (`origin_te`, `dest_te`): with 114 airports, one-hot encoding would add 114+ sparse columns per side. Instead, each airport is replaced with a smoothed average of the target (`dep_del15`), computed from that airport's *own 2024 training rows only*. For training rows the encoding is computed K-fold out-of-fold to avoid self-leakage, with light smoothing toward the global rate for low-count airports. This keeps the feature space compact while still capturing that some airports are structurally more delay-prone than others.
- **Sin/cos cyclical encoding** (`hour_sin/cos`, `dow_sin/cos`, `month_sin/cos`): calendar and time-of-day features are cyclical, not linear. Mapping each value onto a circle via sine and cosine pairs preserves the true distance between, say, 11pm and midnight, without the column count that one-hot encoding would add.

## 4. GDELT categories and meaning

News events are pulled from GDELT's BigQuery table, restricted to events within 50 km of a retained airport. Four CAMEO root codes are kept:

| Code | Category | Share of matched events |
|---|---|---|
| 17 | COERCE | ~67% |
| 18 | ASSAULT | ~20% |
| 14 | PROTEST | ~12% |
| 20 | MASS VIOLENCE | <1% |

GDELT counts reflect **media coverage**, not the number of discrete real-world events. A higher count means more or louder reporting near an airport, not necessarily more incidents. This is treated as a known limitation of the data, not something to hide.

## 5. Weather variables and aggregation

Hourly weather is pulled from Meteostat (`ms.hourly()`, station lookup via `ms.stations.nearby`) for each retained airport. Weather variables are aggregated into 12-hour trailing windows ending at the scheduled departure time: `temp_mean_12h`, `wspd_mean_12h`, `cldc_mean_12h` and `prcp_sum_12h`. This gives the model a picture of the weather before departure on the same day, rather than raw hourly readings.

## 6. Train/test split and subsampling

- **Temporal split**: all of 2024 is used for training and all of 2025 as the out-of-sample holdout. This is a matter of *temporal integrity*, not an arbitrary ratio: the model is trained only on the past and evaluated only on the future, matching how it would be deployed and avoiding any leakage from 2025 into training.
- **Subsampling**: TabPFN v3.5 has a joint row×feature budget it cannot exceed, so it uses a ~1M-row subsample of the **2024 training data only**. The subsample is stratified jointly on the delay outcome (`dep_del15`) and origin airport, so both the true class balance and the relative traffic volume of each of the 114 retained airports are preserved rather than left to chance.
- **The full 2025 test set (~5.97M rows) is never subsampled.** Every model (RF or TabPFN, full-data or subsampled, zero-shot or fine-tuned) is scored against the identical, complete 2025 holdout, which keeps comparisons across models and layers like-for-like.
- **Subsample vs. full data**: RF is run both on the full training data and on the same subsample. The subsample-vs-full comparison isolates the effect of data volume from the effect of model architecture.

## 7. Columns per dataset

Every layer file also carries `flight_id`, `year` and the target `dep_del15`, which are excluded from the features.

| Layer | Total cols | Feature cols | Feature families |
|---|---|---|---|
| A — `layer_a_bts` | 28 | 25 | Cyclical time encodings (`hour_sin/cos`, `month_sin/cos`, `dow_sin/cos`), target-encoded `origin_te`/`dest_te`, `distance`, `crs_elapsed_time`, one-hot `carrier_*` (15 airlines) |
| B — `layer_b_bts_meteostat` | 32 | 29 | Layer A + 4 weather features: `temp_mean_12h`, `prcp_sum_12h`, `wspd_mean_12h`, `cldc_mean_12h` |
| C — `layer_c_bts_gdelt` | 43 | 40 | Layer A + 15 GDELT features: `_lag1d`/`_lag3d`/`_lag7d` rolling counts for `total`, `coerce`, `assault`, `protest`, `mass_violence` (5 categories × 3 lag windows). Same-day counts are deliberately excluded. |
| D — `layer_d_bts_meteostat_gdelt` | 47 | 44 | Layer A + weather (4) + GDELT (15) |

Row counts: the full layers have 12,028,696 rows (6,062,137 train / 5,966,559 test). The subsampled layers have 6,966,559 rows (1,000,000 subsampled train / 5,966,559 full test, unchanged).

## 8. Threshold and beta: prioritizing recall

Missing a real delay is treated as more costly than a false alarm, so all models tune their decision threshold with an **F-beta score, beta = 1.2**, which weights recall above precision. In every case, `tune_threshold()` runs `precision_recall_curve` on a set of "safe" probabilities that never touch the test set, computes F-beta at every candidate threshold and keeps the argmax. That threshold is then frozen and applied once to the 2025 test probabilities. Beta itself is fixed, not tuned. A sensitivity sweep across beta ∈ {1, 1.2, 1.5, 2} was run afterwards to show the precision/recall trade-off at each setting.

The model families differ in where those safe probabilities come from:

- **Random Forest** uses out-of-bag (OOB) probabilities from the bagged trees. These are unbiased estimates that need no held-out split, because each tree only votes on rows it did not see in its own bootstrap sample.
- **TabPFN (zero-shot and fine-tuned)** has no OOB equivalent. A stratified 15% validation split is carved out of the 2024 subsample instead, and `tune_threshold()` runs on the validation probabilities. For the fine-tuned model, the same validation split is also used for early stopping.

## 9. Hyperparameter tuning

**Random Forest**: a full grid search over `max_depth` × `min_samples_leaf` × `n_estimators` (4×4×4 = 64 combinations) was run on a 100k-row stratified sample drawn from the 1M-row subsample, small enough to make the full grid tractable. Each combination's OOB score, fit time and OOB-per-minute efficiency were logged. The best OOB score came from `max_depth=None, min_samples_leaf=1, n_estimators=300` (OOB ≈ 0.753). The configuration used for the final runs was `max_depth=30, min_samples_leaf=10, n_estimators=300`: nearly as strong, much faster to fit and safer against overfitting at full data scale. `max_features="sqrt"` and `class_weight="balanced"` were fixed throughout. The full-dataset RF additionally uses `max_samples=0.3` because of computational and memory limits.

**TabPFN v3.5 zero-shot**: the model is not hyperparameter-tuned in the classical sense. The settings that matter are the ensemble size (`n_estimators`) and the in-context sample budget (`n_inference_subsample_samples`), and both trade accuracy against GPU time. They were set on a small sample, given the Kaggle GPU session time available: `n_estimators=2` and 20,000 context samples. Prediction is done in 20k-row batches to manage memory.

**TabPFN v3.5 fine-tuned**: the learning rate was chosen in a separate search (`fine_tuning_epochs.ipynb`) on a stratified 100k-row sample of the Layer D training data (85k train / 15k validation). Five learning rates were tested (3e-5, 1e-5, 3e-6, 1e-6, 3e-7), each with up to 30 epochs and early stopping on validation ROC AUC (patience 8). All five ended between 0.6985 and 0.6996 validation AUC. **1e-5** scored highest (0.6996, best epoch 7) and was used for the final runs, but it is best described as the top of a flat search, not a clear winner: the differences are smaller than the epoch-to-epoch variation. The final fine-tuning runs (`FM_fine_tuned.ipynb`) use `learning_rate=1e-5`, a cap of 8 epochs with early stopping on validation ROC AUC, and `n_estimators=2` for fine-tuning and final inference, on 850k training rows (85% of the 1M subsample).

## 10. Computational setup

The data pipeline, Random Forest models and result analysis run locally (Python 3.12, conda environment `thesis`). The three TabPFN notebooks (`FM_zero-shot.ipynb`, `fine_tuning_epochs.ipynb`, `FM_fine_tuned.ipynb`) need an NVIDIA GPU with CUDA and were run on Kaggle. Their outputs (summary CSVs and per-flight test probabilities) were copied back into `data/final/modeling_results/fm/` and `fm_tuned/`. `results.ipynb` combines all four model variants into one comparison table, runs the DeLong tests and produces the figures.

## 11. Results

### Random Forest: subsampled layers (1M train / full 2025 test), beta = 1.2

| Layer | F1 | Precision | Recall | AUC-ROC | AUC-PR | MCC | Threshold | OOB score | Fit time |
|---|---|---|---|---|---|---|---|---|---|
| A — BTS | 0.422 | 0.313 | 0.647 | 0.673 | 0.357 | 0.208 | 0.446 | 0.686 | 166s |
| B — +weather | 0.431 | 0.331 | 0.616 | 0.686 | 0.370 | 0.226 | 0.454 | 0.708 | 194s |
| C — +GDELT | 0.424 | 0.326 | 0.605 | 0.677 | 0.357 | 0.215 | 0.453 | 0.717 | 221s |
| D — all | 0.430 | 0.337 | 0.593 | 0.686 | 0.367 | 0.226 | 0.455 | 0.723 | 241s |

### Random Forest: full dataset (6.06M train / full 2025 test), beta = 1.2

| Layer | F1 | Precision | Recall | AUC-ROC | AUC-PR | MCC | Threshold | OOB score | Fit time |
|---|---|---|---|---|---|---|---|---|---|
| A — BTS | 0.425 | 0.320 | 0.635 | 0.678 | 0.362 | 0.214 | 0.422 | 0.719 | 1,436s (~24 min) |
| B — +weather | 0.432 | 0.339 | 0.592 | 0.689 | 0.373 | 0.230 | 0.426 | 0.749 | 1,801s (~30 min) |
| C — +GDELT | 0.424 | 0.332 | 0.587 | 0.680 | 0.359 | 0.218 | 0.421 | 0.758 | 2,066s (~34 min) |
| D — all | 0.431 | 0.342 | 0.584 | 0.689 | 0.370 | 0.230 | 0.418 | 0.763 | 2,268s (~38 min) |

### TabPFN v3.5 zero-shot: subsampled layers, beta = 1.2

| Layer | F1 | Precision | Recall | AUC-ROC | AUC-PR | MCC | Threshold | Train time |
|---|---|---|---|---|---|---|---|---|
| A — BTS | 0.422 | 0.308 | 0.668 | 0.674 | 0.359 | 0.206 | 0.188 | 6,271s (~1h 45m) |
| B — +weather | 0.430 | 0.315 | 0.678 | 0.684 | 0.369 | 0.221 | 0.187 | 6,413s (~1h 47m) |
| C — +GDELT | 0.422 | 0.306 | 0.678 | 0.673 | 0.355 | 0.205 | 0.187 | 6,819s (~1h 54m) |
| D — all | 0.429 | 0.321 | 0.646 | 0.682 | 0.364 | 0.219 | 0.196 | 6,969s (~1h 56m) |

### TabPFN v3.5 fine-tuned: subsampled layers, beta = 1.2

| Layer | F1 | Precision | Recall | AUC-ROC | AUC-PR | MCC | Threshold | Epochs run (best) | Train time |
|---|---|---|---|---|---|---|---|---|---|
| A — BTS | 0.422 | 0.314 | 0.641 | 0.675 | 0.362 | 0.208 | 0.189 | 8 (7) | 15,984s (~4h 26m) |
| B — +weather | 0.432 | 0.326 | 0.637 | 0.686 | 0.371 | 0.225 | 0.193 | 8 (5) | 16,418s (~4h 34m) |
| C — +GDELT | 0.421 | 0.311 | 0.651 | 0.673 | 0.358 | 0.205 | 0.191 | 7 (6) | 16,796s (~4h 40m) |
| D — all | 0.428 | 0.321 | 0.643 | 0.682 | 0.364 | 0.219 | 0.197 | 2 (1) | 9,453s (~2h 38m) |

Train time includes fine-tuning plus final inference on the 2025 test set.

**Layer D fine-tuning stopped after 2 of the 8 allowed epochs** (best epoch 1), while A–C ran 7–8 epochs. Its validation AUC was still rising when it stopped (0.7048 → 0.7051), so the Layer D fine-tuned model is likely under-trained relative to A–C. Comparisons involving fine-tuned Layer D should be read with this in mind.

### Summary of the results

- **Weather (Layer B) gives the most consistent lift** across all four model variants: AUC-ROC improves by about 0.010–0.012 over Layer A, with matching gains in F1, AUC-PR and MCC.
- **GDELT on its own (Layer C) adds little.** For RF it adds +0.004 AUC-ROC over Layer A; for both TabPFN variants it adds nothing (±0.002).
- **Combining all sources (Layer D) does not improve on weather alone.** Layer D matches Layer B for RF and is slightly below it for TabPFN.
- **Model differences are small.** All four variants sit within about 0.01 AUC-ROC of each other on every layer. RF (Full) scores highest on every layer (0.678–0.689), but with 6× more training data its lead over the subsampled RF is only 0.003–0.005.
- **Fine-tuning adds almost nothing over zero-shot TabPFN.** The AUC-ROC differences are +0.001 (A), +0.002 (B), −0.001 (C) and 0.000 (D), at roughly 2.5× the training time. The fine-tuned model has lower recall and higher precision than zero-shot (e.g. Layer B: recall 0.637 vs. 0.678, precision 0.326 vs. 0.315), which moves its operating point toward RF's.

**Recall/precision difference between RF and TabPFN**: TabPFN's thresholds (0.187–0.197) sit at a very different point on its probability scale than RF's (0.418–0.455), and TabPFN consistently ends up with higher recall and lower precision than RF at similar F1. This is a calibration effect, not a quality difference. RF is trained with `class_weight="balanced"`, which pushes its predicted probabilities for the positive class toward 0.5. TabPFN's probabilities are not rebalanced and stay close to the true ~20% delay rate, so a much lower cutoff is needed to reach a comparable operating point. Both models are tuned to the same F-beta objective (beta = 1.2), so they reach similar F1 but resolve the precision/recall trade-off differently: zero-shot TabPFN leans furthest into recall, fine-tuned TabPFN sits in between, and RF is closest to balanced. Given that the thesis explicitly prioritizes recall, this is not a weakness of the TabPFN results; if anything they are closer to the stated objective.

### Feature importance (RF, subsampled layers, impurity-based)

`hour_sin` is the strongest single feature in every layer. Weather features (`temp_mean_12h`, `wspd_mean_12h`, `cldc_mean_12h`) enter the top 5 as soon as they are available, ahead of schedule-only features like `distance` and `crs_elapsed_time`. GDELT features only appear in the top 10 for Layer C (not Layer D), and even there they rank below both the weather and the schedule features. This is consistent with GDELT contributing a smaller, layer-dependent signal.

| Rank | Layer A | Layer B | Layer C | Layer D |
|---|---|---|---|---|
| 1 | hour_sin (.196) | hour_sin (.166) | hour_sin (.151) | hour_sin (.139) |
| 2 | origin_te (.137) | temp_mean_12h (.099) | origin_te (.058) | temp_mean_12h (.058) |
| 3 | dest_te (.123) | origin_te (.090) | dest_te (.056) | origin_te (.050) |
| 4 | distance (.107) | wspd_mean_12h (.082) | month_cos (.047) | dest_te (.050) |
| 5 | crs_elapsed_time (.105) | dest_te (.080) | crs_elapsed_time (.046) | wspd_mean_12h (.044) |

## 12. Statistical significance: DeLong's test

Pairwise AUC-ROC differences are tested with DeLong's test on the identical 2025 test set, with Bonferroni correction across the layer-pair comparisons.

Layer vs. layer, AUC-ROC difference (row minus column), all Bonferroni-significant:

| Comparison | RF | TabPFN zero-shot |
|---|---|---|
| B − A | +0.012 | +0.010 |
| C − A | +0.004 | −0.000 |
| D − A | +0.013 | +0.008 |
| B − C | +0.008 | +0.011 |
| D − B | +0.000 | −0.002 |
| D − C | +0.009 | +0.008 |

TODO: add the fine-tuned TabPFN to the layer-pair and model-pair DeLong tests once its per-flight test probabilities are in `fm_tuned/`.

DeLong's test flags essentially every pairwise comparison as significant, both model vs. model and layer vs. layer, including differences below 0.001. The clearest example is TabPFN zero-shot Layer C vs. Layer A: a difference of −0.0004 AUC-ROC is still significant after Bonferroni correction. With ~5.97M test flights and highly correlated predictions from paired models, the test detects differences far too small to matter in practice. Statistical significance therefore says little here. The results are interpreted by effect size: weather adds about 0.01 AUC-ROC consistently; GDELT, the choice of model and fine-tuning each move results by a few thousandths at most.