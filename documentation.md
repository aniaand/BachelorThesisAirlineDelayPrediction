# US Flight Delay Prediction: Modeling Documentation

## 1. Research design

The study tests whether adding external signals (weather and structured news events) improves prediction of US flight delays (`dep_del15`, departure delay ≥15 min) beyond historical flight data alone. Four information layers are compared:

- **Layer A**: BTS On-Time Performance only
- **Layer B**: BTS + Meteostat weather
- **Layer C**: BTS + GDELT news events
- **Layer D**: BTS + weather + GDELT

Each layer is run with four model variants:

- **Random Forest (RF)**, trained on the 1M-row subsample
- **Random Forest (Full)**, trained on the full 2024 training data
- **TabPFN v3.5 zero-shot**, a tabular foundation model doing in-context inference without weight updates
- **TabPFN v3.5 fine-tuned (FT)**, the same model with its weights fine-tuned on the subsample

All models train on 2024 and are evaluated on all of 2025.

## 2. Data and cleaning

### 2.1 Sample construction

| Step | Flights | Airports |
|---|---|---|
| Raw BTS data, 2024–2025 | 14,080,680 | 358 |
| After airport filter | 12,226,431 | 114 |
| Excluded: cancelled | −167,656 | |
| Excluded: diverted | −30,079 | |
| Analysis sample | 12,028,696 | 114 |
| Training set (2024) | 6,062,137 | 114 |
| Holdout set (2025) | 5,966,559 | 114 |
| Training subsample (2024, stratified) | 1,000,000 | 114 |

### 2.2 BTS flight data

- **Columns**: the raw file has 110 columns. 36 relevant ones are kept and expanded to 39 by parsing the date and adding ICAO airport codes.
- **Missing values** were investigated and are intentional, not quality problems. The cancellation-code column is 98.6% empty because it is only filled for cancelled flights, and the target is missing for under 2% of flights, all of them cancelled or diverted.
- **Airport filter**: airports are kept if their *minimum* (not average) daily flight count is at least 10. Minimum daily counts across the 358 airports range from 1 to 621 (median 2, mean 29), a heavily right-skewed distribution. The filter keeps 114 airports (31.8% of airports, 86.8% of rows). It focuses the task on operationally significant hubs and ensures enough data per airport for target encoding.
- **Cancelled and diverted flights are excluded**, because their delay outcome is undefined.
- **Only information available before scheduled departure is used.** Actual departure and arrival times, taxi times and the delay-cause breakdown are dropped, as they are only known after the outcome.
- **Processing/engineering separation**: the cleaning and merging scripts (`process_bts.py`, `process_meteostat.py`, `process_gdelt.py`, `merge_layers.py`) do no encoding or modeling transformations. Encodings are computed later in `feature_engineering.py`.

### 2.3 Meteostat weather

- Hourly observations are retrieved for the 114 airports from the nearest weather station, subset to temperature, precipitation, wind speed and cloud cover.
- Missingness is below 5% for every variable. A per-station check found a few stations with up to about 15% missing precipitation values, which was accepted.
- Variables are aggregated into **12-hour trailing windows** before the scheduled departure hour, excluding that hour itself: mean temperature, wind speed and cloud cover, and summed precipitation (`temp_mean_12h`, `wspd_mean_12h`, `cldc_mean_12h`, `prcp_sum_12h`).
- Remaining missing values in the four lookback columns (0.9–1.2%) are filled with the median of the 2024 rows of the same file. The same 2024 median is applied to 2025, so no holdout information is used.
- **Honolulu (HNL) coverage gap**: its assigned station only reports from July 2025 (22% of hours in the study window). HNL's weather features are therefore imputed for all of 2024 and the first half of 2025, so the models receive no HNL-specific weather during training. HNL was kept to avoid over-cleaning the data, and a robustness check (re-evaluating the trained models on the holdout without HNL, no retraining) is planned.

### 2.4 GDELT news events

News events come from GDELT's BigQuery table, restricted to events within 50 km of a retained airport (geocoded city/landmark level), for 2023-12-25 to 2025-12-31 (the first week is a lag buffer). Four CAMEO root codes are kept:

| Code | Category | Share of matched events |
|---|---|---|
| 17 | COERCE | ~67% |
| 18 | ASSAULT | ~20% |
| 14 | PROTEST | ~12% |
| 20 | MASS VIOLENCE | <1% |

Events are aggregated to daily airport-level counts per category plus a total, then summed over **1-, 3- and 7-day lag windows before the flight date, excluding the flight date itself** (same-day counts are not used, because they are joined by calendar date and could include events reported after departure). This gives 3 windows × 5 counts = 15 features.

GDELT counts reflect **media coverage**, not the number of discrete real-world events. A higher count means more or louder reporting near an airport, not necessarily more incidents. This is a known limitation of the data.

### 2.5 Target and descriptive patterns

- About one flight in five is delayed: 20.9% in the 2024 training set, 21.8% in the 2025 holdout and 20.9% in the stratified subsample.
- The delay rate rises from about 7% for early-morning departures to about 33% around 20:00, and peaks in summer and December. This motivates the cyclical encoding of hour and month.
- Precipitation shows the clearest link to delays: 20.1% after a dry 12-hour window (82% of flights), rising to 31.9% above 25 mm.
- Within the same airport and month, days with more GDELT coverage show a small, consistent rise in the delay rate (from about −0.8 to +0.6 percentage points across quintiles). This is descriptive only and does not show that the events cause delays.

## 3. Feature encoding

Three encoding strategies are used:

- **One-hot carrier** (`carrier_*`): 15 airlines with no natural ordering, so one-hot encoding is cheap and avoids a false ranking.
- **Target-encoded origin and destination** (`origin_te`, `dest_te`): one-hot encoding the 114 airports would add over 220 sparse columns and raise memory use and TabPFN's inference cost. Each airport is instead replaced by a smoothed average of the target (smoothing toward the global rate for low-count airports, m = 10), computed from 2024 rows only. For training rows the average is computed out-of-fold with five folds, so a flight's own outcome never enters its encoding. Holdout rows use the statistics of the full training year.
- **Sin/cos encoding** of month, day of week and scheduled departure hour (`month_sin/cos`, `dow_sin/cos`, `hour_sin/cos`): these features are cyclical, so 23:00 and 00:00 stay adjacent, without the column count of one-hot encoding.

## 4. Columns per dataset

Every layer file also carries `flight_id`, `year` and the target `dep_del15`, which are not model inputs. `flight_id` ensures the identical flights are in every subsampled layer.

| Layer | Total cols | Feature cols | Feature families |
|---|---|---|---|
| A: `layer_a_bts` | 28 | 25 | Calendar (6, sin/cos), airports (2, target-encoded), carrier (15, one-hot), scheduled elapsed time and distance (2) |
| B: `layer_b_bts_meteostat` | 32 | 29 | Layer A + 4 weather features (12-hour lookback) |
| C: `layer_c_bts_gdelt` | 43 | 40 | Layer A + 15 GDELT lag counts |
| D: `layer_d_bts_meteostat_gdelt` | 47 | 44 | Layer A + weather (4) + GDELT (15) |

The full layers have 12,028,696 rows (6,062,137 train / 5,966,559 test). The subsampled layers have 6,966,559 rows (1,000,000 subsampled train / 5,966,559 full test, unchanged).

## 5. Train/test split and subsampling

- **Temporal split**: all of 2024 for training, all of 2025 as the out-of-sample holdout. The model is trained only on the past and evaluated only on the future, as it would be deployed, with no leakage from 2025 into training.
- **Subsampling**: TabPFN v3.5 has a recommended limit on training rows, so it uses a ~1M-row subsample of the **2024 training data only**. The subsample is stratified jointly on the delay outcome and origin airport, preserving both the class balance and each airport's relative traffic volume.
- **The full 2025 test set (about 5.97M rows) is never subsampled.** Every model is scored on the identical, complete holdout.
- **RF is run on both the full data and the subsample**, which separates the effect of data volume from the effect of model architecture.
- **TabPFN training rows**: a stratified 15% validation split is held out of the 1M subsample, leaving 850,000 rows for training or in-context use (150,000 for validation).

## 6. Threshold and beta: prioritizing recall

Missing a real delay is treated as more costly than a false alarm, so every model tunes its decision threshold with an **F-beta score at beta = 1.2**, which weights recall above precision. `tune_threshold()` runs `precision_recall_curve` on probabilities that never touch the test set, computes F-beta at every candidate threshold and keeps the argmax. That threshold is then frozen and applied once to the 2025 test probabilities. Tuning the threshold also addresses the class imbalance, since the default cutoff of 0.5 is poorly suited to a target where about 20% of flights are delayed. It changes the decision cutoff only, not the model's ranking of flights, so AUC-ROC and AUC-PR are unaffected.

- **Random Forest** uses out-of-bag (OOB) probabilities: each flight is scored only by the trees that did not see it, so no separate validation split is needed. RF additionally uses `class_weight="balanced"` during training.
- **TabPFN (zero-shot and fine-tuned)** has no OOB equivalent, so the 15% validation split of the 2024 subsample is used. For the fine-tuned model the same split also drives early stopping. TabPFN has no class weighting, so only the threshold addresses imbalance.

Beta was fixed at 1.2 in advance as a moderate preference for recall. A sensitivity analysis on the subsampled RF (run afterwards, not used to choose beta) shows the trade-off on the 2025 test set:

| Layer | F1 (β = 1 / 1.2 / 1.5 / 2) | Recall (β = 1 / 1.2 / 1.5 / 2) |
|---|---|---|
| A | 0.414 / **0.422** / 0.419 / 0.404 | 0.540 / 0.647 / 0.749 / 0.871 |
| B | 0.421 / **0.431** / 0.429 / 0.410 | 0.526 / 0.616 / 0.740 / 0.875 |
| C | 0.412 / **0.424** / 0.424 / 0.409 | 0.508 / 0.605 / 0.734 / 0.864 |
| D | 0.415 / **0.430** / 0.431 / 0.414 | 0.487 / 0.593 / 0.733 / 0.859 |

β = 1.2 is within 0.001 of the highest F1 in every layer, and MCC is highest at 1.2 in three of four layers, so the recall preference costs no meaningful overall classification quality.

## 7. Hyperparameter tuning

### Random Forest

A grid over `max_depth` × `min_samples_leaf` × `n_estimators` (4 × 4 × 4 = 64 combinations, up to 500 trees) was run on a stratified 100,000-row sample of the Layer D subsample, scoring each combination by **OOB AUC-ROC** (computed from the OOB probabilities) and logging fit time. `max_features="sqrt"` and `class_weight="balanced"` were fixed.

| max_depth | min_samples_leaf | trees | OOB AUC-ROC | Fit time |
|---|---|---|---|---|
| 30 | 1 | 300 | 0.683 | 23.0 s |
| **30** | **10** | **300 (used)** | **0.703** | **20.5 s** |
| 30 | 25 | 300 | 0.705 | 18.8 s |
| 10 | 10 | 300 | 0.699 | 14.9 s |
| 20 | 10 | 500 (best in grid) | 0.707 | 32.6 s |

- **Minimum leaf size mattered most.** Leaf size 1 overfit (worst), while leaf sizes 10 and 25 were almost identical (the order flips with the number of trees), so 10 was kept as the less restrictive.
- **Depth mattered little** beyond a minimum: depth 10 slightly underfit, and depths 20, 30 and unlimited were within 0.002 at leaf size 10. Depth 30 was fixed.
- **Trees had diminishing returns**: 100 → 200 added about 0.010, 200 → 300 about 0.004, 300 → 500 about 0.003, while fit time grew by roughly two-thirds.
- The final configuration (**depth 30, leaf 10, 300 trees**) is within 0.004 of the best grid configuration. All layers and both RF tiers share it, so the gap does not affect the layer comparison.
- The full-data RF additionally uses `max_samples=0.3` (each tree sees about 1.8M rows, still more than the whole 1M subsample), because full-size bootstrap samples exceeded the available memory and time.
- The OOB score saved with the final RF runs (`oob_score` in the result tables) is scikit-learn's default OOB **accuracy** at a 0.5 cutoff. It is a diagnostic only and does not enter the threshold tuning.

### TabPFN v3.5 zero-shot

The model has no classical hyperparameters. Two settings trade accuracy against GPU time: the number of ensemble members (`n_estimators`) and the number of training rows each member uses as context (`n_inference_subsample_samples`). They were probed on 30,000 flights from Layer D (AUC-ROC, with estimated hours to predict one full layer of 5.97M flights):

| Context rows | 1 estimator | 2 estimators | 4 estimators |
|---|---|---|---|
| 10,000 | 0.6776 (0.5 h) | 0.6821 (1.0 h) | 0.6830 (2.0 h) |
| 20,000 | 0.6827 (1.0 h) | 0.6821 (2.1 h) | 0.6832 (4.2 h) |
| 40,000 | 0.6812 (2.7 h) | 0.6822 (5.4 h) | 0.6817 (10.8 h) |

Apart from the smallest configuration, everything lies within about 0.002 of the best (4 estimators, 20,000 context rows), which is within run-to-run noise. Runtime was the binding constraint, so **2 estimators with 20,000 context rows** was retained: it halves the runtime of the best configuration while keeping the larger context, which was expected to matter more than extra estimators on large training data. Predictions run in 20,000-row batches.

Consequence for the model comparison: each zero-shot estimator conditions on a random draw of 20,000 rows, so a prediction draws on at most about 40,000 training flights, while the subsampled RF is trained on all 1M rows. The comparison shows each model family under a practical compute budget, not identical exposure to the training data.

### TabPFN v3.5 fine-tuned

A separate search (`fine_tuning_epochs.ipynb`) used a stratified 100,000-row sample of Layer D (85k train / 15k validation). Five learning rates on a log scale in half-decade steps were each run for up to 30 epochs with early stopping on validation AUC-ROC (patience 8):

| Learning rate | Best val. AUC-ROC | Best epoch | Epochs run | Runtime |
|---|---|---|---|---|
| **3e-5** | **0.6986** | 19 | 28 | 68 min |
| 1e-5 | 0.6967 | 0 | 8 | 20 min |
| 3e-6 | 0.6974 | 11 | 20 | 48 min |
| 1e-6 | 0.6974 | 12 | 19 | 46 min |
| 3e-7 | 0.6970 | 0 | 8 | 20 min |

3e-5 scored highest and kept improving for 19 epochs, so the final runs use:

- `learning_rate=3e-5`, `epochs=20` (cap), `early_stopping=True`, `early_stopping_patience=8`, `eval_metric="roc_auc"`
- `n_estimators_finetune=2` and `n_estimators_final_inference=2` (the class default for final inference is 8, so it is set explicitly to match the zero-shot ensemble size)
- `n_inference_subsample_samples=20,000`, `random_state=42`
- `n_finetune_ctx_plus_query_samples=10,000` (lowered from the 50,000 default, which ran out of GPU memory on a 16 GB card)
- training on 850,000 rows with the 150,000-row validation split for early stopping and threshold tuning

## 8. Computational setup

- **Local** (Python 3.12, conda environment `thesis`): the data pipeline, both Random Forest tiers and the result analysis. RF fit times are 2–4 minutes per layer on the subsample and 24–38 minutes on the full data.
- **Kaggle GPU**: all TabPFN notebooks (zero-shot, learning-rate search, fine-tuning), since they need an NVIDIA GPU with CUDA. Zero-shot takes about 1h45 to 1h56 per layer, almost all of it test-set prediction. Fine-tuning is run as four separate notebooks, one per layer, started in parallel; each is a committed run with its own checkpoint folder. A fine-tuning run can take up to about 10 hours per layer (Layer C: 8h 7m of fine-tuning plus about 2h of prediction), close to Kaggle's 12-hour session cap.
- **Outputs** (summary CSVs, per-flight test probabilities, checkpoints) are copied back into `data/final/modeling_results/` (`rf/`, `rf_full/`, `fm/`, `fm_tuned/`). `fm_finetuned_results.ipynb` combines the four fine-tuning runs, and `results.ipynb` builds the comparison tables, DeLong tests and figures.

## 9. Results

All results are on the full 2025 holdout, with beta = 1.2.

### Average per model across the four layers

![Average metric per model, averaged across all 4 layers](avg_metrics_by_model.png)

| Model | F1 | Precision | Recall | AUC-ROC | AUC-PR | MCC |
|---|---|---|---|---|---|---|
| RF | 0.427 | 0.327 | 0.615 | 0.680 | 0.363 | 0.219 |
| TabPFN zero-shot | 0.426 | 0.310 | 0.683 | 0.678 | 0.362 | 0.213 |
| TabPFN fine-tuned | 0.426 | 0.318 | 0.645 | 0.678 | 0.358 | 0.215 |
| RF (Full) | 0.428 | 0.333 | 0.600 | 0.684 | 0.366 | 0.223 |

### Random Forest: subsampled layers (1M train)

| Layer | F1 | Precision | Recall | AUC-ROC | AUC-PR | MCC | Threshold | OOB accuracy | Fit time |
|---|---|---|---|---|---|---|---|---|---|
| A: BTS | 0.422 | 0.313 | 0.647 | 0.673 | 0.357 | 0.208 | 0.446 | 0.686 | 166s |
| B: +weather | 0.431 | 0.331 | 0.616 | 0.686 | 0.370 | 0.226 | 0.454 | 0.708 | 194s |
| C: +GDELT | 0.424 | 0.326 | 0.605 | 0.677 | 0.357 | 0.215 | 0.453 | 0.717 | 221s |
| D: all | 0.430 | 0.337 | 0.593 | 0.686 | 0.367 | 0.226 | 0.455 | 0.723 | 241s |

### Random Forest: full dataset (6.06M train)

| Layer | F1 | Precision | Recall | AUC-ROC | AUC-PR | MCC | Threshold | OOB accuracy | Fit time |
|---|---|---|---|---|---|---|---|---|---|
| A: BTS | 0.425 | 0.320 | 0.635 | 0.678 | 0.362 | 0.214 | 0.422 | 0.719 | 1,436s (~24 min) |
| B: +weather | 0.432 | 0.339 | 0.592 | 0.689 | 0.373 | 0.230 | 0.426 | 0.749 | 1,801s (~30 min) |
| C: +GDELT | 0.424 | 0.332 | 0.587 | 0.680 | 0.359 | 0.218 | 0.421 | 0.758 | 2,066s (~34 min) |
| D: all | 0.431 | 0.342 | 0.584 | 0.689 | 0.370 | 0.230 | 0.418 | 0.763 | 2,268s (~38 min) |

### TabPFN v3.5 zero-shot: subsampled layers

| Layer | F1 | Precision | Recall | AUC-ROC | AUC-PR | MCC | Threshold | Train time |
|---|---|---|---|---|---|---|---|---|
| A: BTS | 0.422 | 0.308 | 0.668 | 0.674 | 0.359 | 0.206 | 0.188 | 6,271s (~1h 45m) |
| B: +weather | 0.430 | 0.315 | 0.678 | 0.684 | 0.369 | 0.221 | 0.187 | 6,413s (~1h 47m) |
| C: +GDELT | 0.422 | 0.306 | 0.678 | 0.673 | 0.355 | 0.205 | 0.187 | 6,819s (~1h 54m) |
| D: all | 0.429 | 0.321 | 0.646 | 0.682 | 0.364 | 0.219 | 0.196 | 6,969s (~1h 56m) |

Per-layer values are from the earlier summary file; see the open items about the averaged chart.

### TabPFN v3.5 fine-tuned, final configuration (lr 3e-5, up to 20 epochs, patience 8)

| Layer | F1 | Precision | Recall | AUC-ROC | AUC-PR | MCC | Threshold | Epochs run (best) | Train time |
|---|---|---|---|---|---|---|---|---|---|
| A: BTS | pending | | | | | | | | |
| B: +weather | pending | | | | | | | | |
| C: +GDELT | 0.420 | 0.319 | 0.613 | 0.672 | 0.350 | 0.208 | 0.202 | 20 (17) | 36,286s (~10h 5m) |
| D: all | pending | | | | | | | | |

Best epoch is counted from 0. For Layer C the validation AUC-ROC rose steadily from 0.7017 (first epoch) to 0.7143 (best epoch) and the run used the full 20 epochs without early stopping. The test AUC-ROC (0.672) is nevertheless no higher than zero-shot (0.673): the gain on the 2024 validation split did not carry over to the 2025 holdout.


### Summary of the results

- **Weather (Layer B) gives the most consistent lift.** AUC-ROC improves by about 0.010–0.012 over Layer A for RF and zero-shot TabPFN, with matching gains in F1, AUC-PR and MCC.
- **GDELT on its own (Layer C) adds little**: +0.004 AUC-ROC for RF and about zero for TabPFN, below the 0.005 minimum used to call a difference meaningful (Section 10).
- **Combining all sources (Layer D) does not improve on weather alone.** Layer D matches Layer B for RF and is slightly below it for TabPFN.
- **Model differences are small.** Averaged over layers, AUC-ROC lies between 0.678 and 0.684 and F1 between 0.426 and 0.428 for all four variants. RF (Full) is highest, but its lead over the subsampled RF is only 0.003–0.005 per layer despite 6× more training data.
- **Fine-tuning adds nothing measurable over zero-shot.** The average AUC-ROC is identical (0.678). Fine-tuning Layer C under the final configuration took about 10 hours, against under 2 hours for zero-shot, with no gain on the holdout.

**Recall/precision difference between RF and TabPFN**: TabPFN's thresholds (about 0.19–0.20) sit at a very different point on its probability scale than RF's (0.42–0.46), and on average TabPFN reaches higher recall and lower precision than RF at similar F1 (zero-shot: recall 0.683, precision 0.310; fine-tuned in between; RF (Full): 0.600 and 0.333). This is a calibration effect, not a quality difference. RF is trained with `class_weight="balanced"`, which pushes its predicted probabilities for the positive class toward 0.5. TabPFN's probabilities are not rebalanced and stay close to the true ~20% delay rate, so a much lower cutoff is needed to reach a comparable operating point. All models are tuned to the same F-beta objective, so they reach similar F1 but resolve the precision/recall trade-off differently. Since the thesis prioritizes recall, this is not a weakness of the TabPFN results.


## 10. Statistical significance: DeLong's test

Pairwise AUC-ROC differences are tested with DeLong's test for correlated ROC curves on the identical 2025 test set. A Bonferroni correction is applied within each family of comparisons (α = 0.05 divided by the number of tests).

With almost six million test flights, even negligible differences become significant. A difference is therefore only called **meaningful if it is significant after correction and at least 0.005 AUC-ROC**. That minimum is more than twice the variation between near-best hyperparameter configurations (at most about 0.002).

Layer vs. layer, AUC-ROC difference (first minus second), all Bonferroni-significant:

| Comparison | RF | TabPFN zero-shot | Meaningful (≥ 0.005)? |
|---|---|---|---|
| B − A | +0.012 | +0.010 | yes |
| C − A | +0.004 | −0.000 | no |
| D − A | +0.013 | +0.008 | yes |
| B − C | +0.008 | +0.011 | yes |
| D − B | +0.000 | −0.002 | no |
| D − C | +0.009 | +0.008 | yes |

Significance flags nearly every comparison, including a difference of −0.0004 (TabPFN zero-shot, Layer C vs. A), which is why the results are read by effect size. Weather adds about 0.01 AUC-ROC consistently; GDELT, the choice of model and fine-tuning each move results by a few thousandths at most.