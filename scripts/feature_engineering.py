'''
Write a Python script that takes the merged layer datasets (4 layers x full/
subsampled = 8 files, from merge_layers.py and subsample.py) and turns them into
model-ready feature matrices. It should:

1. Load each of the 8 merged parquet files in turn.
2. Cyclically encode month, day_of_week, and scheduled departure hour (derived
   from crs_dep_time) as sin/cos pairs, instead of one-hot, to avoid exploding
   column count and to correctly represent adjacency (December/January,
   hour 23/hour 0).
3. Target-encode origin and dest airport, with light smoothing toward the global
   rate for low-count categories, fit on that file's own 2024 rows only. Unseen
   categories fall back to the train global mean.
4. One-hot encode reporting_airline.
5. Impute missing weather lookback columns with the
   median from that file's own 2024 rows.
6. Drop columns superseded by the above (raw month/day_of_week/crs_dep_time,
   raw origin/dest/reporting_airline strings) and columns that were only ever
   identifiers or redundant with what's now encoded (flight_date, origin_city,
   origin_state, dest_city, dest_state, flight_number) for traceability.
7. Save each result to a separate features output directory, report the
   before/after column count per file.

Use snake_case naming, # %% cell markers for VS Code, and pull paths from the
central config.py rather than hardcoding them. No class-imbalance handling here
(class weighting happens at modeling.py) and no scaling/normalization (not needed
for the tree-based models or TabPFN).
'''
# 10.09.2026 CET
# Author: Anna Andruszkiewicz (code and adjustments), Claude Sonnet 5 (code)

# %% Imports & config
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from config import processed_data_path, final_data_path
from appendix_utils import save_appendix_table

processed_data_path = Path(processed_data_path)
final_data_path = Path(final_data_path)
merged_dir = processed_data_path / "merged"
features_dir = final_data_path
features_dir.mkdir(parents=True, exist_ok=True)

layer_files = [
    "layer_a_bts",
    "layer_b_bts_meteostat",
    "layer_c_bts_gdelt",
    "layer_d_bts_meteostat_gdelt",
]
variants = ["", "_subsampled"]

target_col = "dep_del15"
train_year = 2024
target_encode_cols = ["origin", "dest"]
target_encode_smoothing = 10
target_encode_folds = 5  # K-fold out-of-fold encoding for training rows, to avoid self-leakage
random_state = 42

drop_cols_always = [
    "flight_date",
    "origin_city",
    "origin_state",
    "dest_city",
    "dest_state",
    "flight_number",
]

imputation_log = []
feature_sets = {}


# %% Helpers
def add_cyclical(df: pd.DataFrame, col: str, period: int, prefix: str) -> pd.DataFrame:
    radians = 2 * np.pi * df[col] / period
    df[f"{prefix}_sin"] = np.sin(radians)
    df[f"{prefix}_cos"] = np.cos(radians)
    return df


def target_encode(df: pd.DataFrame, col: str, train_mask: pd.Series, smoothing: float) -> pd.Series:
    encoded = pd.Series(index=df.index, dtype="float64")
 
    # Full-2024 stats: used for test (2025) rows, and as the fallback for any
    # category unseen at encoding time. This part was already leak-free.
    global_mean = df.loc[train_mask, target_col].mean()
    full_stats = df.loc[train_mask].groupby(col)[target_col].agg(["mean", "count"])
    full_smoothed = (
        full_stats["mean"] * full_stats["count"] + global_mean * smoothing
    ) / (full_stats["count"] + smoothing)
    encoded.loc[~train_mask] = df.loc[~train_mask, col].map(full_smoothed)
 
    # Training rows: K-fold out-of-fold encoding, so a row's own label never
    # contributes to the statistic used to encode it (avoids self-leakage).
    train_idx = df.index[train_mask]
    kf = KFold(n_splits=target_encode_folds, shuffle=True, random_state=random_state)
    for fit_pos, holdout_pos in kf.split(train_idx):
        fit_idx = train_idx[fit_pos]
        holdout_idx = train_idx[holdout_pos]
 
        fold_mean = df.loc[fit_idx, target_col].mean()
        fold_stats = df.loc[fit_idx].groupby(col)[target_col].agg(["mean", "count"])
        fold_smoothed = (
            fold_stats["mean"] * fold_stats["count"] + fold_mean * smoothing
        ) / (fold_stats["count"] + smoothing)
        encoded.loc[holdout_idx] = df.loc[holdout_idx, col].map(fold_smoothed)
 
    encoded = encoded.fillna(global_mean)  # categories unseen in the relevant fold/file
    return encoded


def engineer_features(df: pd.DataFrame, file_name: str = "") -> pd.DataFrame:
    df = df.copy()
    train_mask = df["year"] == train_year

    # Calendar cyclical encodings
    df = add_cyclical(df, "month", 12, "month")
    df = add_cyclical(df, "day_of_week", 7, "dow")

    df["dep_hour"] = (df["crs_dep_time"] // 100).clip(upper=23)
    df = add_cyclical(df, "dep_hour", 24, "hour")
    df = df.drop(columns=["dep_hour"])

    # Airport target encoding, fit on this file's own 2024 rows
    for col in target_encode_cols:
        df[f"{col}_te"] = target_encode(df, col, train_mask, target_encode_smoothing)

    # Carrier one-hot
    carrier_dummies = pd.get_dummies(df["reporting_airline"], prefix="carrier", dtype="int8")
    df = pd.concat([df, carrier_dummies], axis=1)

    # Weather missing-value imputation (Layer B/D only), median from this file's 2024 rows
    weather_cols = [
        c for c in df.columns if c.startswith(("temp_mean", "prcp_sum", "wspd_mean", "cldc_mean"))
    ]
    for col in weather_cols:
        n_missing = df[col].isna().sum()
        if n_missing:
            median = df.loc[train_mask, col].median()
            print(f"  Imputing {n_missing} missing {col} with 2024 train median {median:.2f}")
            imputation_log.append({
                "file": file_name,
                "variable": col,
                "n_missing": int(n_missing),
                "missing_pct": 100 * n_missing / len(df),
                "train_median": median,
            })
            df[col] = df[col].fillna(median)

    drop_cols = drop_cols_always + [
        "month", "day_of_week", "crs_dep_time", "origin", "dest", "reporting_airline",
    ]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    return df


# %% Apply to all 8 files
for name in layer_files:
    for variant in variants:
        in_path = merged_dir / f"{name}{variant}.parquet"
        df = pd.read_parquet(in_path)
        print(f"\n{name}{variant}: {df.shape[1]} columns in, {df.shape[0]:,} rows")

        df_fe = engineer_features(df, file_name=f"{name}{variant}")
        if variant == "":
            feature_sets[name] = list(df_fe.columns)
        print(f"{name}{variant}: {df_fe.shape[1]} columns out")

        out_path = features_dir / f"{name}{variant}_features.parquet"
        df_fe.to_parquet(out_path, index=False)
        print(f"Saved -> {out_path}")


# %% Appendix B.1 -- full feature list by layer ----------------------------------
# Built from the full-data feature files; the subsampled files have the same columns.
def describe_feature(col: str) -> tuple[str, str]:
    """Return (source variable, transformation) for a feature column."""
    if col == "flight_id":
        return "flight_id", "Identifier, not a model input"
    if col == "year":
        return "year", "Train/holdout split only, not a model input"
    if col == target_col:
        return target_col, "Target"
    cyclical = {"month": ("month", 12), "dow": ("day_of_week", 7), "hour": ("crs_dep_time (hour)", 24)}
    if (m := re.fullmatch(r"(month|dow|hour)_(sin|cos)", col)):
        source, period = cyclical[m.group(1)]
        return source, f"Cyclical {m.group(2)} encoding, period {period}"
    if col in ("origin_te", "dest_te"):
        return col[:-3], (
            f"Smoothed target encoding (m = {target_encode_smoothing}); "
            f"{target_encode_folds}-fold out-of-fold on 2024, full-2024 mapping for 2025"
        )
    if col.startswith("carrier_"):
        return "reporting_airline", "One-hot indicator"
    if (m := re.fullmatch(r"(temp|prcp|wspd|cldc)_(mean|sum)_(\d+)h", col)):
        return m.group(1), f"{m.group(3)} h trailing {m.group(2)} (merge stage); missing imputed with 2024 median"
    if (m := re.fullmatch(r"gdelt_(.+)_count_lag(\d+)d", col)):
        return f"GDELT {m.group(1)} count", f"Trailing {m.group(2)}-day sum excluding flight date (merge stage)"
    if col in ("crs_elapsed_time", "distance"):
        return col, "None (used as is)"
    print(f"WARNING: no feature description for {col}")
    return col, ""


layer_letters = dict(zip(layer_files, ["A", "B", "C", "D"]))
all_features = list(dict.fromkeys(c for name in layer_files for c in feature_sets[name]))

feature_rows = []
for col in all_features:
    source, transformation = describe_feature(col)
    row = {"feature": col, "source_variable": source, "transformation": transformation}
    for name, letter in layer_letters.items():
        row[letter] = "x" if col in feature_sets[name] else ""
    feature_rows.append(row)

b1_features = pd.DataFrame(feature_rows)
save_appendix_table(
    b1_features,
    "b1_feature_list",
    "Model features, their source variables and transformations, and the information "
    "layers (A: BTS, B: + Meteostat, C: + GDELT, D: all) that contain them.",
    longtable=True,
)

# %% Appendix B.1b -- weather imputation values per file ---------------------------
if imputation_log:
    save_appendix_table(
        pd.DataFrame(imputation_log),
        "b1b_weather_imputation",
        "Missing weather values imputed with the 2024 training median, per feature file.",
        longtable=True,
    )