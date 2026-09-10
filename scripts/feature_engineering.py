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
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from config import processed_data_path, final_data_path

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
target_encode_smoothing = 10  # higher = pulls low-count airports harder toward the global rate
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


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
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

        df_fe = engineer_features(df)
        print(f"{name}{variant}: {df_fe.shape[1]} columns out")

        out_path = features_dir / f"{name}{variant}_features.parquet"
        df_fe.to_parquet(out_path, index=False)
        print(f"Saved -> {out_path}")