'''
Write a Python script that merges the processed BTS, Meteostat, and GDELT outputs
(each already produced by its own load/audit/filter/save processing script) into the
four nested information-layer datasets used for the thesis's layer-ablation design:
Layer A (BTS only), Layer B (BTS + Meteostat), Layer C (BTS + GDELT), and Layer D
(BTS + Meteostat + GDELT). This script does joins and the aggregations required to
make those joins possible at the flight level; it does not do feature engineering
(no target encoding, no cyclical encoding, no imputation) -- that stays in a separate
feature_engineering.py that runs afterward. It should:

1. Load the processed BTS, Meteostat, and GDELT daily-count parquet files.
2. Build Layer A from BTS: drop cancelled and diverted flights, subset to only
   columns knowable before departure (drop everything post-outcome -- arrival
   actuals, taxi/wheels times, delay-cause breakdown), rename all columns to
   snake_case, assign a stable flight_id, and cast the target (departure delay
   >= 15 min) to a compact int dtype. Report how many rows were dropped and why
   at each step.
3. Build a join key by flooring each flight's scheduled departure time to the hour.
4. Aggregate the hourly Meteostat data into a trailing lookback window immediately
   before that scheduled hour (a configurable number of hours, defaulting to 12,
   excluding the departure hour itself so only weather that had already occurred by
   prediction time is used) -- mean temperature/wind/cloud cover, summed
   precipitation, per origin airport. Join this onto Layer A to form Layer B, and
   report the match rate.
5. From the GDELT daily (airport, date) count table, compute trailing rolling-sum
   lag windows over a configurable list of day counts (defaulting to 1, 3, and 7
   days), excluding the current day, per origin airport and per count column
   (each CAMEO category plus the total). Keep the same-day counts as well.
6. Join the same-day + lagged GDELT counts onto Layer A and Layer B (by origin
   airport and flight date) to form Layer C and Layer D.
7. Save all four full (unsampled) layer datasets to a merged-data output directory,
   dropping any columns that only existed to support the joins (e.g. the
   departure-hour join key).

Use snake_case naming, # %% cell markers for VS Code, and pull paths from the
central config.py rather than hardcoding them. Subsampling each of the four layers
into full and subsampled versions is a separate concern and happens in a later
subsample.py script, not here.
'''
# 9.09.2026 22:15 CET
# Author: Anna Andruszkiewicz (code and adjustments), Claude Sonnet 5 (code)

# %% Imports & config
from pathlib import Path

import pandas as pd
from config import processed_data_path

processed_data_path = Path(processed_data_path)

# How many hours of weather history (strictly before scheduled departure) to
# aggregate per flight. 
lookback_hours = 12

# %% Load processed layer inputs
bts = pd.read_parquet(processed_data_path / "bts" / "bts_processed.parquet")
weather = pd.read_parquet(processed_data_path / "weather" / "weather_processed.parquet")

print(f"BTS raw:     {bts.shape}")
print(f"Weather raw: {weather.shape}")

# %% Layer A base — pre-departure-known BTS columns only, target = dep_del15
# Excluded: everything only known after departure/arrival (ArrTime, ArrDelay, ArrDel15,
# TaxiOut/In, WheelsOff/On, ActualElapsedTime, AirTime, delay-cause breakdown, DepTime,
# DepDelay).
keep_cols = {
    "FlightDate": "flight_date",
    "Year": "year",
    "Month": "month",
    "DayOfWeek": "day_of_week",
    "Reporting_Airline": "reporting_airline",
    "Flight_Number_Reporting_Airline": "flight_number",
    "Origin": "origin",
    "OriginCityName": "origin_city",
    "OriginState": "origin_state",
    "Dest": "dest",
    "DestCityName": "dest_city",
    "DestState": "dest_state",
    "CRSDepTime": "crs_dep_time",
    "CRSElapsedTime": "crs_elapsed_time",
    "Distance": "distance",
    "DepDel15": "dep_del15",
}


def build_layer_a(bts: pd.DataFrame) -> pd.DataFrame:
    before = len(bts)
    df = bts[(bts["Cancelled"] == 0) & (bts["Diverted"] == 0)].copy()
    dropped = before - len(df)
    print(f"Dropped {dropped} cancelled/diverted flights ({dropped / before:.2%})")

    df = df[list(keep_cols.keys())].rename(columns=keep_cols)

    n_missing_target = df["dep_del15"].isna().sum()
    if n_missing_target:
        print(f"Dropping {n_missing_target} rows with missing dep_del15")
        df = df.dropna(subset=["dep_del15"])
    df["dep_del15"] = df["dep_del15"].astype("int8")

    df = df.reset_index(drop=True)
    # Stable ID assigned once, before any merges, so it carries unchanged through
    # Layer B/C/D and lets subsample.py select the same flights across all 4 layers
    # by ID rather than relying on row order/position staying aligned.
    df.insert(0, "flight_id", df.index.astype("int64"))
    return df


layer_a = build_layer_a(bts)
print(f"Layer A (BTS only): {layer_a.shape}")

# %% Join key — floor scheduled departure time to the hour
# crs_dep_time is HHMM (e.g. 830 -> 8:30, 2359 -> 23:59). BTS sometimes uses 2400 for
# midnight; clip to hour 23 rather than rolling into the next day.
def add_dep_hour_key(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    hour = (df["crs_dep_time"] // 100).clip(upper=23)
    df["dep_datetime_hour"] = pd.to_datetime(df["flight_date"]) + pd.to_timedelta(hour, unit="h")
    return df


layer_a = add_dep_hour_key(layer_a)

# %% Layer B — + Meteostat, aggregated over the trailing LOOKBACK_HOURS window
# (mean temp/wind/cloud cover, summed precip) per origin airport, excluding the
# scheduled departure hour itself (closed="left").
def build_weather_lookback(weather: pd.DataFrame, lookback_hours: int) -> pd.DataFrame:
    w = weather.sort_values(["iata", "time"]).set_index("time")
    window = f"{lookback_hours}h"
    grouped = w.groupby("iata")

    suffix = f"_{lookback_hours}h"
    lookback = pd.DataFrame(index=w.index)
    lookback["iata"] = w["iata"]
    lookback[f"temp_mean{suffix}"] = (
        grouped["temp"].rolling(window, closed="left").mean().reset_index(level=0, drop=True)
    )
    lookback[f"prcp_sum{suffix}"] = (
        grouped["prcp"].rolling(window, closed="left").sum().reset_index(level=0, drop=True)
    )
    lookback[f"wspd_mean{suffix}"] = (
        grouped["wspd"].rolling(window, closed="left").mean().reset_index(level=0, drop=True)
    )
    lookback[f"cldc_mean{suffix}"] = (
        grouped["cldc"].rolling(window, closed="left").mean().reset_index(level=0, drop=True)
    )

    return lookback.reset_index().rename(columns={"time": "dep_datetime_hour", "iata": "origin"})


weather_lookback = build_weather_lookback(weather, lookback_hours)


def build_layer_b(layer_a: pd.DataFrame, weather_lookback: pd.DataFrame) -> pd.DataFrame:
    df = layer_a.merge(weather_lookback, on=["origin", "dep_datetime_hour"], how="left")
    matched = df[f"temp_mean_{lookback_hours}h"].notna().mean()
    print(f"Weather match rate ({lookback_hours}h lookback): {matched:.2%}")
    return df


layer_b = build_layer_b(layer_a, weather_lookback)
print(f"Layer B (BTS + Meteostat, {lookback_hours}h lookback): {layer_b.shape}")

# %% GDELT daily counts — already aggregated to (airport, date) by process_gdelt.py
gdelt_daily = pd.read_parquet(processed_data_path / "gdelt" / "gdelt_daily_counts.parquet")
gdelt_daily = gdelt_daily.rename(columns={"airport_iata": "origin", "event_date": "flight_date"})
print(f"GDELT daily counts (incl. buffer window): {gdelt_daily.shape}")

# %% Complete daily grid — gdelt_daily only has rows for days with >=1 event, so a
# genuinely quiet day (zero events across all categories) is a missing row, not a
# zero row. Reindexing to every (airport, date) combo makes those quiet days real
# zeros, which matters for the lag windows below: without this, a quiet day today
# would lose any real lag signal from active days just before it once NaNs get
# filled to 0 after the merge onto flights.
gdelt_count_cols = [c for c in gdelt_daily.columns if c.startswith("gdelt_")]

full_dates = pd.date_range(gdelt_daily["flight_date"].min(), gdelt_daily["flight_date"].max(), freq="D")
full_index = pd.MultiIndex.from_product(
    [gdelt_daily["origin"].unique(), full_dates], names=["origin", "flight_date"]
)
gdelt_daily = (
    gdelt_daily.set_index(["origin", "flight_date"])
    .reindex(full_index, fill_value=0)
    .reset_index()
    .sort_values(["origin", "flight_date"])
    .reset_index(drop=True)
)
print(f"GDELT daily counts, complete grid: {gdelt_daily.shape}")

# %% Lag windows — trailing rolling sums per airport, excluding the current day.
# Same-day counts (above) capture same-day coverage; these capture coverage building
# up over the preceding 1/3/7 days. The 7-day buffer in the raw download exists
# specifically so the 7-day window is fully covered from day one of the study period.
# Integer-window rolling (fast, vectorized) rather than a time-offset window ("7D"),
# which is a known slow path in pandas — the complete daily grid above makes an
# integer window equivalent to a calendar window. shift(1) excludes the current day.
lag_windows_days = [1, 3, 7]

grouped = gdelt_daily.groupby("origin")[gdelt_count_cols]
lag_frames = []
for window_days in lag_windows_days:
    rolled = grouped.rolling(window=window_days, min_periods=1).sum().reset_index(level=0, drop=True)
    lagged = rolled.groupby(gdelt_daily["origin"]).shift(1)
    lagged = lagged.add_suffix(f"_lag{window_days}d")
    lag_frames.append(lagged)

gdelt_daily = pd.concat([gdelt_daily] + lag_frames, axis=1)
lag_cols = [c for c in gdelt_daily.columns if "_lag" in c]
gdelt_daily[lag_cols] = gdelt_daily[lag_cols].fillna(0).astype("int32")
print(f"GDELT daily counts + {lag_windows_days}-day lags: {gdelt_daily.shape}")

# %% Layer C / D — join daily GDELT counts onto Layer A / Layer B (same-day, origin airport)
def add_gdelt(df: pd.DataFrame, gdelt_daily: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["flight_date"] = pd.to_datetime(df["flight_date"])
    gdelt_daily = gdelt_daily.copy()
    gdelt_daily["flight_date"] = pd.to_datetime(gdelt_daily["flight_date"])

    df = df.merge(gdelt_daily, on=["origin", "flight_date"], how="left")
    count_cols = [c for c in df.columns if c.startswith("gdelt_")]
    df[count_cols] = df[count_cols].fillna(0).astype("int32")
    return df


layer_c = add_gdelt(layer_a, gdelt_daily)  # BTS + GDELT
layer_d = add_gdelt(layer_b, gdelt_daily)  # BTS + Meteostat + GDELT

print(f"Layer C (BTS + GDELT): {layer_c.shape}")
print(f"Layer D (BTS + Meteostat + GDELT): {layer_d.shape}")

# %% Save the 4 full (unsampled) stages — subsampling happens in subsample.py
out_dir = processed_data_path / "merged"
out_dir.mkdir(parents=True, exist_ok=True)

stages = [
    ("layer_a_bts", layer_a),
    ("layer_b_bts_meteostat", layer_b),
    ("layer_c_bts_gdelt", layer_c),
    ("layer_d_bts_meteostat_gdelt", layer_d),
]

for name, df in stages:
    df = df.drop(columns=["dep_datetime_hour"], errors="ignore")
    path = out_dir / f"{name}.parquet"
    df.to_parquet(path, index=False)
    print(f"Saved {name}: {df.shape} -> {path}")