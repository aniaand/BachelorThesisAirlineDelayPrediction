'''
PROMPT:

Write a Python script to process raw Meteostat hourly weather data (previously downloaded and cached by a separate download script) for a flight-delay-prediction pipeline. It should:

Load the combined raw hourly weather file.
Print every column available in the raw data, so the schema can be audited against expectations.
Subset the data to only the core weather variables needed (temperature, precipitation, wind speed, cloud cover) plus join/reference keys (timestamp, airport code, station id, station distance from airport).
Report the percentage of missing values per column, across all stations combined.
Separately, report missingness per station for the core weather variables — since overall column-level missingness can hide the fact that a subset of stations barely report a given variable (e.g. cloud cover) while others report it fine. Flag (don't auto-drop) any station whose missingness on any core variable exceeds a threshold, so exclusion/imputation decisions can be made deliberately later rather than baked in here.
Save the subset, audited result to a processed-data location.

The script must not impute, aggregate, resample, or otherwise transform any weather values — this is a load/audit/subset/save pipeline only, mirroring the equivalent BTS processing script. Use snake_case naming, # %% cell markers for VS Code, and pull paths from the project's config.py rather than hardcoding them.
'''
# 7.09.2026 21:45 CET
# Author: Anna Andruszkiewicz (code and adjustments), Claude Sonnet 5 (code)

"""
process_meteostat.py

Stage 1 cleaning/audit pass over raw Meteostat hourly weather data
(downloaded by download_meteostat.py).

Pipeline:
    1. Load the combined raw hourly weather file
    2. Audit available columns
    3. Subset to the columns needed for the thesis
    4. Audit missingness per column (overall)
    5. Audit missingness per station (some stations may be too sparse to use)
    6. Save the cleaned/subset result

This script only loads, audits, subsets, and saves -- it does not impute,
aggregate, or otherwise transform values. That happens later, at the merge
and feature-engineering stages.

Run cell-by-cell in VS Code (Code Runner respects the "# %%" markers) or as a
plain script: `python process_meteostat.py`.
"""

# %% Imports and config ---------------------------------------------------
from pathlib import Path

import pandas as pd

from config import raw_data_path, processed_data_path

weather_raw_dir = Path(raw_data_path) / "weather"
weather_processed_dir = Path(processed_data_path) / "weather"
weather_processed_dir.mkdir(parents=True, exist_ok=True)

# Core weather variables needed for the thesis (Meteostat hourly schema):
#   temp = air temperature (°C), prcp = precipitation (mm),
#   wspd = wind speed (km/h), cldc = cloud cover (%)
# Kept alongside join/reference keys (time, iata, station_id, distance).
keep_cols = [
    "time",
    "iata",
    "station_id",
    "station_distance_km",
    "temp",
    "prcp",
    "wspd",
    "cldc",
]

# A station whose core variables are missing above this share of the time
# is flagged as too sparse to be a reliable weather proxy for its airport.
station_missingness_flag_threshold = 0.30


# %% 1. Load raw combined weather file --------------------------------------
raw_path = weather_raw_dir / "weather_hourly_raw.parquet"
weather_raw = pd.read_parquet(raw_path)
print(f"Loaded {len(weather_raw):,} rows, {weather_raw['iata'].nunique()} airports.")

# %% 2. Audit available columns ----------------------------------------------
print(f"\n{len(weather_raw.columns)} columns available:")
for col in weather_raw.columns:
    print(f"  - {col}")

missing_from_keep = [c for c in keep_cols if c not in weather_raw.columns]
if missing_from_keep:
    print(f"\nWARNING: expected columns not found in raw data: {missing_from_keep}")

# %% 3. Subset to needed columns ----------------------------------------------
present_keep_cols = [c for c in keep_cols if c in weather_raw.columns]
weather = weather_raw[present_keep_cols].copy()
print(f"\nSubset to {len(weather.columns)} columns, {len(weather):,} rows.")

# %% 4. Missingness audit (overall, per column) -------------------------------
missing_pct = (weather.isna().mean() * 100).round(2).sort_values(ascending=False)
print("\nMissing value % per column (all stations combined):")
print(missing_pct.to_string())

# %% 5. Missingness audit (per station) ----------------------------------------
core_vars = [c for c in ["temp", "prcp", "wspd", "cldc"] if c in weather.columns]

station_missingness = (
    weather.groupby("iata")[core_vars]
    .apply(lambda g: g.isna().mean())
    .round(3)
)
station_missingness["n_obs"] = weather.groupby("iata").size()

print(f"\nPer-station missingness for core variables {core_vars}:")
print(station_missingness.sort_values(core_vars[0], ascending=False).to_string())

flagged_stations = station_missingness[
    (station_missingness[core_vars] > station_missingness_flag_threshold).any(axis=1)
]
print(
    f"\n{len(flagged_stations)} / {len(station_missingness)} airports have "
    f">= {station_missingness_flag_threshold:.0%} missingness in at least one "
    f"core variable:"
)
if len(flagged_stations):
    print(flagged_stations.index.tolist())

# %% 6. Save --------------------------------------------------------------------
output_path = weather_processed_dir / "weather_processed.parquet"
weather.to_parquet(output_path, index=False)
print(f"\nSaved {len(weather):,} rows, {len(weather.columns)} columns to {output_path}")