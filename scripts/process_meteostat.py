'''
PROMPT:

Write a Python script to process raw Meteostat hourly weather data (previously downloaded and cached by a separate download script) for a flight-delay-prediction pipeline. It should:

1. Load the combined raw hourly weather file.
2. Print every column available in the raw data, so the schema can be audited against expectations.
3. Subset the data to only the core weather variables needed (temperature, precipitation, wind speed, cloud cover) plus join/reference keys (timestamp, airport code, station id, station distance from airport).
4. Report the percentage of missing values per column, across all stations combined.
5. Separately, report missingness per station for the core weather variables — since overall column-level missingness can hide the fact that a subset of stations barely report a given variable (e.g. cloud cover) while others report it fine. Flag (don't auto-drop) any station whose missingness on any core variable exceeds a threshold, so exclusion/imputation decisions can be made deliberately later rather than baked in here.
6. Save the subset, audited result to a processed-data location.

The script must not impute, aggregate, resample, or otherwise transform any weather values — this is a load/audit/subset/save pipeline only, mirroring the equivalent BTS processing script. Use snake_case naming, # %% cell markers for VS Code, and pull paths from the project's config.py rather than hardcoding them.
'''
# 7.09.2026 21:45 CET   original version
# 4.10.2026             rewrite: coverage-aware flagging + data-quality checks
# Author: Anna Andruszkiewicz (code and adjustments), Claude Sonnet 5 / Claude Opus 5.5 (code)

"""
process_meteostat.py

Stage 1 cleaning/audit pass over raw Meteostat hourly weather data
(downloaded by download_meteostat.py).

Pipeline:
    1. Load the combined raw hourly weather file
    2. Audit available columns (incl. any provenance / model-data columns)
    3. Subset to the columns needed for the thesis
    4. Structural checks: download window, duplicate rows, stations per airport
    5. Missingness audit per column (overall)
    6. Missingness audit per station -- measured against the FULL download
       window, so that hours with no row at all count as missing
    7. Data-quality checks
        7a. Observation cadence (hourly vs. synoptic reporting, e.g. HNL)
        7b. Shared missing-hour patterns (source-level outages)
        7c. Duplicated / near-identical series between stations
    8. Appendix A.3 table
    9. Save the subset result and the station audit

Fix vs. the previous version: missingness used to be computed only over rows
that exist. A station that simply has no row for most hours (HNL: ~22% of
hours) therefore looked complete and was never flagged. Missingness is now
measured against the expected number of hours in the download window.

This script only loads, audits, subsets, and saves -- it does not impute,
aggregate, or otherwise transform values. That happens later, at the merge
and feature-engineering stages.

Run cell-by-cell in VS Code (Code Runner respects the "# %%" markers) or as a
plain script: `python process_meteostat.py`.
"""

# %% Imports and config ---------------------------------------------------
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from config import raw_data_path, processed_data_path
from appendix_utils import save_appendix_table

weather_raw_dir = Path(raw_data_path) / "weather"
weather_processed_dir = Path(processed_data_path) / "weather"
weather_processed_dir.mkdir(parents=True, exist_ok=True)

# Core weather variables needed for the thesis (Meteostat hourly schema):
#   temp = air temperature (°C), prcp = precipitation (mm),
#   wspd = wind speed (km/h), cldc = cloud cover (%)
# Kept alongside join/reference keys (time, iata, station_id, distance).
core_vars_expected = ["temp", "prcp", "wspd", "cldc"]
key_cols = ["time", "iata", "station_id", "station_distance_km"]
keep_cols = key_cols + core_vars_expected

# A station is flagged as sparse if, measured over the full download window,
# any core variable is missing for more than this share of hours.
station_missingness_flag_threshold = 0.30

# Data-quality check settings
cadence_flag_median_gap_h = 1.5        # median gap above this = not hourly reporting
duplicate_corr_threshold = 0.999       # temp correlation above this between two stations
duplicate_identical_share = 0.50       # or: identical temp in more than this share of hours
max_runs_to_print = 10                 # missing-hour runs printed per group

# Keywords that suggest provenance / model-fill columns in the raw download
provenance_keywords = ["source", "flag", "model", "src", "qc", "quality"]


def summarise_runs(timestamps: pd.DatetimeIndex) -> pd.DataFrame:
    """Collapse a sorted set of hourly timestamps into contiguous runs."""
    if len(timestamps) == 0:
        return pd.DataFrame(columns=["start", "end", "hours"])
    ts = pd.Series(timestamps.sort_values())
    run_id = (ts.diff() != pd.Timedelta(hours=1)).cumsum()
    runs = ts.groupby(run_id).agg(start="min", end="max", hours="size")
    return runs.reset_index(drop=True)


# %% 1. Load raw combined weather file --------------------------------------
raw_path = weather_raw_dir / "weather_hourly_raw.parquet"
weather_raw = pd.read_parquet(raw_path)
print(f"Loaded {len(weather_raw):,} rows, {weather_raw['iata'].nunique()} airports.")

# %% 2. Audit available columns ----------------------------------------------
print(f"\n{len(weather_raw.columns)} columns available:")
for col in weather_raw.columns:
    print(f"  - {col} ({weather_raw[col].dtype})")

missing_from_keep = [c for c in keep_cols if c not in weather_raw.columns]
if missing_from_keep:
    print(f"\nWARNING: expected columns not found in raw data: {missing_from_keep}")

# Provenance check: is there any column that says whether a value was observed
# or model-filled? If not, provenance has to be checked in download_meteostat.py.
provenance_cols = [
    c for c in weather_raw.columns
    if any(k in c.lower() for k in provenance_keywords)
]
if provenance_cols:
    print(f"\nPossible provenance columns found: {provenance_cols}")
    for c in provenance_cols:
        print(f"\n  Value counts for '{c}' (top 10):")
        print(weather_raw[c].value_counts(dropna=False).head(10).to_string())
else:
    print(
        "\nNo provenance/source/model columns in the raw data. Whether values are "
        "observed or model-filled cannot be determined here -- check the settings "
        "used in download_meteostat.py (e.g. whether model data was enabled)."
    )

# %% 3. Subset to needed columns ----------------------------------------------
present_keep_cols = [c for c in keep_cols if c in weather_raw.columns]
weather = weather_raw[present_keep_cols].copy()
core_vars = [c for c in core_vars_expected if c in weather.columns]
print(f"\nSubset to {len(weather.columns)} columns, {len(weather):,} rows.")

# %% 4. Structural checks -------------------------------------------------------
# 4a. Download window. The expected-hours count is based on this window.
window_start, window_end = weather["time"].min(), weather["time"].max()
expected_hours = int((window_end - window_start) / pd.Timedelta(hours=1)) + 1
full_index = pd.date_range(window_start, window_end, freq="h")
print(f"\nDownload window: {window_start} -> {window_end}")
print(f"  timezone: {getattr(weather['time'].dt, 'tz', None)}")
print(f"  expected hours: {expected_hours:,}")

# 4b. Duplicate rows per airport-hour would inflate row-based statistics.
dup_mask = weather.duplicated(subset=["iata", "time"], keep=False)
print(f"\nDuplicate airport-hour rows: {dup_mask.sum():,}")
if dup_mask.any():
    print(weather.loc[dup_mask].groupby("iata").size().sort_values(ascending=False).head(10).to_string())

# 4c. Each airport should map to exactly one station.
stations_per_airport = weather.groupby("iata")["station_id"].nunique()
multi_station = stations_per_airport[stations_per_airport > 1]
print(f"\nAirports with more than one station: {len(multi_station)}")
if len(multi_station):
    print(multi_station.to_string())

# 4d. Each station should serve exactly one airport (otherwise two airports
# share identical weather by construction).
airports_per_station = weather.groupby("station_id")["iata"].nunique()
shared_stations = airports_per_station[airports_per_station > 1]
print(f"Stations shared by more than one airport: {len(shared_stations)}")
if len(shared_stations):
    print(shared_stations.to_string())

# %% 5. Missingness audit (overall, per column) -------------------------------
# Row-based: only rows that exist. Missing hours (no row) are covered in step 6.
missing_pct = (weather.isna().mean() * 100).round(2).sort_values(ascending=False)
print("\nMissing value % per column (existing rows only, all stations combined):")
print(missing_pct.to_string())

# %% 6. Missingness audit (per station, against the full window) -------------
hours_observed = weather.groupby("iata")["time"].nunique()
non_missing_counts = weather[core_vars].notna().groupby(weather["iata"]).sum()

station_audit = pd.DataFrame({"hours_observed": hours_observed})
station_audit["hour_coverage"] = station_audit["hours_observed"] / expected_hours

for var in core_vars:
    # among existing rows
    station_audit[f"{var}_missing_rows"] = 1 - non_missing_counts[var] / station_audit["hours_observed"]
    # over the full window: a missing hour counts as missing
    station_audit[f"{var}_missing_window"] = 1 - non_missing_counts[var] / expected_hours

window_cols = [f"{v}_missing_window" for v in core_vars]
station_audit["flagged_sparse"] = (
    station_audit[window_cols] > station_missingness_flag_threshold
).any(axis=1)

print(f"\nPer-station missingness over the full window ({expected_hours:,} hours):")
print(
    station_audit[["hours_observed", "hour_coverage"] + window_cols]
    .sort_values("hour_coverage")
    .head(15)
    .round(3)
    .to_string()
)

flagged_airports = station_audit.index[station_audit["flagged_sparse"]].tolist()
print(
    f"\n{len(flagged_airports)} / {len(station_audit)} airports exceed "
    f"{station_missingness_flag_threshold:.0%} missingness (full window) in at "
    f"least one core variable: {flagged_airports}"
)

# %% 7a. Observation cadence ----------------------------------------------------
# Hourly stations have a median gap of 1h. Synoptic stations report every 3 or
# 6 hours, which shows up as a larger median gap and spikes at fixed hours.
weather_sorted = weather.sort_values(["iata", "time"]).reset_index(drop=True)
gaps_h = weather_sorted.groupby("iata")["time"].diff() / pd.Timedelta(hours=1)
cadence = gaps_h.groupby(weather_sorted["iata"]).agg(
    median_gap_h="median", max_gap_h="max"
)
station_audit = station_audit.join(cadence)

non_hourly = cadence[cadence["median_gap_h"] > cadence_flag_median_gap_h]
print(f"\nStations not reporting hourly (median gap > {cadence_flag_median_gap_h}h): {len(non_hourly)}")
if len(non_hourly):
    print(non_hourly.to_string())

# Detail for low-coverage airports: hour-of-day distribution and gap sizes.
low_coverage = station_audit.index[station_audit["hour_coverage"] < 0.90].tolist()
for iata in low_coverage:
    sub = weather_sorted[weather_sorted["iata"] == iata]
    print(f"\n--- {iata} (station {sub['station_id'].iloc[0]}, coverage "
          f"{station_audit.loc[iata, 'hour_coverage']:.1%}) ---")
    print("Observations per hour of day (UTC/stored tz):")
    print(sub["time"].dt.hour.value_counts().sort_index().to_string())
    print("Most common gaps between observations (hours):")
    print(gaps_h.loc[sub.index].value_counts().head(5).to_string())
    monthly = sub.groupby(sub["time"].dt.strftime("%Y-%m")).size()
    print("Observations per month:")
    print(monthly.to_string())

# %% 7b. Shared missing-hour patterns -------------------------------------------
# Stations with exactly the same set of missing hours most likely share a data
# source that had an outage -- not a coincidence and not the same station.
missing_signatures = {}
for iata, g in weather.groupby("iata"):
    missing_hours = full_index.difference(pd.DatetimeIndex(g["time"].unique()))
    if len(missing_hours):
        missing_signatures[iata] = missing_hours

signature_groups = {}
for iata, missing_hours in missing_signatures.items():
    key = hash(tuple(missing_hours.asi8))
    signature_groups.setdefault(key, []).append(iata)

shared_groups = [sorted(g) for g in signature_groups.values() if len(g) > 1]
print(f"\nGroups of airports with identical missing hours: {len(shared_groups)}")
for group in sorted(shared_groups, key=len, reverse=True):
    missing_hours = missing_signatures[group[0]]
    runs = summarise_runs(missing_hours)
    stations = weather[weather["iata"].isin(group)].groupby("iata")["station_id"].first()
    print(f"\n  {len(group)} airports, {len(missing_hours)} missing hours in {len(runs)} run(s):")
    print(f"  airports/stations: {dict(stations)}")
    print(runs.head(max_runs_to_print).to_string(index=False))

# Also print the missing runs for every other airport with gaps, for reference.
grouped_airports = {a for g in shared_groups for a in g}
other_gappy = [a for a in missing_signatures if a not in grouped_airports]
print(f"\nAirports with unique missing-hour patterns: {len(other_gappy)}")
for iata in sorted(other_gappy):
    runs = summarise_runs(missing_signatures[iata])
    print(f"  {iata}: {len(missing_signatures[iata])} missing hours in {len(runs)} run(s), "
          f"longest {runs['hours'].max()}h")

# %% 7c. Duplicated / near-identical series between stations -------------------
# Two distinct stations should never have identical temperatures most of the
# time. If they do, one series is copied or model-generated.
temp_wide = weather.pivot_table(index="time", columns="iata", values="temp", aggfunc="first")
temp_corr = temp_wide.corr(min_periods=500)

station_coords_note = (
    "Note: high correlation between NEARBY airports (e.g. two NYC airports) is "
    "expected; identical values between distant airports is the red flag."
)

suspicious_pairs = []
for a, b in combinations(temp_wide.columns, 2):
    both = temp_wide[[a, b]].dropna()
    if len(both) < 500:
        continue
    identical_share = (both[a] == both[b]).mean()
    corr = temp_corr.loc[a, b]
    if corr > duplicate_corr_threshold or identical_share > duplicate_identical_share:
        suspicious_pairs.append(
            {"airport_a": a, "airport_b": b, "temp_corr": round(corr, 4),
             "identical_share": round(identical_share, 3), "overlap_hours": len(both)}
        )

print(f"\nStation pairs with near-identical temperature series: {len(suspicious_pairs)}")
print(station_coords_note)
if suspicious_pairs:
    print(pd.DataFrame(suspicious_pairs).to_string(index=False))

# Explicit check of the shared-pattern groups from 7b: are their VALUES related?
for group in shared_groups:
    sub_corr = temp_corr.loc[group, group].round(3)
    print(f"\nTemperature correlation within shared-gap group {group}:")
    print(sub_corr.to_string())

# %% 8. Appendix A.3 -- airport-to-station mapping and coverage ------------
# Hour coverage = distinct hours with a row / hours in the download window.
# Variable availability = share of existing rows where the variable is non-missing.
a3_stations = weather.groupby("iata").agg(
    station_id=("station_id", lambda s: ", ".join(sorted(s.dropna().astype(str).unique()))),
    station_distance_km=("station_distance_km", "mean"),
    hours_observed=("time", "nunique"),
)
a3_stations["hour_coverage_pct"] = 100 * a3_stations["hours_observed"] / expected_hours
availability = weather[core_vars].notna().groupby(weather["iata"]).mean() * 100
a3_stations = a3_stations.join(availability.add_suffix("_available_pct"))
a3_stations["flagged_sparse"] = np.where(
    a3_stations.index.isin(flagged_airports), "yes", "no"
)
a3_stations = a3_stations.reset_index().sort_values("iata").reset_index(drop=True)

save_appendix_table(
    a3_stations,
    "a3_weather_station_mapping",
    f"Meteostat station assigned to each airport and data coverage ({expected_hours:,} hours "
    f"in the download window). Flagged stations exceed "
    f"{station_missingness_flag_threshold:.0%} missingness over the full window in at "
    f"least one core variable.",
    longtable=True,
    float_format="%.1f",
)

# %% 9. Save --------------------------------------------------------------------
output_path = weather_processed_dir / "weather_processed.parquet"
weather.to_parquet(output_path, index=False)
print(f"\nSaved {len(weather):,} rows, {len(weather.columns)} columns to {output_path}")

audit_path = weather_processed_dir / "weather_station_audit.csv"
station_audit.round(4).to_csv(audit_path)
print(f"Saved station audit ({len(station_audit)} airports) to {audit_path}")

if suspicious_pairs:
    pairs_path = weather_processed_dir / "weather_suspicious_station_pairs.csv"
    pd.DataFrame(suspicious_pairs).to_csv(pairs_path, index=False)
    print(f"Saved suspicious station pairs to {pairs_path}")