'''
Write a Python script to process a raw GDELT event dataset (previously downloaded via BigQuery by a separate script, already filtered by root code, date range, and radius around a set of airports) for a flight-delay-prediction pipeline.

The raw file is already a curated column set (not the full GDELT schema), so the main job here is less about trimming columns and more about validating that the upstream query actually did what it was supposed to. The script should:

1. Load the raw GDELT events file.
2. Print every column available, so the schema can be audited against expectations.
3. Subset to the columns needed downstream, dropping fields that were only pulled for manual check of the geocoding (raw lat/lon, full place name) now that a computed distance-to-airport field already exists.
4. Report missing value percentage per column.
5. Run query-correctness sanity checks rather than assuming the upstream filtering worked:
   - Confirm only the intended CAMEO root codes are present in the data (flag anything unexpected), and print the root code distribution.
   - Confirm no row exceeds the intended search radius (validates the geo join actually constrained results).
   - Confirm the event date range falls within the intended buffer + study window (catches boundary/partition-filter mismatches).
   - Check for duplicate (event ID, airport) pairs, which would indicate a join bug — as distinct from the same event legitimately matching multiple nearby airports, which is expected and should be reported separately, not flagged as an error.
   - Report the distribution of event counts per airport, and flag airports with very few matched events over the whole window, since that's relevant to interpreting weak or missing signal for that airport later.
6. Save the subset, audited event-level result to a processed-data location.
7. Aggregate the audited event-level data to one row per (airport, date), with a count column per intended CAMEO root code plus a total count across all codes.

The event-level load/audit/subset/save portion must not impute anything or otherwise transform values — matching the same shape as the earlier BTS and Meteostat processing scripts — right up until the final aggregation step, which is the one deliberate exception. Use snake_case naming, # %% cell markers for VS Code, and pull paths and the query parameters being validated against (expected root codes, radius, date window) from constants at the top of the script rather than hardcoding them inline.
'''
# 7.09.2026 22:40 CET
# Author: Anna Andruszkiewicz (code and adjustments), Claude Sonnet 5 (code)

"""
process_gdelt.py

Stage 1 cleaning/audit pass over raw GDELT event data (downloaded by
download_gdelt.py), followed by aggregation to a daily airport-level count
table.
Pipeline:
    1. Load the raw GDELT events file
    2. Audit available columns
    3. Subset to columns needed downstream (drop QA-only geocode fields)
    4. Missingness audit per column
    5. Query-correctness sanity checks:
         - only the intended root codes are present
         - all matches actually fall within the intended radius
         - date range matches the intended buffer + study window
         - no duplicate (event, airport) pairs
         - per-airport event count distribution (flag airports with ~0 events)
    6. Save the cleaned/subset event-level result
    7. Aggregate to (airport, date) counts per CAMEO root code + total, and
       save that separately -- a grain change, not feature engineering, so
       it belongs here rather than in feature_engineering.py
Run cell-by-cell in VS Code (Code Runner respects the "# %%" markers) or as
a plain script: `python process_gdelt.py`.
"""

# %% Imports and config ---------------------------------------------------
from pathlib import Path

import pandas as pd

from config import raw_data_path, processed_data_path
from appendix_utils import save_appendix_table

gdelt_raw_dir = Path(raw_data_path) / "gdelt"
gdelt_processed_dir = Path(processed_data_path) / "gdelt"
gdelt_processed_dir.mkdir(parents=True, exist_ok=True)

# Values the download query was built with -- used here to check the query
expected_root_codes = {"14", "17", "18", "20"}
expected_radius_meters = 50_000
expected_start_date = pd.Timestamp("2024-01-01") - pd.Timedelta(days=7)
expected_end_date = pd.Timestamp("2025-12-31")
source_table = "gdelt-bq.gdeltv2.events_partitioned"  # check against download_gdelt.py


keep_cols = [
    "GLOBALEVENTID",
    "SQLDATE",
    "airport_iata",
    "EventRootCode",
    "EventCode",
    "GoldsteinScale",
    "NumMentions",
    "NumSources",
    "NumArticles",
    "AvgTone",
    "ActionGeo_Type",
    "distance_meters",
]

# CAMEO root code label used for the aggregated count columns
root_code_labels = {
    "14": "protest",
    "17": "coerce",
    "18": "assault",
    "20": "mass_violence",
}


# %% 1. Load raw GDELT events ------------------------------------------------
raw_path = gdelt_raw_dir / "gdelt_events_raw.parquet"
gdelt_raw = pd.read_parquet(raw_path)
print(f"Loaded {len(gdelt_raw):,} airport-event matches.")

# %% 2. Audit available columns ----------------------------------------------
print(f"\n{len(gdelt_raw.columns)} columns available:")
for col in gdelt_raw.columns:
    print(f"  - {col}")

missing_from_keep = [c for c in keep_cols if c not in gdelt_raw.columns]
if missing_from_keep:
    print(f"\nWARNING: expected columns not found in raw data: {missing_from_keep}")

# %% 3. Subset to needed columns ----------------------------------------------
present_keep_cols = [c for c in keep_cols if c in gdelt_raw.columns]
gdelt = gdelt_raw[present_keep_cols].copy()
gdelt["event_date"] = pd.to_datetime(gdelt["SQLDATE"], format="%Y%m%d")
print(f"\nSubset to {len(gdelt.columns)} columns, {len(gdelt):,} rows.")

# %% 4. Missingness audit ------------------------------------------------------
missing_pct = (gdelt.isna().mean() * 100).round(2).sort_values(ascending=False)
print("\nMissing value % per column:")
print(missing_pct.to_string())

# %% 5a. Sanity check: root codes -----------------------------------------------
observed_root_codes = set(gdelt["EventRootCode"].astype(str).unique())
unexpected_codes = observed_root_codes - expected_root_codes
if unexpected_codes:
    print(f"\nWARNING: unexpected root codes present (query filter may not have applied): {unexpected_codes}")
else:
    print(f"\nOK: only expected root codes present {sorted(observed_root_codes)}.")

print("\nRoot code distribution:")
print(gdelt["EventRootCode"].value_counts().to_string())

# %% 5b. Sanity check: distance within radius -----------------------------------
max_distance = gdelt["distance_meters"].max()
print(f"\nMax distance_meters observed: {max_distance:,.0f} (radius limit: {expected_radius_meters:,})")
if max_distance > expected_radius_meters:
    n_over = (gdelt["distance_meters"] > expected_radius_meters).sum()
    print(f"WARNING: {n_over} rows exceed the intended radius -- check the ST_DWITHIN query.")
else:
    print("OK: all matches fall within the intended radius.")

# %% 5c. Sanity check: date range -------------------------------------------------
print(f"\nEvent date range: {gdelt['event_date'].min().date()} to {gdelt['event_date'].max().date()}")
print(f"Expected range:    {expected_start_date.date()} to {expected_end_date.date()}")
if gdelt["event_date"].min() < expected_start_date or gdelt["event_date"].max() > expected_end_date:
    print("WARNING: event dates fall outside the expected buffer + study window.")
else:
    print("OK: event dates within expected range.")

# %% 5d. Sanity check: duplicate (event, airport) pairs ---------------------------
n_duplicates = gdelt.duplicated(subset=["GLOBALEVENTID", "airport_iata"]).sum()
print(f"\nDuplicate (event, airport) pairs: {n_duplicates}")
if n_duplicates:
    print("WARNING: unexpected duplicates -- check the join in download_gdelt.py.")

n_multi_airport_events = (
    gdelt.groupby("GLOBALEVENTID")["airport_iata"].nunique().gt(1).sum()
)
print(
    f"Events matched to multiple airports (expected/normal for overlapping "
    f"radii): {n_multi_airport_events:,}"
)

# %% 5e. Per-airport event count distribution --------------------------------------
events_per_airport = gdelt.groupby("airport_iata").size().sort_values()
print(f"\nEvents per airport -- distribution:")
print(events_per_airport.describe().to_string())

low_coverage_airports = events_per_airport[events_per_airport < 10]
if len(low_coverage_airports):
    print(f"\n{len(low_coverage_airports)} airports with < 10 matched events over the whole window:")
    print(low_coverage_airports.to_string())

# %% 6. Save event-level result -----------------------------------------------------
output_path = gdelt_processed_dir / "gdelt_processed.parquet"
gdelt.to_parquet(output_path, index=False)
print(f"\nSaved {len(gdelt):,} rows, {len(gdelt.columns)} columns to {output_path}")

# %% 7. Aggregate to daily (airport, date) counts per root code + total

gdelt["root_label"] = gdelt["EventRootCode"].astype(str).map(root_code_labels)

gdelt_daily = (
    gdelt.groupby(["airport_iata", "event_date", "root_label"])
    .size()
    .unstack("root_label", fill_value=0)
    .reset_index()
)
gdelt_daily.columns.name = None

for label in root_code_labels.values():
    if label not in gdelt_daily.columns:
        gdelt_daily[label] = 0
    gdelt_daily = gdelt_daily.rename(columns={label: f"gdelt_{label}_count"})

count_cols = [f"gdelt_{label}_count" for label in root_code_labels.values()]
gdelt_daily["gdelt_total_count"] = gdelt_daily[count_cols].sum(axis=1)

daily_output_path = gdelt_processed_dir / "gdelt_daily_counts.parquet"
gdelt_daily.to_parquet(daily_output_path, index=False)
print(
    f"\nSaved daily counts: {gdelt_daily.shape[0]:,} (airport, date) rows, "
    f"{gdelt_daily.shape[1]} columns to {daily_output_path}"
)

# %% 8. Appendix A.6 -- GDELT extraction specification and validation --------
# The BigQuery SQL itself lives in download_gdelt.py; save it there as
# appendix/a6_gdelt_query.sql and include it as a code listing.
geo_types = sorted(gdelt["ActionGeo_Type"].dropna().astype(int).unique()) if "ActionGeo_Type" in gdelt else []

a6_gdelt_spec = pd.DataFrame(
    [
        ("Source table", source_table),
        ("CAMEO root codes", ", ".join(f"{c} ({root_code_labels[c]})" for c in sorted(expected_root_codes))),
        ("ActionGeo_Type values present", ", ".join(map(str, geo_types))),
        ("Search radius around airport", f"{expected_radius_meters / 1000:.0f} km (ST_DWITHIN)"),
        ("Query window (incl. 7-day lag buffer)", f"{expected_start_date.date()} to {expected_end_date.date()}"),
        ("Observed event dates", f"{gdelt['event_date'].min().date()} to {gdelt['event_date'].max().date()}"),
        ("Airport-event matches", f"{len(gdelt):,}"),
        ("Unique events", f"{gdelt['GLOBALEVENTID'].nunique():,}"),
        ("Events matched to more than one airport", f"{n_multi_airport_events:,}"),
        ("Duplicate (event, airport) pairs", f"{n_duplicates:,}"),
        ("Max. distance to airport", f"{max_distance / 1000:.1f} km"),
        ("Airports with at least one match", f"{gdelt['airport_iata'].nunique():,}"),
        ("Airports with fewer than 10 matches", f"{len(low_coverage_airports):,}"),
        ("Airport-days with at least one match", f"{len(gdelt_daily):,}"),
        ("Max. missing share in any kept column", f"{missing_pct.max():.2f}%"),
    ],
    columns=["item", "value"],
)
save_appendix_table(
    a6_gdelt_spec,
    "a6_gdelt_extraction",
    "GDELT extraction parameters and validation checks. Counts refer to news "
    "coverage of events, not to distinct real-world events.",
)