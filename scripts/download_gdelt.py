'''
Write a Python script to download GDELT 2.0 event data via BigQuery for a fixed set of airports, to be joined later with BTS flight and Meteostat weather data in a flight-delay-prediction pipeline.

Requirements:

1. Load the airport list from a retained_airports.csv file (IATA code, lat, lon) produced by an earlier BTS processing step.
2. Query GDELT's public BigQuery event table (gdelt-bq.gdeltv2.events_partitioned), filtered by _PARTITIONTIME for cost efficiency, over the study window (2024-01-01 to 2025-12-31) plus a short buffer before the start date (matching the pipeline's max lag/rolling window, currently 7 days) so early-window features have complete history to compute from.
3. Match events to airports by location: within a fixed radius (50 km) of each airport's coordinates, using BigQuery geography functions (ST_DWITHIN/ST_GEOGPOINT), and additionally restrict to city/landmark-level geocoded events (ActionGeo_Type 3 or 4)
4. Filter to a small set of CAMEO root codes representing physical/operational disruption relevant to a domestic US aviation context — not diplomatic/statement-type codes. Make the code list an easily editable constant with a comment explaining the rationale for each choice.
5. Stage the airport list as a temporary table in the user's own GCP project/dataset (since the public GDELT project is read-only) to join against, rather than inlining the airport list as a literal SQL array. Delete the temp table at the end of the script regardless of success.
6. Before running the real query, run a dry-run cost estimate and hard-abort with a clear error if the estimated bytes scanned exceed a configurable safety cap — don't silently run an expensive query.
7. After the query completes, report: total events matched, distinct events (a single event can match multiple nearby airports), how many of the airports got zero matches, and use the BigQuery Storage API with a visible progress bar for the results download (the default REST download path is slow for larger result sets).
8. Save the raw, unaggregated event-level result (one row per airport-event match) to a raw-data location. This script only queries, filters by radius/code/date, and saves — it does not aggregate to daily/per-airport features, impute, or otherwise transform values; that's a separate downstream script.

Use snake_case naming, # %% cell markers for VS Code, and pull GCP project ID, BigQuery dataset ID, and data paths from the project's config.py rather than hardcoding them.
'''
# 7.09.2026 22:30 CET
# Author: Anna Andruszkiewicz (code and adjustments), Claude Sonnet 5 (code)

"""
Downloads GDELT 2.0 event data via BigQuery for the airports retained by
process_bts.py, filtered to:
  - events within a fixed radius of each airport (ActionGeo location)
  - a set of operationally-relevant CAMEO root codes
  - the study window, with a buffer before the start date so early-window
    rolling/lag features have complete data to compute from

This script only queries, filters, and saves raw event-level rows -- no
aggregation to daily/per-airport features happens here (that's
process_gdelt.py, downstream).

Cost safety: runs a dry run first and aborts if the estimated bytes
scanned exceed a configured cap, rather than silently running an
expensive query.

Run cell-by-cell in VS Code or as a script: `python download_gdelt.py`
"""

# %% Imports and config ---------------------------------------------------
from pathlib import Path

import pandas as pd
from google.cloud import bigquery

from config import raw_data_path, processed_data_path, gcp_project_id, bq_dataset_id

processed_bts_dir = Path(processed_data_path) / "bts"
gdelt_raw_dir = Path(raw_data_path) / "gdelt"
gdelt_raw_dir.mkdir(parents=True, exist_ok=True)

study_start = pd.Timestamp("2024-01-01")
study_end = pd.Timestamp("2025-12-31")
buffer_days = 7
query_start = study_start - pd.Timedelta(days=buffer_days)

radius_meters = 50_000  # 50 km
require_precise_geocode = True  # restrict to ActionGeo_Type 3 (city) / 4 (landmark)

# CAMEO root codes representing physical/operational disruption, as opposed
# to diplomatic statements/appeals/cooperation (01-13).
#   14 PROTEST, 17 COERCE, 18 ASSAULT, 20 USE UNCONVENTIONAL MASS VIOLENCE
root_codes = ["14", "17", "18", "20"]

max_bytes_billed = 500 * 1024**3  # 500 GB safety cap -- abort rather than overrun quota


# %% 1. Load retained airports -----------------------------------------------
airports = pd.read_csv(processed_bts_dir / "retained_airports.csv")
airports = airports.dropna(subset=["lat", "lon"])
print(f"{len(airports)} airports to match against GDELT events.")

client = bigquery.Client(project=gcp_project_id)

# %% 2. Stage airport list as a temp table (own project, not gdelt-bq)
airport_table_id = f"{gcp_project_id}.{bq_dataset_id}.retained_airports_tmp"
job_config = bigquery.LoadJobConfig(
    schema=[
        bigquery.SchemaField("iata", "STRING"),
        bigquery.SchemaField("lat", "FLOAT64"),
        bigquery.SchemaField("lon", "FLOAT64"),
    ],
    write_disposition="WRITE_TRUNCATE",
)
load_job = client.load_table_from_dataframe(
    airports[["iata", "lat", "lon"]], airport_table_id, job_config=job_config
)
load_job.result()
print(f"Staged airport list to {airport_table_id}")

# %% 3. Build the query -------------------------------------------------------
geocode_filter = "AND e.ActionGeo_Type IN (3, 4)" if require_precise_geocode else ""

query = f"""
SELECT
    e.GLOBALEVENTID,
    e.SQLDATE,
    e.EventRootCode,
    e.EventCode,
    e.GoldsteinScale,
    e.NumMentions,
    e.NumSources,
    e.NumArticles,
    e.AvgTone,
    e.ActionGeo_Type,
    e.ActionGeo_FullName,
    e.ActionGeo_Lat,
    e.ActionGeo_Long,
    a.iata AS airport_iata,
    ST_DISTANCE(
        ST_GEOGPOINT(e.ActionGeo_Long, e.ActionGeo_Lat),
        ST_GEOGPOINT(a.lon, a.lat)
    ) AS distance_meters
FROM `gdelt-bq.gdeltv2.events_partitioned` e
CROSS JOIN `{airport_table_id}` a
WHERE e._PARTITIONTIME >= TIMESTAMP('{query_start.date()}')
  AND e._PARTITIONTIME <= TIMESTAMP('{study_end.date()}')
  AND e.EventRootCode IN UNNEST({root_codes})
  AND e.ActionGeo_Lat IS NOT NULL
  AND e.ActionGeo_Long IS NOT NULL
  {geocode_filter}
  AND ST_DWITHIN(
        ST_GEOGPOINT(e.ActionGeo_Long, e.ActionGeo_Lat),
        ST_GEOGPOINT(a.lon, a.lat),
        {radius_meters}
      )
"""

# %% 4. Dry run cost check ----------------------------------------------------
dry_run_config = bigquery.QueryJobConfig(dry_run=True, use_query_cache=False)
dry_run_job = client.query(query, job_config=dry_run_config)
estimated_gb = dry_run_job.total_bytes_processed / 1024**3
print(f"Estimated bytes to be scanned: {estimated_gb:.2f} GB")

if dry_run_job.total_bytes_processed > max_bytes_billed:
    raise RuntimeError(
        f"Estimated scan ({estimated_gb:.2f} GB) exceeds the safety cap "
        f"({max_bytes_billed / 1024**3:.0f} GB). Narrow the date range, radius, "
        f"or root codes before re-running, or raise max_bytes_billed deliberately."
    )

# %% 5. Run the actual query --------------------------------------------------
run_config = bigquery.QueryJobConfig(maximum_bytes_billed=max_bytes_billed)
query_job = client.query(query, job_config=run_config)
print("Query submitted, waiting for completion...")
query_job.result()  # blocks until the query itself finishes server-side
print("Query finished. Downloading results...")
gdelt_events = query_job.to_dataframe(
    create_bqstorage_client=True,  # much faster parallel download than the default REST API
    progress_bar_type="tqdm",
)

print(f"\nRetrieved {len(gdelt_events):,} airport-event matches.")
print(f"Distinct events: {gdelt_events['GLOBALEVENTID'].nunique():,}")
print(f"Distinct airports matched: {gdelt_events['airport_iata'].nunique()} / {len(airports)}")

unmatched_airports = set(airports["iata"]) - set(gdelt_events["airport_iata"].unique())
if unmatched_airports:
    print(f"Airports with ZERO matched events: {sorted(unmatched_airports)}")

# %% 6. Save --------------------------------------------------------------------
output_path = gdelt_raw_dir / "gdelt_events_raw.parquet"
gdelt_events.to_parquet(output_path, index=False)
print(f"\nSaved {len(gdelt_events):,} rows to {output_path}")

# %% 7. Cleanup temp table ----------------------------------------------------
client.delete_table(airport_table_id, not_found_ok=True)
print(f"Deleted temp table {airport_table_id}")