"""
Data Source Validation Tester
==============================
Purpose: pull a small (1-2 month) slice from all three data sources used in
the thesis (BTS TranStats, Meteostat, GDELT) and check whether each is
actually workable for the Model A / B / C design -- BEFORE committing to the
full pipeline build. Especially stress-tests GDELT, since it's the novel,
highest-risk component (airport-coordinate matching, event density, lag
feature viability).

v2 changes:
  - Every source is cached to disk on first successful run. Reruns load
    from ./validation_output/cache/ instead of hitting the network/API
    again. Set FORCE_REFRESH = True (below) or delete the cache file to
    force a fresh pull.
  - GDELT querying is now cost-aware:
      * uses `gdeltv2.events_partitioned` + a _PARTITIONTIME filter, so
        BigQuery only scans the requested days instead of the full
        unpartitioned events table (this is the single biggest lever --
        can be a >90% reduction in bytes scanned)
      * runs a dry run first and prints the exact bytes it WILL scan
        before spending any quota
      * refuses to run (or trims columns further) if the estimate exceeds
        MAX_SAFE_GB, and always sets maximum_bytes_billed as a hard cap so
        a mistake can't silently eat your 1TB/month free tier
      * selects only the columns actually needed for the diagnostics

This is a diagnostic script, not the real pipeline. It downloads a small
window (default: Jan-Feb 2024), runs structural checks, and prints a
PASS / WARN / FAIL verdict per source with the specific numbers behind it.

Requirements (pip install):
    requests pandas numpy pyarrow meteostat google-cloud-bigquery db-dtypes

GDELT via BigQuery needs auth. Easiest options:
    - Colab: from google.colab import auth; auth.authenticate_user()
    - Local: gcloud auth application-default login
    - Or set GOOGLE_APPLICATION_CREDENTIALS to a service account JSON

Run: python validate_data_sources.py
Output: prints a report to stdout, saves results.json + cached samples to
        ./validation_output/
"""

import io
import os
import json
import time
import hashlib
import zipfile
from datetime import datetime

import numpy as np
import pandas as pd
import requests

# ============================================================
# CONFIG -- adjust window / airports here
# ============================================================
TEST_YEAR = 2024
TEST_MONTHS = [1, 2]                 # Jan-Feb 2024: 2-month validation window
GDELT_MATCH_RADIUS_KM = 50           # primary coordinate-distance threshold
GDELT_MIN_EVENTS_FOR_RADIUS_OK = 5   # below this per month, fallback should trigger
GDELT_MAX_SAFE_GB = 2.0              # abort/ask before scanning more than this
GDELT_MAX_BILLED_GB = 5.0            # hard cap passed to BigQuery -- query errors
                                      # out rather than silently billing more

FORCE_REFRESH = False                # set True (or pass --refresh) to ignore cache

# If you download the monthly zips manually (recommended given BTS's slow
# servers -- browser downloads can run in parallel and resume on failure),
# drop them here using BTS's own filenames and the script will use them
# instead of hitting the network. Leave as None to always download via HTTP.
LOCAL_BTS_DIR = None  # e.g. "/Users/aniaandruszkiewicz/FS/Thesis/bts_zips"

# Your GCP project ID (needed for BigQuery -- `gcloud auth application-default
# login` alone does NOT tell the client which project to bill against).
# Find it with `gcloud config get-value project`, or in the GCP console.
GCP_PROJECT_ID = "thesis-507219"  # e.g. "my-thesis-project-123456"

TEST_AIRPORTS = ["ATL", "ORD", "DFW", "DEN", "LAX", "JFK"]
AIRPORT_COORDS = {
    "ATL": (33.6407, -84.4277, "GA"),
    "ORD": (41.9742, -87.9073, "IL"),
    "DFW": (32.8998, -97.0403, "TX"),
    "DEN": (39.8561, -104.6737, "CO"),
    "LAX": (33.9416, -118.4085, "CA"),
    "JFK": (40.6413, -73.7781, "NY"),
}

OUTPUT_DIR = "validation_output"
CACHE_DIR = os.path.join(OUTPUT_DIR, "cache")
os.makedirs(CACHE_DIR, exist_ok=True)

results = {}  # verdict + diagnostics per source, dumped to results.json


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def section(title):
    print("\n" + "=" * 72, flush=True)
    print(title, flush=True)
    print("=" * 72, flush=True)


def month_bounds(year, month):
    start = pd.Timestamp(year, month, 1)
    end = start + pd.offsets.MonthEnd(0)
    return start, end


# ============================================================
# CACHING -- one cache key per source, derived from the config that
# actually affects the pulled data. Change the window/airports/radius
# and the key changes automatically, so stale cache never gets reused
# silently.
# ============================================================
def _cache_key(*parts):
    raw = "_".join(str(p) for p in parts)
    return hashlib.md5(raw.encode()).hexdigest()[:10]


def cache_paths(name, *key_parts):
    key = _cache_key(name, *key_parts)
    data_path = os.path.join(CACHE_DIR, f"{name}_{key}.parquet")
    meta_path = os.path.join(CACHE_DIR, f"{name}_{key}.meta.json")
    return data_path, meta_path


def load_cache(data_path, meta_path):
    if FORCE_REFRESH or not (os.path.exists(data_path) and os.path.exists(meta_path)):
        return None, None
    df = pd.read_parquet(data_path)
    with open(meta_path) as f:
        meta = json.load(f)
    log(f"  Loaded from cache: {data_path} ({len(df):,} rows, cached {meta.get('cached_at')})")
    return df, meta


def save_cache(df, data_path, meta_path, meta):
    df.to_parquet(data_path)
    meta = {**meta, "cached_at": datetime.now().isoformat(), "rows": int(len(df))}
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)


# ============================================================
# MODEL A -- BTS TRANSTATS
# ============================================================
def bts_candidate_urls(year, month):
    # BTS has used several table names over time. The no-parentheses
    # "1987_present" name has been confirmed working (2024 data) and is
    # tried first; the others are kept as fallbacks in case naming shifts
    # for other years.
    base = "https://transtats.bts.gov/PREZIP/"
    names = [f"On_Time_Reporting_Carrier_On_Time_Performance_1987_present_{year}_{month}.zip"]
    if year >= 2018:
        names.append(f"On_Time_Marketing_Carrier_On_Time_Performance_Beginning_January_2018_{year}_{month}.zip")
    names.append(f"On_Time_Reporting_Carrier_On_Time_Performance_(1987_present)_{year}_{month}.zip")
    return [base + n for n in names]


def stream_download(url, overall_timeout=420, progress_every=8):
    """GET with progress logging and a hard wall-clock cap. requests' own
    read-timeout only fires on total silence between chunks, so a slow
    trickle (BTS is notoriously slow -- observed ~120KB/s) can run forever
    without tripping it or telling you anything is happening. This reports
    bytes as they come in and aborts cleanly at overall_timeout regardless
    of trickle speed. 420s comfortably covers a ~30MB file even at ~100KB/s."""
    t0 = time.time()
    last_log = t0
    chunks = []
    total = 0
    with requests.get(url, timeout=(10, 45), stream=True) as r:
        r.raise_for_status()
        content_length = r.headers.get("Content-Length")
        size_str = f"{int(content_length)/1e6:.1f} MB" if content_length else "unknown size"
        log(f"    connected, downloading ({size_str}) ...")
        for chunk in r.iter_content(chunk_size=1024 * 256):
            if chunk:
                chunks.append(chunk)
                total += len(chunk)
            now = time.time()
            if now - last_log > progress_every:
                log(f"    ... {total/1e6:.1f} MB so far ({now - t0:.0f}s elapsed)")
                last_log = now
            if now - t0 > overall_timeout:
                raise TimeoutError(f"exceeded {overall_timeout}s wall-clock cap at {total/1e6:.1f} MB")
    log(f"    done: {total/1e6:.1f} MB in {time.time() - t0:.0f}s")
    return b"".join(chunks)


def download_bts_month(year, month):
    # 1. Check for a manually-downloaded local zip first (any of the known
    #    filename patterns, since BTS's naming has shifted over time).
    if LOCAL_BTS_DIR:
        for url in bts_candidate_urls(year, month):
            fname = url.split("/")[-1]
            local_path = os.path.join(LOCAL_BTS_DIR, fname)
            if os.path.exists(local_path):
                log(f"    found local file: {local_path}")
                with zipfile.ZipFile(local_path) as zf:
                    csv_name = [n for n in zf.namelist() if n.lower().endswith(".csv")][0]
                    with zf.open(csv_name) as f:
                        return pd.read_csv(f, low_memory=False)
        log(f"    LOCAL_BTS_DIR is set but no matching zip found for {year}-{month:02d} "
            f"in {LOCAL_BTS_DIR} -- falling back to HTTP download")

    # 2. Fall back to downloading over HTTP.
    last_err = None
    for url in bts_candidate_urls(year, month):
        try:
            log(f"    trying {url.split('/')[-1]} ...")
            content = stream_download(url)
            zf = zipfile.ZipFile(io.BytesIO(content))
            csv_name = [n for n in zf.namelist() if n.lower().endswith(".csv")][0]
            with zf.open(csv_name) as f:
                return pd.read_csv(f, low_memory=False)
        except Exception as e:
            log(f"    -> failed: {e}")
            last_err = e
            continue
    raise last_err


def test_bts():
    section("MODEL A -- BTS TRANSTATS")
    data_path, meta_path = cache_paths("bts", TEST_YEAR, TEST_MONTHS)

    bts, meta = load_cache(data_path, meta_path)
    if bts is None:
        frames = []
        for m in TEST_MONTHS:
            try:
                log(f"Downloading BTS {TEST_YEAR}-{m:02d} ...")
                df = download_bts_month(TEST_YEAR, m)
                frames.append(df)
                log(f"  -> {len(df):,} rows, {df.shape[1]} columns")
            except Exception as e:
                log(f"  FAILED: {e}")
        if not frames:
            results["bts"] = {"status": "FAIL", "reason": "no monthly file downloaded"}
            return None
        bts = pd.concat(frames, ignore_index=True)
        save_cache(bts, data_path, meta_path, {"source": "transtats.bts.gov", "months": TEST_MONTHS})
    else:
        log("  (skipped download -- using cached data)")

    # BTS's raw column names use CamelCase, not the underscored convention
    # some tutorials/Kaggle mirrors use (e.g. "FlightDate" not "FL_DATE").
    col_map = {
        "FlightDate": "FlightDate",
        "Origin": "Origin",
        "Dest": "Dest",
        "Reporting_Airline": "Reporting_Airline",
        "DepDelay": "DepDelay",
        "ArrDelay": "ArrDelay",
        "Cancelled": "Cancelled",
    }
    missing_cols = [c for c in col_map.values() if c not in bts.columns]
    if missing_cols:
        log(f"  WARNING -- expected columns not found (schema may have shifted again): {missing_cols}")
        log(f"  Actual columns in this file ({len(bts.columns)} total):")
        log(f"    {sorted(bts.columns.tolist())}")

    subset = bts[bts["Origin"].isin(TEST_AIRPORTS)].copy() if "Origin" in bts.columns else bts.copy()
    log(f"  Rows restricted to test airports (origin): {len(subset):,}")

    delay_rate, arr_missing = None, None
    if "ArrDelay" in subset.columns:
        subset["label_delayed"] = (subset["ArrDelay"] > 15).astype("Int64")
        delay_rate = float(subset["label_delayed"].mean())
        arr_missing = float(subset["ArrDelay"].isna().mean())
        log(f"  Delay rate (>15min DOT threshold): {delay_rate:.1%}")
        log(f"  ArrDelay missing rate (mostly cancellations): {arr_missing:.1%}")

    present_required = [c for c in col_map.values() if c in subset.columns]
    missingness = subset[present_required].isna().mean().round(3).to_dict()
    log(f"  Missingness by key column: {missingness}")

    status = "PASS"
    if missing_cols:
        status = "WARN"
    if len(subset) < 1000:
        status = "FAIL"

    results["bts"] = {
        "status": status,
        "rows_total": int(len(bts)),
        "rows_test_airports": int(len(subset)),
        "missing_columns": missing_cols,
        "delay_rate": delay_rate,
        "arr_delay_missing_rate": arr_missing,
        "missingness_by_column": missingness,
    }
    return subset


# ============================================================
# MODEL B -- METEOSTAT
# ============================================================
def test_meteostat():
    section("MODEL B -- METEOSTAT")
    data_path, meta_path = cache_paths("meteostat", TEST_YEAR, TEST_MONTHS, TEST_AIRPORTS)

    weather, meta = load_cache(data_path, meta_path)
    if weather is None:
        try:
            import meteostat as ms
        except ImportError:
            log("  meteostat not installed -- attempting `pip install meteostat` ...")
            import subprocess, sys
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "meteostat"])
                import meteostat as ms
                log("  meteostat installed successfully.")
            except Exception as e:
                log(f"  Auto-install failed ({e}). Run `pip install meteostat` manually and rerun.")
                results["meteostat"] = {"status": "FAIL", "reason": "package not installed"}
                return None

        start, _ = month_bounds(TEST_YEAR, TEST_MONTHS[0])
        _, end = month_bounds(TEST_YEAR, TEST_MONTHS[-1])
        start_dt, end_dt = start.to_pydatetime(), end.to_pydatetime()

        frames = []
        for code in TEST_AIRPORTS:
            lat, lon, _ = AIRPORT_COORDS[code]
            try:
                # Point-based hourly queries go through an endpoint capped at
                # 30 days/request and silently return None if exceeded (our
                # 2-month window blows past that). Looking up the nearest
                # station and querying by station instead avoids that cap
                # and matches meteostat's own documented usage pattern.
                point = ms.Point(lat, lon)
                stations = ms.stations.nearby(point, limit=1)
                if stations is None or stations.empty:
                    log(f"  {code} FAILED: no nearby station found")
                    continue
                data = ms.hourly(stations, start_dt, end_dt).fetch()
                if data is None or data.empty:
                    log(f"  {code} FAILED: station query returned no data")
                    continue
                data["airport"] = code
                frames.append(data)
                log(f"  {code}: {len(data):,} hourly rows returned (station {stations.index[0]})")
            except Exception as e:
                log(f"  {code} FAILED: {e}")

        if not frames:
            results["meteostat"] = {"status": "FAIL", "reason": "no station data returned for any airport"}
            return None

        weather = pd.concat(frames).reset_index()
        log(f"  Columns returned: {weather.columns.tolist()}")
        save_cache(weather, data_path, meta_path, {"source": "meteostat", "airports": TEST_AIRPORTS})
    else:
        log("  (skipped API pull -- using cached data)")

    key_fields = ["temp", "prcp", "wspd", "coco"]
    present_fields = [f for f in key_fields if f in weather.columns]
    absent_fields = [f for f in key_fields if f not in weather.columns]
    if absent_fields:
        log(f"  WARNING -- fields not present in Hourly() output: {absent_fields}")

    completeness = weather[present_fields].notna().mean().round(3).to_dict()
    log(f"  Field completeness (non-null rate): {completeness}")

    coco_vals = sorted(weather["coco"].dropna().unique().tolist()) if "coco" in weather.columns else []
    log(f"  Distinct coco codes observed: {coco_vals if coco_vals else 'N/A'}")

    avg_completeness = float(np.mean(list(completeness.values()))) if completeness else 0.0
    status = "PASS" if avg_completeness > 0.8 and not absent_fields else "WARN"

    results["meteostat"] = {
        "status": status,
        "rows": int(len(weather)),
        "fields_present": present_fields,
        "fields_absent": absent_fields,
        "completeness": completeness,
        "distinct_coco_codes": coco_vals,
    }
    return weather


# ============================================================
# MODEL C -- GDELT (via BigQuery) -- cost-aware version
# ============================================================
def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def build_gdelt_query(start_str, end_str, partition_start, partition_end):
    # `events_partitioned` + _PARTITIONTIME lets BigQuery prune to only the
    # requested days instead of scanning the full events table. Column list
    # is kept to exactly what the diagnostics below use -- BigQuery bills by
    # column bytes read, so every extra column here has a real cost.
    return f"""
        SELECT
          SQLDATE,
          ActionGeo_Lat,
          ActionGeo_Long,
          ActionGeo_CountryCode,
          ActionGeo_ADM1Code,
          GoldsteinScale,
          AvgTone
        FROM `gdelt-bq.gdeltv2.events_partitioned`
        WHERE _PARTITIONTIME >= TIMESTAMP("{partition_start}")
          AND _PARTITIONTIME <= TIMESTAMP("{partition_end}")
          AND SQLDATE BETWEEN {start_str} AND {end_str}
          AND ActionGeo_CountryCode = 'US'
          AND ActionGeo_Lat IS NOT NULL
          AND ActionGeo_Long IS NOT NULL
    """


def test_gdelt():
    section("MODEL C -- GDELT (via BigQuery, cost-aware)")
    data_path, meta_path = cache_paths("gdelt", TEST_YEAR, TEST_MONTHS)

    gdelt_df, meta = load_cache(data_path, meta_path)

    if gdelt_df is None:
        try:
            from google.cloud import bigquery
        except ImportError:
            log("  google-cloud-bigquery not installed (`pip install google-cloud-bigquery db-dtypes`)")
            results["gdelt"] = {"status": "FAIL", "reason": "package not installed"}
            return None

        try:
            client = bigquery.Client(project=GCP_PROJECT_ID) if GCP_PROJECT_ID else bigquery.Client()
        except Exception as e:
            log(f"  BigQuery client init failed: {e}")
            if "Project was not passed" in str(e) or "could not be determined" in str(e):
                log("  Fix: set GCP_PROJECT_ID near the top of this script to your GCP project ID")
                log("       (find it with `gcloud config get-value project`), or run:")
                log("       gcloud config set project YOUR_PROJECT_ID")
            else:
                log("  Fix: gcloud auth application-default login, or set GOOGLE_APPLICATION_CREDENTIALS,")
                log("       or in Colab: from google.colab import auth; auth.authenticate_user()")
            results["gdelt"] = {"status": "FAIL", "reason": "client init failure"}
            return None

        start_str = f"{TEST_YEAR}{TEST_MONTHS[0]:02d}01"
        _, end_ts = month_bounds(TEST_YEAR, TEST_MONTHS[-1])
        end_str = end_ts.strftime("%Y%m%d")
        partition_start = pd.Timestamp(TEST_YEAR, TEST_MONTHS[0], 1).strftime("%Y-%m-%d")
        partition_end = end_ts.strftime("%Y-%m-%d")

        query = build_gdelt_query(start_str, end_str, partition_start, partition_end)

        # --- dry run: see the cost before spending any quota ---
        dry_config = bigquery.QueryJobConfig(dry_run=True, use_query_cache=False)
        try:
            dry_job = client.query(query, job_config=dry_config)
            est_gb = dry_job.total_bytes_processed / (1024 ** 3)
        except Exception as e:
            log(f"  Dry run FAILED (check table/partition syntax): {e}")
            results["gdelt"] = {"status": "FAIL", "reason": f"dry run failed: {e}"}
            return None

        log(f"  Dry run estimate: {est_gb:.3f} GB will be scanned "
            f"(partitioned to {TEST_MONTHS[0]}-{TEST_MONTHS[-1]}/{TEST_YEAR} only)")

        if est_gb > GDELT_MAX_SAFE_GB:
            log(f"  ABORTING -- estimate ({est_gb:.2f} GB) exceeds GDELT_MAX_SAFE_GB "
                f"({GDELT_MAX_SAFE_GB} GB). Narrow TEST_MONTHS or raise the limit deliberately.")
            results["gdelt"] = {
                "status": "FAIL",
                "reason": "dry run estimate exceeded safety threshold",
                "estimated_gb": round(est_gb, 3),
            }
            return None

        # --- real run, with a hard billing cap as a second safety net in
        # case the estimate and actual bytes ever diverge ---
        run_config = bigquery.QueryJobConfig(
            maximum_bytes_billed=int(GDELT_MAX_BILLED_GB * 1024 ** 3)
        )
        log(f"  Running query (hard cap: {GDELT_MAX_BILLED_GB} GB billed)...")
        t0 = time.time()
        try:
            query_job = client.query(query, job_config=run_config)
            log("  Query submitted, waiting for it to finish executing ...")
            result = query_job.result()  # blocks until the query itself is done (usually fast)
            total_rows = result.total_rows
            log(f"  Query executed. Fetching {total_rows:,} rows "
                f"(REST fallback -- install `google-cloud-bigquery-storage` for a faster path) ...")

            rows = []
            t_fetch0 = time.time()
            last_log = t_fetch0
            for page in result.pages:
                rows.extend(dict(r) for r in page)
                now = time.time()
                if now - last_log > 5:
                    log(f"    ... fetched {len(rows):,}/{total_rows:,} rows ({now - t_fetch0:.0f}s elapsed)")
                    last_log = now
            gdelt_df = pd.DataFrame(rows)
            log(f"  Fetch complete: {len(gdelt_df):,} rows in {time.time() - t_fetch0:.0f}s")
        except Exception as e:
            log(f"  Query FAILED: {e}")
            results["gdelt"] = {"status": "FAIL", "reason": str(e)}
            return None

        actual_gb = query_job.total_bytes_billed / (1024 ** 3)
        log(f"  Query returned {len(gdelt_df):,} rows in {time.time() - t0:.1f}s "
            f"| billed {actual_gb:.3f} GB (of your 1 TB free tier)")

        if gdelt_df.empty:
            results["gdelt"] = {"status": "FAIL", "reason": "empty result set for this window"}
            return None

        save_cache(
            gdelt_df, data_path, meta_path,
            {"source": "bigquery:gdelt-bq.gdeltv2.events_partitioned",
             "billed_gb": round(actual_gb, 3), "months": TEST_MONTHS},
        )
    else:
        log("  (skipped BigQuery call entirely -- using cached data, 0 bytes billed)")

    gdelt_df["date"] = pd.to_datetime(gdelt_df["SQLDATE"], format="%Y%m%d")
    _, end_ts = month_bounds(TEST_YEAR, TEST_MONTHS[-1])
    n_days_total = int((end_ts - month_bounds(TEST_YEAR, TEST_MONTHS[0])[0]).days) + 1

    per_airport = {}
    daily_frames = []
    for code in TEST_AIRPORTS:
        lat, lon, state = AIRPORT_COORDS[code]
        dist = haversine_km(gdelt_df["ActionGeo_Lat"].values, gdelt_df["ActionGeo_Long"].values, lat, lon)
        matched = gdelt_df[dist <= GDELT_MATCH_RADIUS_KM].copy()

        used_fallback = False
        if len(matched) < GDELT_MIN_EVENTS_FOR_RADIUS_OK:
            state_matched = gdelt_df[gdelt_df["ActionGeo_ADM1Code"] == f"US{state}"].copy()
            if len(state_matched) > len(matched):
                matched = state_matched
                used_fallback = True

        daily_counts = matched.groupby(matched["date"].dt.date).size()
        n_zero_days = n_days_total - len(daily_counts)
        pct_zero = n_zero_days / n_days_total

        full_range = pd.date_range(month_bounds(TEST_YEAR, TEST_MONTHS[0])[0], end_ts, freq="D")
        daily_series = pd.Series(0, index=full_range.date)
        daily_series.update(daily_counts)
        lag1_defined = daily_series.shift(1).notna().sum()
        lag3_defined = daily_series.shift(3).notna().sum()

        goldstein_rate = float(matched["GoldsteinScale"].notna().mean()) if len(matched) else 0.0

        per_airport[code] = {
            "matched_events": int(len(matched)),
            "used_state_fallback": used_fallback,
            "days_with_events": int(len(daily_counts)),
            "total_days": n_days_total,
            "pct_zero_event_days": round(pct_zero, 2),
            "mean_daily_events": round(float(daily_counts.mean()), 2) if len(daily_counts) else 0.0,
            "goldstein_nonnull_rate": round(goldstein_rate, 2),
            "lag1_defined_days": int(lag1_defined),
            "lag3_defined_days": int(lag3_defined),
        }
        daily_frames.append(daily_series.rename(code))

        log(
            f"  {code}: {len(matched):,} events "
            f"({'state fallback' if used_fallback else f'{GDELT_MATCH_RADIUS_KM}km radius'}) | "
            f"{pct_zero:.0%} zero-event days | avg {per_airport[code]['mean_daily_events']}/day | "
            f"Goldstein present {goldstein_rate:.0%}"
        )

    gdelt_daily = pd.concat(daily_frames, axis=1)

    high_zero_airports = [c for c, v in per_airport.items() if v["pct_zero_event_days"] > 0.5]
    status = "PASS"
    if high_zero_airports:
        status = "WARN"
    if all(v["matched_events"] == 0 for v in per_airport.values()):
        status = "FAIL"

    results["gdelt"] = {
        "status": status,
        "rows_queried": int(len(gdelt_df)),
        "per_airport": per_airport,
        "airports_over_50pct_zero_days": high_zero_airports,
    }
    return gdelt_daily


# ============================================================
# CROSS-SOURCE JOINABILITY TEST
# ============================================================
def test_joinability(bts, weather, gdelt_daily):
    section("CROSS-SOURCE JOIN -- can Model A + B + C actually be assembled?")
    if bts is None or weather is None or gdelt_daily is None:
        log("  Skipped: one or more sources failed above.")
        results["joinability"] = {"status": "SKIPPED"}
        return

    bts_daily = (
        bts.assign(date=pd.to_datetime(bts["FlightDate"]).dt.date)
        .groupby(["date", "Origin"])["label_delayed"]
        .mean()
        .rename("delay_rate")
        .reset_index()
    )

    time_col = "time" if "time" in weather.columns else weather.index.name or "time"
    weather_daily = (
        weather.assign(date=lambda d: pd.to_datetime(d[time_col]).dt.date)
        .groupby(["date", "airport"])[["temp", "prcp", "wspd"]]
        .mean()
        .reset_index()
    )

    gdelt_long = gdelt_daily.reset_index().melt(id_vars="index", var_name="airport", value_name="gdelt_events")
    gdelt_long = gdelt_long.rename(columns={"index": "date"})

    merged = bts_daily.merge(
        weather_daily, left_on=["date", "Origin"], right_on=["date", "airport"], how="left"
    ).merge(gdelt_long, on=["date", "airport"], how="left")

    log(f"  Merged panel shape: {merged.shape}")
    miss = merged[["delay_rate", "temp", "prcp", "wspd", "gdelt_events"]].isna().mean().round(3).to_dict()
    log(f"  Post-merge missingness: {miss}")

    merged.to_csv(f"{OUTPUT_DIR}/merged_panel_sample.csv", index=False)

    status = "PASS" if merged.shape[0] > 0 and miss.get("delay_rate", 1) < 0.1 else "WARN"
    results["joinability"] = {"status": status, "merged_rows": int(len(merged)), "missingness": miss}


# ============================================================
# MAIN
# ============================================================
def main():
    import sys
    global FORCE_REFRESH
    if "--refresh" in sys.argv:
        FORCE_REFRESH = True
        log("--refresh passed: ignoring cache, will re-pull from all sources")

    print(f"Validation window: {TEST_YEAR}-{TEST_MONTHS[0]:02d} to {TEST_YEAR}-{TEST_MONTHS[-1]:02d}", flush=True)
    print(f"Test airports: {TEST_AIRPORTS}", flush=True)
    print(f"Cache dir: {os.path.abspath(CACHE_DIR)} (delete files here, or run with --refresh, to force a re-pull)", flush=True)

    bts = test_bts()
    weather = test_meteostat()
    gdelt_daily = test_gdelt()
    test_joinability(bts, weather, gdelt_daily)

    section("VERDICT SUMMARY")
    for source, r in results.items():
        print(f"  {source.upper():<15} {r.get('status', 'UNKNOWN')}")

    with open(f"{OUTPUT_DIR}/results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    log(f"\nFull diagnostics written to {OUTPUT_DIR}/results.json")
    log(f"Cached data in {CACHE_DIR}/ -- rerun any time with no extra network/BigQuery cost")


if __name__ == "__main__":
    main()