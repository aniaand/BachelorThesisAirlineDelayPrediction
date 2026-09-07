'''
PROMPT:

Write a Python script to download hourly Meteostat weather data for a fixed set of airports over a 2024–2025 study window, to be joined later with BTS flight data for a flight-delay-prediction pipeline.

Requirements:

Load the airport list from a retained_airports.csv file (produced by an earlier BTS processing step) containing IATA code, ICAO code, name, city, state, lat, lon.
Use hourly resolution, not daily — the target variable is a per-flight departure-delay flag, and delay-relevant weather (a storm burst, a wind gust window) can occur within an hour and would be washed out by a daily average. Each flight will later be matched to its nearest hourly observation via scheduled departure time.
For each airport, find the nearest Meteostat station using its lat/lon, and fetch hourly data for that station across the full 2024-01-01 to 2025-12-31 window.
Make the download resumable: cache each airport's data as its own file, skip airports already cached on re-run, and retry failed fetches a few times before giving up on that airport (matching the resume/skip/retry pattern already used in the BTS download pipeline).
After downloading, concatenate all cached airport files into one combined raw weather file and save it.
Report data-quality issues rather than silently proceeding: airports with no nearby station, airports where the fetch failed or returned no data, and airports whose nearest station is unusually far away (flag, don't auto-exclude, since that's a judgment call for the methodology write-up).

This script only downloads and caches raw weather data — it should not clean, aggregate, impute, or otherwise transform the fetched values; that's a separate downstream step. Use snake_case naming, # %% cell markers for VS Code, and pull paths from the project's config.py rather than hardcoding them.
'''
# 7.09.2026 21:25 CET
# Author: Anna Andruszkiewicz (code and adjustments), Claude Sonnet 5 (code)

"""
download_meteostat.py

Downloads hourly Meteostat weather data for the airports retained by
process_bts.py, over the full study window (2024-2025).

Hourly, not daily: flight delay is driven by conditions at the specific
departure hour (a short storm burst, a wind gust window), which a daily
average would smear out. Each flight will later be matched to its nearest
hourly observation via CRSDepTime.

Resumable: each airport's data is cached as its own parquet file, so a
re-run skips airports already downloaded and only retries failures/gaps.

Pipeline:
    1. Load retained_airports.csv (produced by process_bts.py)
    2. For each airport, find its nearest Meteostat station
    3. Fetch hourly data for that station over the study window
    4. Cache per-airport (skip if already cached)
    5. Concatenate everything into one combined raw weather file
    6. Report stations with no nearby match / fetch failures / large distances

Run cell-by-cell in VS Code or as a script: `python download_meteostat.py`
"""

# %% Imports and config ---------------------------------------------------
import time
from pathlib import Path

import pandas as pd
import meteostat as ms
from tqdm import tqdm

from config import raw_data_path, processed_data_path

processed_bts_dir = Path(processed_data_path) / "bts"
weather_raw_dir = Path(raw_data_path) / "weather"
weather_raw_dir.mkdir(parents=True, exist_ok=True)

buffer_days = 3
start_date = pd.Timestamp("2024-01-01") - pd.Timedelta(days=buffer_days) 
end_date = pd.Timestamp("2025-12-31")

# Flag (not exclude) stations further than this from their airport, since a
# distant station is a weaker proxy for on-site conditions.
max_station_distance_km = 50

max_retries = 3
retry_backoff_seconds = 5


# %% 1. Load retained airports ---------------------------------------------
airports = pd.read_csv(processed_bts_dir / "retained_airports.csv")
print(f"{len(airports)} retained airports loaded.")

missing_coords = airports[airports["lat"].isna() | airports["lon"].isna()]
if len(missing_coords):
    print(
        f"WARNING: {len(missing_coords)} airports have no lat/lon and will be "
        f"skipped: {missing_coords['iata'].tolist()}"
    )
airports = airports.dropna(subset=["lat", "lon"]).reset_index(drop=True)


# %% 2-4. Find nearest station + fetch hourly data, per airport (resumable) -
def fetch_airport_weather(iata: str, lat: float, lon: float) -> pd.DataFrame | None:
    """Find the nearest Meteostat station to an airport and fetch its hourly
    data for the study window. Returns None if no station / no data found."""
    point = ms.Point(lat, lon)
    nearby = ms.stations.nearby(point, limit=1)
    if nearby.empty:
        print(f"  {iata}: no Meteostat station found nearby.")
        return None

    station_id = nearby.index[0]
    distance_km = nearby.iloc[0]["distance"] / 1000  # nearby() reports meters
    if distance_km > max_station_distance_km:
        print(f"  {iata}: nearest station is {distance_km:.0f} km away (station {station_id}).")

    for attempt in range(1, max_retries + 1):
        try:
            df = ms.hourly(station_id, start_date, end_date).fetch()
            break
        except Exception as e:
            print(f"  {iata}: fetch attempt {attempt}/{max_retries} failed ({e}).")
            if attempt == max_retries:
                return None
            time.sleep(retry_backoff_seconds)

    if df.empty:
        print(f"  {iata}: station {station_id} returned no data for the study window.")
        return None

    df = df.reset_index()
    df["iata"] = iata
    df["station_id"] = station_id
    df["station_distance_km"] = distance_km
    return df


failed_airports = []
skipped_airports = []

for _, row in tqdm(airports.iterrows(), total=len(airports), desc="Downloading weather"):
    iata = row["iata"]
    cache_path = weather_raw_dir / f"{iata}.parquet"

    if cache_path.exists():
        skipped_airports.append(iata)
        continue

    df = fetch_airport_weather(iata, row["lat"], row["lon"])
    if df is None:
        failed_airports.append(iata)
        continue

    df.to_parquet(cache_path, index=False)

print(f"\nSkipped (already cached): {len(skipped_airports)}")
print(f"Failed / no data: {len(failed_airports)} -> {failed_airports}")


# %% 5. Concatenate all cached airport files into one combined file --------
cached_files = sorted(weather_raw_dir.glob("*.parquet"))
print(f"\n{len(cached_files)} cached airport weather files found.")

weather_raw = pd.concat(
    [pd.read_parquet(f) for f in cached_files], ignore_index=True
)
print(f"Combined hourly weather: {len(weather_raw):,} rows, {weather_raw['iata'].nunique()} airports.")

output_path = weather_raw_dir / "weather_hourly_raw.parquet"
weather_raw.to_parquet(output_path, index=False)
print(f"Saved combined file to {output_path}")

# %% 6. Coverage check -------------------------------------------------------
missing_airports = set(airports["iata"]) - set(weather_raw["iata"].unique())
if missing_airports:
    print(f"\nAirports with NO weather data at all: {sorted(missing_airports)}")

far_stations = (
    weather_raw.drop_duplicates("iata")[["iata", "station_id", "station_distance_km"]]
    .query("station_distance_km > @max_station_distance_km")
    .sort_values("station_distance_km", ascending=False)
)
if len(far_stations):
    print(f"\n{len(far_stations)} airports matched to a station > {max_station_distance_km} km away:")
    print(far_stations.to_string(index=False))
# %%
