'''
Write a Python script (not a notebook) to process raw BTS On-Time Performance data for a flight delay prediction pipeline. It should:

Load monthly BTS files directly from their .zip archives (no separate unzip step) and merge all months into one DataFrame.
Print every column available in the raw data, so the schema can be audited against expectations.
Subset the data to only the columns needed downstream (operational fields relevant to delay prediction — carrier, origin/dest, scheduled and actual times, delay causes, cancellation/diversion flags — but not yet trimmed for modeling-stage leakage, since that happens later at feature engineering).
Report the percentage of missing values per column.
Compute, per origin airport, the minimum and average flights per day across the data window, and print the distribution so a sensible cutoff can be chosen.
Filter to airports meeting a minimum-flights-per-day floor (use the minimum, not the average — the floor should hold on an airport's worst day, not just on average, since downstream lag/rolling features and target encoding need consistent daily volume).
Map ORIGIN/DEST IATA codes to ICAO codes where not already present (needed for a later Meteostat weather join), flagging any codes that fail to map.
Save the filtered result to a processed-data location and print a final summary (rows, columns, airports retained).

The script must not modify, impute, recompute, or otherwise alter any existing data values — this is a read/audit/filter/save pipeline only (dropping unneeded columns and rows for unqualifying airports is fine; changing what's inside a retained cell/row is not). Any new columns added (e.g. the ICAO mapping) must be clearly additive, never overwriting a source field.

Use snake_case for all Python variable/function names (source data column names stay as BTS provides them, e.g. FL_DATE). Structure the script with # %% cell markers so it can be run cell-by-cell in VS Code. Paths should come from a project config.py (raw_data_path, processed_data_path), not be hardcoded.
'''
# 7.09.2026 20:15 CET
# Author: Anna Andruszkiewicz (code and adjustments), Claude Sonnet 5 (code)

"""
process_bts_data.py

Stage 1 cleaning/audit pass over raw BTS On-Time Performance data.

Pipeline:
    1. Load raw monthly BTS files
    2. Audit available columns
    3. Subset to the columns needed for the thesis
    4. Audit missingness per column
    5. Compute flights/day per airport (min + average) and print the distribution
    6. Filter to airports meeting a minimum-flights-per-day floor
    7. Map ORIGIN/DEST IATA codes to ICAO codes (needed later for Meteostat joins),
       skipping any that are already present

Run cell-by-cell in VS Code (Code Runner respects the "# %%" markers) or as a
plain script: `python process_bts_data.py`.

Adjust RAW_DATA_DIR / PROCESSED_DATA_DIR to match config.py before running.
"""

# %% Imports and config -------------------------------------------------
import zipfile
from pathlib import Path

import pandas as pd
from config import raw_data_path, processed_data_path

raw_data_dir = Path(raw_data_path) / "bts"
processed_data_dir = Path(processed_data_path) / "bts"
processed_data_dir.mkdir(parents=True, exist_ok=True)

# Minimum flights/day an airport must have, on its WORST day in the window,
# to be retained.
min_flights_per_day = 10

# Columns needed downstream (raw operational layer). This is intentionally
# broader than the final modeling feature set -- leakage-relevant trimming
# (e.g. dropping ARR_DELAY, actual times) happens later at feature
# engineering, not here.
keep_cols = [
    "FlightDate",
    "Year",
    "Month",
    "DayOfWeek",
    "Reporting_Airline",
    "Flight_Number_Reporting_Airline",
    "Origin",
    "OriginCityName",
    "OriginState",
    "Dest",
    "DestCityName",
    "DestState",
    "CRSDepTime",
    "DepTime",
    "DepDelay",
    "DepDel15",
    "TaxiOut",
    "WheelsOff",
    "WheelsOn",
    "TaxiIn",
    "CRSArrTime",
    "ArrTime",
    "ArrDelay",
    "ArrDel15",
    "Cancelled",
    "CancellationCode",
    "Diverted",
    "CRSElapsedTime",
    "ActualElapsedTime",
    "AirTime",
    "Distance",
    "CarrierDelay",
    "WeatherDelay",
    "NASDelay",
    "SecurityDelay",
    "LateAircraftDelay",
]


# %% 1. Load raw files ---------------------------------------------------
def read_csv_from_zip(zip_path: Path) -> pd.DataFrame:
    """Read the CSV member out of a single monthly BTS zip archive.

    BTS zips normally contain exactly one CSV plus a readme; if a zip
    contains more than one CSV this raises so you can inspect it rather
    than silently picking the wrong file.
    """
    with zipfile.ZipFile(zip_path) as zf:
        csv_names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        if not csv_names:
            raise ValueError(f"No CSV found inside {zip_path.name}")
        if len(csv_names) > 1:
            raise ValueError(
                f"{zip_path.name} contains multiple CSVs {csv_names} -- "
                "inspect manually before proceeding."
            )
        with zf.open(csv_names[0]) as f:
            return pd.read_csv(f, low_memory=False)


def load_raw_bts(raw_dir: Path) -> pd.DataFrame:
    """Load and concatenate all monthly BTS zip archives found in raw_dir."""
    zip_paths = sorted(raw_dir.glob("*.zip"))
    if not zip_paths:
        raise FileNotFoundError(f"No .zip files found in {raw_dir}")

    print(f"Found {len(zip_paths)} monthly zip archives.")
    frames = []
    for zip_path in zip_paths:
        df_month = read_csv_from_zip(zip_path)
        frames.append(df_month)
        print(f"  {zip_path.name}: {len(df_month):,} rows")

    df = pd.concat(frames, ignore_index=True)
    print(f"Loaded {len(df):,} rows total.")
    return df

bts_raw = load_raw_bts(raw_data_dir)

# %% 2. Audit available columns ------------------------------------------
print(f"\n{len(bts_raw.columns)} columns available:")
for col in bts_raw.columns:
    print(f"  - {col}")

missing_from_keep = [c for c in keep_cols if c not in bts_raw.columns]
if missing_from_keep:
    print(f"\nWARNING: expected columns not found in raw data: {missing_from_keep}")
    print("BTS occasionally renames fields between vintages -- check schema.")

# %% 3. Subset to needed columns ------------------------------------------
present_keep_cols = [c for c in keep_cols if c in bts_raw.columns]
bts = bts_raw[present_keep_cols].copy()
print(f"\nSubset to {len(bts.columns)} columns, {len(bts):,} rows.")

# %% 4. Missingness audit --------------------------------------------------
missing_pct = (bts.isna().mean() * 100).round(2).sort_values(ascending=False)
print("\nMissing value % per column:")
print(missing_pct.to_string())

# %% 5. Flights/day per airport --------------------------------------------
bts["fl_date"] = pd.to_datetime(bts["FlightDate"], errors="coerce")

flights_per_day = (
    bts.groupby(["Origin", "fl_date"]).size().rename("n_flights").reset_index()
)

airport_flight_stats = (
    flights_per_day.groupby("Origin")["n_flights"]
    .agg(min_flights_per_day="min", avg_flights_per_day="mean", n_days_active="count")
    .sort_values("min_flights_per_day", ascending=False)
)

n_airports_total = airport_flight_stats.shape[0]
print(f"\n{n_airports_total} distinct origin airports in the data.")
print("\nFlights/day distribution across airports:")
print(airport_flight_stats[["min_flights_per_day", "avg_flights_per_day"]].describe())

# %% 6. Filter to airports meeting the minimum-flights-per-day floor -------
qualifying_airports = airport_flight_stats[
    airport_flight_stats["min_flights_per_day"] >= min_flights_per_day
].index

n_qualifying = len(qualifying_airports)
print(
    f"\n{n_qualifying} / {n_airports_total} airports have "
    f">= {min_flights_per_day} flights on their worst day "
    f"({n_qualifying / n_airports_total:.1%})."
)

bts_filtered = bts[
    bts["Origin"].isin(qualifying_airports) & bts["Dest"].isin(qualifying_airports)
].copy()

print(
    f"Rows retained after airport filter: {len(bts_filtered):,} "
    f"({len(bts_filtered) / len(bts):.1%} of subset)."
)

# %% 7. Map ORIGIN/DEST codes to a universal (ICAO) code -------------------
# BTS ORIGIN/DEST are already 3-letter IATA codes. Meteostat station lookups
# key off ICAO codes (4-letter) or lat/lon, so an ICAO column is added now
# rather than re-derived at the weather-join stage.

try:
    import airportsdata

    iata_to_icao = {
        rec["iata"]: icao
        for icao, rec in airportsdata.load("ICAO").items()
        if rec.get("iata")
    }

    def map_iata_to_icao(df: pd.DataFrame, col: str) -> pd.Series:
        mapped = df[col].map(iata_to_icao)
        n_unmapped = mapped.isna().sum()
        if n_unmapped:
            unmapped_codes = sorted(df.loc[mapped.isna(), col].unique())
            print(
                f"WARNING: {n_unmapped} rows in {col} had no ICAO match "
                f"({len(unmapped_codes)} unique codes): {unmapped_codes[:10]}"
                f"{'...' if len(unmapped_codes) > 10 else ''}"
            )
        return mapped

    if "ORIGIN_ICAO" not in bts_filtered.columns:
        bts_filtered["Origin_ICAO"] = map_iata_to_icao(bts_filtered, "Origin")
    if "DEST_ICAO" not in bts_filtered.columns:
        bts_filtered["Dest_ICAO"] = map_iata_to_icao(bts_filtered, "Dest")

except ImportError:
    print(
        "\nairportsdata not installed -- skipping IATA->ICAO mapping.\n"
        "Install with: pip install airportsdata"
    )

# %% 8. Save + summary ------------------------------------------------------
output_path = processed_data_dir / "bts_processed.parquet"
bts_filtered.to_parquet(output_path, index=False)

print(f"\nSaved {len(bts_filtered):,} rows, {len(bts_filtered.columns)} columns to {output_path}")
print(f"Airports retained: {n_qualifying}")

# %%
