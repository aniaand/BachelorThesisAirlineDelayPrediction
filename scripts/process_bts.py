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
from pathlib import Path

import pandas as pd

# TODO: point these at the actual paths from your config.py
RAW_DATA_DIR = Path("data/raw/bts")
PROCESSED_DATA_DIR = Path("data/processed")
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

# Minimum flights/day an airport must have, on its WORST day in the window,
# to be retained. See discussion in thesis methodology re: min vs. average.
MIN_FLIGHTS_PER_DAY = 10

# Columns needed downstream (raw operational layer). This is intentionally
# broader than the final modeling feature set -- leakage-relevant trimming
# (e.g. dropping ARR_DELAY, actual times) happens later at feature
# engineering, not here.
KEEP_COLS = [
    "FL_DATE",
    "YEAR",
    "MONTH",
    "DAY_OF_WEEK",
    "OP_UNIQUE_CARRIER",
    "OP_CARRIER_FL_NUM",
    "ORIGIN",
    "ORIGIN_CITY_NAME",
    "ORIGIN_STATE_ABR",
    "DEST",
    "DEST_CITY_NAME",
    "DEST_STATE_ABR",
    "CRS_DEP_TIME",
    "DEP_TIME",
    "DEP_DELAY",
    "DEP_DEL15",
    "TAXI_OUT",
    "WHEELS_OFF",
    "WHEELS_ON",
    "TAXI_IN",
    "CRS_ARR_TIME",
    "ARR_TIME",
    "ARR_DELAY",
    "ARR_DEL15",
    "CANCELLED",
    "CANCELLATION_CODE",
    "DIVERTED",
    "CRS_ELAPSED_TIME",
    "ACTUAL_ELAPSED_TIME",
    "AIR_TIME",
    "DISTANCE",
    "CARRIER_DELAY",
    "WEATHER_DELAY",
    "NAS_DELAY",
    "SECURITY_DELAY",
    "LATE_AIRCRAFT_DELAY",
]


# %% 1. Load raw files ---------------------------------------------------
def load_raw_bts(raw_dir: Path) -> pd.DataFrame:
    """Load and concatenate all monthly BTS CSVs found in raw_dir."""
    csv_paths = sorted(raw_dir.glob("*.csv"))
    if not csv_paths:
        raise FileNotFoundError(f"No CSV files found in {raw_dir}")

    print(f"Found {len(csv_paths)} monthly files.")
    frames = [pd.read_csv(p, low_memory=False) for p in csv_paths]
    df = pd.concat(frames, ignore_index=True)
    print(f"Loaded {len(df):,} rows total.")
    return df


bts_raw = load_raw_bts(RAW_DATA_DIR)

# %% 2. Audit available columns ------------------------------------------
print(f"\n{len(bts_raw.columns)} columns available:")
for col in bts_raw.columns:
    print(f"  - {col}")

missing_from_keep = [c for c in KEEP_COLS if c not in bts_raw.columns]
if missing_from_keep:
    print(f"\nWARNING: expected columns not found in raw data: {missing_from_keep}")
    print("BTS occasionally renames fields between vintages -- check schema.")

# %% 3. Subset to needed columns ------------------------------------------
present_keep_cols = [c for c in KEEP_COLS if c in bts_raw.columns]
bts = bts_raw[present_keep_cols].copy()
print(f"\nSubset to {len(bts.columns)} columns, {len(bts):,} rows.")

# %% 4. Missingness audit --------------------------------------------------
missing_pct = (bts.isna().mean() * 100).round(2).sort_values(ascending=False)
print("\nMissing value % per column:")
print(missing_pct.to_string())

# %% 5. Flights/day per airport --------------------------------------------
bts["fl_date"] = pd.to_datetime(bts["FL_DATE"])

flights_per_day = (
    bts.groupby(["ORIGIN", "fl_date"]).size().rename("n_flights").reset_index()
)

airport_flight_stats = (
    flights_per_day.groupby("ORIGIN")["n_flights"]
    .agg(min_flights_per_day="min", avg_flights_per_day="mean", n_days_active="count")
    .sort_values("min_flights_per_day", ascending=False)
)

n_airports_total = airport_flight_stats.shape[0]
print(f"\n{n_airports_total} distinct origin airports in the data.")
print("\nFlights/day distribution across airports:")
print(airport_flight_stats[["min_flights_per_day", "avg_flights_per_day"]].describe())

# %% 6. Filter to airports meeting the minimum-flights-per-day floor -------
qualifying_airports = airport_flight_stats[
    airport_flight_stats["min_flights_per_day"] >= MIN_FLIGHTS_PER_DAY
].index

n_qualifying = len(qualifying_airports)
print(
    f"\n{n_qualifying} / {n_airports_total} airports have "
    f">= {MIN_FLIGHTS_PER_DAY} flights on their worst day "
    f"({n_qualifying / n_airports_total:.1%})."
)

bts_filtered = bts[
    bts["ORIGIN"].isin(qualifying_airports) & bts["DEST"].isin(qualifying_airports)
].copy()

print(
    f"Rows retained after airport filter: {len(bts_filtered):,} "
    f"({len(bts_filtered) / len(bts):.1%} of subset)."
)

# %% 7. Map ORIGIN/DEST codes to a universal (ICAO) code -------------------
# BTS ORIGIN/DEST are already 3-letter IATA codes. Meteostat station lookups
# key off ICAO codes (4-letter) or lat/lon, so we add an ICAO column now
# rather than re-deriving it at the weather-join stage.
#
# Requires: pip install airportsdata
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
        bts_filtered["ORIGIN_ICAO"] = map_iata_to_icao(bts_filtered, "ORIGIN")
    if "DEST_ICAO" not in bts_filtered.columns:
        bts_filtered["DEST_ICAO"] = map_iata_to_icao(bts_filtered, "DEST")

except ImportError:
    print(
        "\nairportsdata not installed -- skipping IATA->ICAO mapping.\n"
        "Install with: pip install airportsdata"
    )

# %% 8. Save + summary ------------------------------------------------------
output_path = PROCESSED_DATA_DIR / "bts_filtered.parquet"
bts_filtered.to_parquet(output_path, index=False)

print(f"\nSaved {len(bts_filtered):,} rows, {len(bts_filtered.columns)} columns to {output_path}")
print(f"Airports retained: {n_qualifying}")