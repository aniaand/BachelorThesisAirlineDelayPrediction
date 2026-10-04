# 4.10.2026
# Author: Anna Andruszkiewicz, Claude Opus 5.5 (code)

"""
check_hnl_stations.py

Read-only check: is there a Meteostat station near Honolulu (HNL) with better
coverage of 2024-2025 than the assigned station 91180 (which only reports from
July 2025)?

Writes nothing to the project directory: no config paths, no saves, results
are only printed. (Meteostat may keep its own download cache in your user
folder, outside the project.)

Uses the same functional Meteostat API as download_meteostat.py. If a call
signature differs from what that script uses, copy the call from there.

Run cell-by-cell in VS Code or as a plain script: `python check_hnl_stations.py`.
"""

# %% Imports and settings ----------------------------------------------------
from datetime import datetime

import pandas as pd
import meteostat as ms

hnl_lat, hnl_lon, hnl_elev_m = 21.3187, -157.9224, 4   # Daniel K. Inouye Intl (HNL)
assigned_station = "91180"
extra_candidates = ["91182"]   # checked even if not among the nearest stations
n_nearest = 10
max_distance_km = 50

start = datetime(2024, 1, 1, 0, 0)
end = datetime(2025, 12, 31, 23, 0)
core_vars = ["temp", "prcp", "wspd", "cldc"]

full_index = pd.date_range(start, end, freq="h")
expected_hours = len(full_index)
print(f"Window {start:%Y-%m-%d} -> {end:%Y-%m-%d}: {expected_hours:,} hours expected")

# %% 1. Stations near HNL -----------------------------------------------------
point = ms.Point(hnl_lat, hnl_lon, hnl_elev_m)
nearby = ms.stations.nearby(point, limit=n_nearest)
print(f"\n{len(nearby)} nearest stations:")
print(nearby.to_string())

# distance column name differs between versions; convert to km if given in metres
dist_col = next((c for c in nearby.columns if "dist" in c.lower()), None)
if dist_col is not None:
    dist = nearby[dist_col].astype(float)
    nearby["distance_km"] = dist / 1000 if dist.max() > 1000 else dist
    nearby = nearby[nearby["distance_km"] <= max_distance_km]

candidate_ids = list(dict.fromkeys(
    [assigned_station] + [str(i) for i in nearby.index] + extra_candidates
))
print(f"\nStations to check: {candidate_ids}")


# %% 2. Coverage per station ---------------------------------------------------
def fetch_hourly(station_id):
    """Hourly data for one station over the window; None if nothing comes back."""
    try:
        data = ms.hourly(station_id, start, end).fetch()
    except Exception as e:  # report and move on -- one failing station shouldn't stop the check
        print(f"  {station_id}: fetch failed ({type(e).__name__}: {e})")
        return None
    if data is None or len(data) == 0:
        print(f"  {station_id}: no data in window")
        return None
    # index may be (station, time) or just time
    if isinstance(data.index, pd.MultiIndex):
        data = data.reset_index(level=0, drop=True)
    data.index = pd.to_datetime(data.index).tz_localize(None)
    return data


station_data = {}
summary_rows = []
for sid in candidate_ids:
    data = fetch_hourly(sid)
    if data is None:
        continue
    station_data[sid] = data
    hours = data.index.unique()
    row = {
        "station_id": sid,
        "first_obs": hours.min(),
        "last_obs": hours.max(),
        "coverage_total": len(hours) / expected_hours,
        "coverage_2024": (hours.year == 2024).sum() / len(full_index[full_index.year == 2024]),
        "coverage_2025_h1": ((hours.year == 2025) & (hours.month <= 6)).sum()
                            / len(full_index[(full_index.year == 2025) & (full_index.month <= 6)]),
        "coverage_2025_h2": ((hours.year == 2025) & (hours.month >= 7)).sum()
                            / len(full_index[(full_index.year == 2025) & (full_index.month >= 7)]),
    }
    for var in core_vars:
        row[f"{var}_available"] = data[var].notna().sum() / expected_hours if var in data else float("nan")
    if dist_col is not None and sid in nearby.index.astype(str):
        row["distance_km"] = float(nearby.loc[nearby.index.astype(str) == sid, "distance_km"].iloc[0])
    summary_rows.append(row)

summary = pd.DataFrame(summary_rows).set_index("station_id")
print("\nCoverage of the full window (share of hours):")
print(summary.round(3).to_string())

# %% 3. Monthly view -------------------------------------------------------------
monthly = pd.DataFrame({
    sid: d.index.unique().to_series().dt.strftime("%Y-%m").value_counts()
    for sid, d in station_data.items()
}).reindex(sorted(full_index.strftime("%Y-%m").unique())).fillna(0).astype(int)
print("\nObservations per month (about 720-744 = complete):")
print(monthly.to_string())

# %% 4. Verdict --------------------------------------------------------------------
alternatives = summary.drop(index=assigned_station, errors="ignore")
good = alternatives[alternatives["coverage_2024"] >= 0.90]

print("\n" + "=" * 70)
if good.empty:
    print("No nearby station covers 2024 (>= 90% of hours).")
    print("-> HNL is a true data limitation; state this in the limitations section.")
else:
    best = good.sort_values(["coverage_total", "temp_available"], ascending=False).index[0]
    print(f"Station(s) with >= 90% coverage of 2024: {good.index.tolist()}")
    print(f"Best: {best} -- total coverage {summary.loc[best, 'coverage_total']:.1%}, "
          f"2024 {summary.loc[best, 'coverage_2024']:.1%}"
          + (f", {summary.loc[best, 'distance_km']:.1f} km from HNL" if "distance_km" in summary
             and pd.notna(summary.loc[best].get("distance_km")) else ""))
    if assigned_station in station_data:
        union = station_data[assigned_station].index.union(station_data[best].index).unique()
        print(f"Combined with {assigned_station}: {len(union) / expected_hours:.1%} of hours covered")
    print("-> A better station exists. Switching would require retraining; the robustness "
          "check shows HNL does not change the conclusions.")
print("=" * 70)
