'''
PROMPT:
Write `scripts/summary_stats.py`.

Purpose: produce the descriptive statistics for the data chapter. It runs once, after the pipeline is finished, and saves every table through the shared helper `save_appendix_table()` in `scripts/appendix_utils.py`, which writes each table as a CSV and a booktabs LaTeX file.

Context:
- Target: `dep_del15` (departure delay ≥ 15 min, binary).
- Training data is 2024; the test holdout is all of 2025.
- Optional inputs: the layer D TabPFN subsample and the processed hourly Meteostat file. Skip any part that needs one of these if the file is missing.

Outputs:
- T3.3 Target distribution: flights, delayed flights and delay rate for 2024 (train), 2025 (holdout) and the TabPFN subsample.
- T3.4 Weather variables (12h aggregates of temperature, precipitation, wind speed and cloud cover): mean, sd, min, max, % missing at flight level and at station-hour level.
- T3.5 GDELT records by CAMEO root code (14/17/18/20), using previous-day counts per airport-day: records, share, % of airport-days with at least one event, mean records per airport-day, plus a total row. The caption must note that records reflect news coverage, not discrete events.
- F3.1 Delay rate by scheduled departure hour and by month, with one line each for 2024 and 2025.
- F3.2 Delay rate by 12h precipitation bin (training period), showing n per bar and the overall 2024 rate as a reference line.
- F3.3 Excess delay rate by GDELT quintile (prior 3 days), ranked within airport and month, in percentage points vs. the airport-month average (training period).

Requirements:
- Read paths from `scripts/config.py`. Don't hard-code any paths.
- Follow the repo pattern: load → compute → save, with `# %%` cell markers so I can run it cell by cell in VS Code.
- The full dataset is about 16M rows, so read only the columns each table needs (parquet `columns=`).
- Statistics must only describe the data. Don't fit or transform anything.
- Give each table a clear name, a caption and a label `tab:<name>`.
- Keep the code short and readable, in snake_case, with pandas only.
- At the end, print a list of all saved tables.
'''
# 30.09.2026 CET
# Author: Anna Andruszkiewicz (code and adjustments), Claude Sonnet 5 (code)

# %% imports
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy import stats

from config import processed_data_path

# %% settings: paths
merged_file = Path(processed_data_path) / "merged" / "layer_d_bts_meteostat_gdelt.parquet"
processed_meteostat_file = Path(processed_data_path) / "weather" / "weather_processed.parquet"  # optional
subsample_file = Path(processed_data_path) / "merged" / "layer_d_bts_meteostat_gdelt_subsampled.parquet"  # optional

out_dir = Path(processed_data_path) / "stats"
table_dir = out_dir / "tables"
fig_dir = out_dir / "figures"
table_dir.mkdir(parents=True, exist_ok=True)
fig_dir.mkdir(parents=True, exist_ok=True)

# %% settings: columns
target = "dep_del15"
date_col = "flight_date"
year_col = "year"
month_col = "month"
dep_time_col = "crs_dep_time"
origin_col = "origin"
gdelt_prefix = "gdelt_"

# merged column -> (label, unit, hourly column in processed meteostat)
weather_vars = {
    "temp_mean_12h": ("Temperature, 12h mean", "°C", "temp"),
    "prcp_sum_12h": ("Precipitation, 12h sum", "mm", "prcp"),
    "wspd_mean_12h": ("Wind speed, 12h mean", "km/h", "wspd"),
    "cldc_mean_12h": ("Cloud cover, 12h mean", "oktas", "cldc"),
}

# gdelt category in column name -> (CAMEO root code, label)
cameo_categories = {
    "protest": (14, "PROTEST"),
    "coerce": (17, "COERCE"),
    "assault": (18, "ASSAULT"),
    "mass_violence": (20, "MASS VIOLENCE"),
}

# %% settings: study design
train_year = 2024
train_label, holdout_label = "train_2024", "holdout_2025"
period_labels = {train_label: "2024 (train)", holdout_label: "2025 (holdout)"}

prcp_bins = [-np.inf, 0, 1, 5, 10, 25, np.inf]
prcp_bin_labels = ["0", "0–1", "1–5", "5–10", "10–25", ">25"]

# exploratory binary weather indicators (Meteostat units: °C, mm)
weather_indicators = {
    "any_precipitation_12h": ("prcp_sum_12h", lambda s: s > 0),
    "freezing_12h": ("temp_mean_12h", lambda s: s <= 0),
}


# %% helpers
def require(cols, available, source):
    missing = [c for c in cols if c not in available]
    assert not missing, f"{source}: missing columns {missing}\navailable: {list(available)}"


def fmt_p(p):
    if pd.isna(p):
        return ""
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def save_table(table, name, caption, p_cols=(), float_format="%.3f"):
    table.to_csv(table_dir / f"{name}.csv", index=False)
    tex = table.copy()
    for c in tex.columns:
        if c in p_cols:
            tex[c] = tex[c].map(fmt_p)
        elif pd.api.types.is_integer_dtype(tex[c]):
            tex[c] = tex[c].map("{:,}".format)
    tex.to_latex(table_dir / f"{name}.tex", index=False, float_format=float_format,
                 caption=caption, label=f"tab:{name}", escape=True)
    print(f"saved {name}")


def save_fig(fig, name):
    fig.savefig(fig_dir / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {name}")


# %% load merged layer d
merged_cols = pq.read_schema(merged_file).names
gdelt_cols = [c for c in merged_cols if c.startswith(gdelt_prefix)]
gdelt_lag1d_cols = {cat: f"gdelt_{cat}_count_lag1d" for cat in cameo_categories}
base_cols = [target, date_col, year_col, month_col, dep_time_col, origin_col]
require(base_cols + list(weather_vars) + list(gdelt_lag1d_cols.values()), merged_cols, merged_file.name)

df = pd.read_parquet(merged_file, columns=base_cols + list(weather_vars) + gdelt_cols)
df[date_col] = pd.to_datetime(df[date_col])
df["period"] = np.where(df[year_col] == train_year, train_label, holdout_label)
train = df.loc[df["period"] == train_label]
y = train[target]

# %% load optional files
subsample = None
if subsample_file.exists():
    require([target], pq.read_schema(subsample_file).names, subsample_file.name)
    subsample = pd.read_parquet(subsample_file, columns=[target])

met = None
if processed_meteostat_file.exists():
    met_schema = pq.read_schema(processed_meteostat_file).names
    met_hourly = [h for _, _, h in weather_vars.values() if h in met_schema]
    met = pd.read_parquet(processed_meteostat_file, columns=met_hourly)

# %% audit
print(f"merged rows: {len(df):,} | airports: {df[origin_col].nunique()}")
print(df["period"].value_counts().to_string())
print(f"subsample: {'not found, skipped' if subsample is None else f'{len(subsample):,} rows'}")
print(f"processed meteostat: {'not found, skipped' if met is None else f'{len(met):,} station-hours'}")
assert set(df[target].dropna().unique()) <= {0, 1}, "target is not binary"

# gdelt lag counts must be constant within airport-day for the airport-day aggregation in T3.5
max_nunique = df.groupby([origin_col, date_col])[list(gdelt_lag1d_cols.values())].nunique().max().max()
assert max_nunique <= 1, "gdelt lag1d counts vary within airport-day; T3.5 aggregation invalid"

# %% T3.3 target distribution
target_rows = []
for period, g in df.groupby("period"):
    target_rows.append((period_labels[period], g[target].count(), g[target].sum(), g[target].mean()))
if subsample is not None:
    s = subsample[target]
    target_rows.append(("TabPFN-3 subsample", s.count(), s.sum(), s.mean()))
t3_3 = pd.DataFrame(target_rows, columns=["sample", "flights", "delayed", "delay_rate_pct"])
t3_3["delayed"] = t3_3["delayed"].astype(int)
t3_3["delay_rate_pct"] *= 100
save_table(t3_3, "t3_3_target_distribution",
           "Distribution of the target dep\\_del15 (departure delay $\\geq$ 15 min)", float_format="%.2f")

# %% T3.4 weather variables
weather_rows = []
for col, (label, unit, hourly) in weather_vars.items():
    s = df[col]
    row = {"variable": label, "unit": unit,
           "mean": s.mean(), "sd": s.std(), "min": s.min(), "max": s.max(),
           "missing_pct_flights": 100 * s.isna().mean()}
    if met is not None:
        row["missing_pct_station_hours"] = 100 * met[hourly].isna().mean() if hourly in met else np.nan
    weather_rows.append(row)
t3_4 = pd.DataFrame(weather_rows)
save_table(t3_4, "t3_4_weather_variables",
           "Meteostat weather variables at flight level, 2024--2025", float_format="%.2f")

# %% T3.5 gdelt events by cameo root code (airport-day level, previous-day counts)
airport_days = df.drop_duplicates([origin_col, date_col])[list(gdelt_lag1d_cols.values())]
n_airport_days = len(airport_days)

gdelt_rows = []
for cat, (code, label) in cameo_categories.items():
    counts = airport_days[gdelt_lag1d_cols[cat]]
    gdelt_rows.append({"cameo_root": str(code), "category": label,
                       "records": int(counts.sum()),
                       "airport_days_with_event_pct": 100 * (counts > 0).mean(),
                       "mean_records_per_airport_day": counts.mean()})
t3_5 = pd.DataFrame(gdelt_rows)
t3_5.insert(3, "share_pct", 100 * t3_5["records"] / t3_5["records"].sum())
t3_5 = t3_5.sort_values("records", ascending=False)

total_counts = airport_days.sum(axis=1)
t3_5 = pd.concat([t3_5, pd.DataFrame([{
    "cameo_root": "", "category": "TOTAL", "records": int(total_counts.sum()), "share_pct": 100.0,
    "airport_days_with_event_pct": 100 * (total_counts > 0).mean(),
    "mean_records_per_airport_day": total_counts.mean(),
}])], ignore_index=True)
save_table(t3_5, "t3_5_gdelt_cameo",
           f"GDELT event records by CAMEO root code per airport-day "
           f"({df[origin_col].nunique()} airports, {n_airport_days:,} airport-days, 2024--2025). "
           f"Records reflect news coverage, not discrete events.",
           float_format="%.2f")

# %% F3.1 delay rate by hour and month
dep_hour = (pd.to_numeric(df[dep_time_col], errors="coerce") // 100) % 24
panels = [(dep_hour, "Scheduled departure hour", "hour", range(0, 24, 3)),
          (df[month_col], "Month", "month", range(1, 13))]

fig, axes = plt.subplots(1, 2, figsize=(9, 3.2), sharey=True)
for ax, (key, xlabel, name, ticks) in zip(axes, panels):
    rates = df.groupby([key.rename(name), "period"])[target].mean().unstack("period")
    rates.reset_index().to_csv(fig_dir / f"f3_1_delay_rate_by_{name}.csv", index=False)
    for period in rates.columns:
        ax.plot(rates.index, 100 * rates[period], marker="o", ms=3, label=period_labels[period])
    ax.set_xlabel(xlabel)
    ax.set_xticks(list(ticks))
    ax.grid(alpha=0.3)
axes[0].set_ylabel("Delay rate (%)")
axes[0].legend(frameon=False)
save_fig(fig, "f3_1_delay_rate_hour_month")

# %% F3.2 delay rate by precipitation bins (training period)
prcp_bin = pd.cut(train["prcp_sum_12h"], bins=prcp_bins, labels=prcp_bin_labels)
f3_2 = (train.groupby(prcp_bin, observed=False)[target]
             .agg(flights="count", delay_rate="mean")
             .reset_index()
             .rename(columns={"prcp_sum_12h": "prcp_bin_mm"}))
f3_2.to_csv(fig_dir / "f3_2_delay_rate_by_precipitation.csv", index=False)

fig, ax = plt.subplots(figsize=(5, 3.2))
bars = ax.bar(f3_2["prcp_bin_mm"].astype(str), 100 * f3_2["delay_rate"], color="steelblue")
for bar, n in zip(bars, f3_2["flights"]):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"n={n / 1e3:,.0f}k",
            ha="center", va="bottom", fontsize=7)
ax.axhline(100 * y.mean(), ls="--", lw=1, color="grey", label="overall 2024 rate")
ax.set_xlabel("Precipitation, 12h sum (mm)")
ax.set_ylabel("Delay rate (%)")
ax.legend(frameon=False, fontsize=8)
ax.grid(axis="y", alpha=0.3)
save_fig(fig, "f3_2_delay_rate_precipitation")

# %% F3.3 excess delay rate by GDELT coverage, within airport (training period)
# Counts are ranked within each airport, so each airport is compared with itself.
# y-axis: delay rate minus the airport's own 2024 delay rate, in percentage points.
gdelt_fig_col = "gdelt_total_count_lag3d"
quintile_labels = ["Q1 (lowest)", "Q2", "Q3", "Q4", "Q5 (highest)"]

within_rank = train.groupby([origin_col, month_col])[gdelt_fig_col].rank(pct=True, method="average")
gdelt_bin = pd.cut(within_rank, bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
                   labels=quintile_labels, include_lowest=True)
excess = y - train.groupby([origin_col, month_col])[target].transform("mean")

f3_3 = (pd.DataFrame({"bin": gdelt_bin, "excess": excess})
        .groupby("bin", observed=False)["excess"]
        .agg(flights="count", excess_delay_pp="mean")
        .reset_index())
f3_3["excess_delay_pp"] *= 100
f3_3.to_csv(fig_dir / "f3_3_excess_delay_by_gdelt.csv", index=False)

fig, ax = plt.subplots(figsize=(5, 3.2))
ax.bar(f3_3["bin"].astype(str), f3_3["excess_delay_pp"], color="steelblue")
ax.axhline(0, lw=1, color="grey")
ax.set_xlabel("GDELT event records, prior 3 days (quintile within airport & month)")
ax.set_ylabel("Delay rate vs. airport-month average (pp)")
ax.grid(axis="y", alpha=0.3)
save_fig(fig, "f3_3_excess_delay_gdelt")

# %% done
print(f"\nall outputs written to {out_dir}")