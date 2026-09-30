'''
Write a Python script that produces subsampled versions of the four merged
information-layer datasets (layer_a_bts, layer_b_bts_meteostat, layer_c_bts_gdelt,
layer_d_bts_meteostat_gdelt).

Design constraints this must respect:
- Only the 2024 training period is ever subsampled. The 2025 holdout stays
  completely untouched in every layer's subsampled file.
- The sample must be the same ~1,000,000 flights across all four layers (by
  flight_id), not four independently drawn subsamples.
- The sample must be jointly stratified on (origin, dep_del15).

It should:
1. Load layer_a_bts.parquet (flight_id, origin, dep_del15, year are all that's
   needed here) and restrict to year == 2024.
2. Compute a single sampling fraction = target sample size / number of 2024 rows,
   and draw a stratified sample of that target size using scikit-learn's
   train_test_split with a combined (origin, dep_del15) stratify key, so the joint
   proportions are preserved and the sample lands on the exact target size. Report
   the actual sample size achieved and the number of (origin, dep_del15) groups
   that ended up with zero sampled rows, if any.
3. Save the selected flight_id set on its own, for reproducibility/auditing.
4. For each of the four layer parquet files: keep every row where year == 2025
   unchanged, and keep only the 2024 rows whose flight_id is in the selected set.
   Save each result as a new "_subsampled" parquet file alongside the existing
   full one. Report before/after row counts per layer, split by year, so it's
   easy to confirm 2025 truly didn't change size.

Use snake_case naming, # %% cell markers for VS Code, and pull paths from the
central config.py rather than hardcoding them. No feature engineering here --
this only selects which rows survive into the subsampled files.
'''
# 9.09.2026 CET
# Author: Anna Andruszkiewicz (code and adjustments), Claude Sonnet 5 (code)

# %% Imports & config
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from config import processed_data_path
from appendix_utils import save_appendix_table

processed_data_path = Path(processed_data_path)
merged_dir = processed_data_path / "merged"

target_sample_size = 1_000_000
random_state= 42

layer_files = [
    "layer_a_bts",
    "layer_b_bts_meteostat",
    "layer_c_bts_gdelt",
    "layer_d_bts_meteostat_gdelt",
]

# %% 1. Load Layer A just to build the stratified sample of 2024 flight_ids -----------
layer_a = pd.read_parquet(
    merged_dir / "layer_a_bts.parquet", columns=["flight_id", "origin", "dep_del15", "year"]
)
train_2024 = layer_a[layer_a["year"] == 2024]
print(f"2024 training rows (full): {len(train_2024):,}")

# %% 2. Stratified sample — sklearn's train_test_split with a combined
# (origin, dep_del15) stratify key, so the joint distribution is preserved and the
# sample lands on the exact target size rather than an approximate fraction.
strata = train_2024["origin"].astype(str) + "_" + train_2024["dep_del15"].astype(str)

sampled_2024, _ = train_test_split(
    train_2024,
    train_size=target_sample_size,
    stratify=strata,
    random_state=random_state,
)
print(f"Sampled 2024 rows: {len(sampled_2024):,} ({len(sampled_2024) / len(train_2024):.2%} of 2024)")

group_sizes = train_2024.groupby(["origin", "dep_del15"]).size()
sampled_group_sizes = sampled_2024.groupby(["origin", "dep_del15"]).size()
zero_groups = group_sizes.index.difference(sampled_group_sizes.index)
if len(zero_groups):
    print(f"WARNING: {len(zero_groups)} (origin, dep_del15) groups sampled zero rows: {list(zero_groups)}")
else:
    print("OK: every (origin, dep_del15) group has at least one sampled row.")

selected_flight_ids = set(sampled_2024["flight_id"])

# %% 2b. Appendix B.3 (part) -- subsample summary --------------------------------
# Largest absolute gap between a stratum's share in the full 2024 data and in the
# sample, in percentage points: shows the stratification held.
full_share = group_sizes / len(train_2024)
sample_share = sampled_group_sizes.reindex(full_share.index, fill_value=0) / len(sampled_2024)
max_share_gap_pp = (sample_share - full_share).abs().max() * 100

b3_subsample = pd.DataFrame(
    [
        ("Training rows, full 2024", f"{len(train_2024):,}"),
        ("Sampled rows", f"{len(sampled_2024):,}"),
        ("Sampling fraction", f"{len(sampled_2024) / len(train_2024):.2%}"),
        ("Method", "stratified, sklearn train_test_split"),
        ("Stratification key", "origin x dep_del15"),
        ("Strata", f"{len(group_sizes):,}"),
        ("Strata with zero sampled rows", f"{len(zero_groups):,}"),
        ("Max. stratum share gap (pp)", f"{max_share_gap_pp:.4f}"),
        ("Delay rate, full 2024", f"{train_2024['dep_del15'].mean():.2%}"),
        ("Delay rate, sample", f"{sampled_2024['dep_del15'].mean():.2%}"),
        ("Random seed", str(random_state)),
        ("2025 holdout", "not subsampled"),
    ],
    columns=["item", "value"],
)
save_appendix_table(
    b3_subsample,
    "b3_subsample_summary",
    "Construction of the 2024 training subsample used for TabPFN-3 and the "
    "same-sample classical ML comparison.",
)

# %% 3. Save the selected flight_id set for reproducibility/auditing ------------------
ids_path = merged_dir / "subsample_2024_flight_ids.parquet"
sampled_2024[["flight_id"]].to_parquet(ids_path, index=False)
print(f"Saved selected flight_ids: {len(selected_flight_ids):,} -> {ids_path}")

# %% 4. Apply the same flight_id set to all four layers, 2025 left untouched ----------
for name in layer_files:
    path = merged_dir / f"{name}.parquet"
    df = pd.read_parquet(path)

    before_2024 = (df["year"] == 2024).sum()
    before_2025 = (df["year"] == 2025).sum()

    df_sub = df[(df["year"] == 2025) | (df["flight_id"].isin(selected_flight_ids))].copy()

    after_2024 = (df_sub["year"] == 2024).sum()
    after_2025 = (df_sub["year"] == 2025).sum()

    print(
        f"{name}: 2024 {before_2024:,} -> {after_2024:,}, "
        f"2025 {before_2025:,} -> {after_2025:,} (should be unchanged)"
    )

    out_path = merged_dir / f"{name}_subsampled.parquet"
    df_sub.to_parquet(out_path, index=False)
    print(f"Saved {name}_subsampled: {df_sub.shape} -> {out_path}")