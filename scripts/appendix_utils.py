'''
PROMPT

Write `scripts/appendix_utils.py`.

Purpose: one shared helper that saves the appendix tables. Each table is written twice to `<processed_data_path>/appendix/`:
1. a CSV, so I can check the numbers
2. a LaTeX table using booktabs, which I pull into the paper directly.

These scripts will import it: `process_bts.py`, `process_meteostat.py`, `process_gdelt.py`, `merge_layers.py`, `subsample.py`, `feature_engineering.py`.

Requirements:
- Read `processed_data_path` from `scripts/config.py`. Don't hard-code any paths. Create the `appendix/` folder if it doesn't exist.
- Main function: `save_appendix_table(df, name, caption, label, ...)`. It writes `<name>.csv` and `<name>.tex` and returns both paths.
- LaTeX output:
  - Use `df.to_latex()` with booktabs rules (`\toprule`, `\midrule`, `\bottomrule`).
  - Wrap it in a `table` environment with `\centering`, `\caption{}` and `\label{tab:<label>}`.
  - Escape special characters (`_`, `%`, `&`). Column names like `DEP_DEL15` must compile.
  - Make number formatting configurable. Default: thousands separators for integers, 2 decimals for floats, 4 decimals for AUC and p-value columns. Allow a per-column format override.
  - Add an option to rename columns for display without changing the CSV.
  - Add an option for long tables (`longtable`) and one for landscape (`sidewaystable`).
- Saving the same table twice should overwrite the old files, so the scripts can be re-run.
- After saving, print one line, e.g. `Saved appendix table: <name> (rows x cols)`.
- Code style: snake_case, a module docstring (use the one below), type hints, short docstrings. Keep it short and readable, with no extra classes or abstraction.
- Dependencies: pandas and pathlib only.

Before writing, read the six scripts listed above and check which tables they produce (e.g. row counts per filtering step, missingness audits, airport lists, feature lists). Make sure the function handles all of them. Then show one usage example for each script.
'''
# 29.09.2026 18:25 CET
# Author: Anna Andruszkiewicz (code and adjustments), Claude Opus 5.5 (code)

from pathlib import Path
import pandas as pd
from config import processed_data_path

appendix_dir = Path(processed_data_path) / "appendix"


def save_appendix_table(
    table: pd.DataFrame,
    name: str,
    caption: str,
    longtable: bool = False,
    float_format: str = "%.2f",
) -> None:
    """Save a table as <name>.csv and <name>.tex in the appendix folder."""
    appendix_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(appendix_dir / f"{name}.csv", index=False)

    tex = table.copy()
    for col in tex.columns:
        if pd.api.types.is_integer_dtype(tex[col]):
            tex[col] = tex[col].map(lambda v: f"{v:,}" if pd.notna(v) else "")
    tex.to_latex(
        appendix_dir / f"{name}.tex",
        index=False,
        float_format=float_format,
        caption=caption,
        label=f"tab:{name}",
        escape=True,
        longtable=longtable,
    )
    print(f"saved appendix table {name} ({len(table)} rows) -> {appendix_dir}")