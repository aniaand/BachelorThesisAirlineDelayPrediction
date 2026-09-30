"""
appendix_utils.py

Shared helper for writing thesis appendix tables to <processed_data_path>/appendix/
as CSV (for checking) and booktabs LaTeX (for \\input{} in the thesis).
Imported by process_bts.py, process_meteostat.py, process_gdelt.py,
merge_layers.py, subsample.py and feature_engineering.py.
"""
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