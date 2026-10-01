# Information Layers and Modeling Approaches in US Flight Delay Prediction

Bachelor's thesis, Computational Business Analytics, Frankfurt School of Finance & Management.

**Research question:** Does adding external signals (weather data and structured news events) meaningfully improve the prediction of US airline operational disruptions beyond historical delay patterns alone?

**Design:** The study is an ablation over four information layers, each run with Random Forest (on the full data and on a ~1M-row subsample) and TabPFN (zero-shot and fine-tuned, on the subsample only). The target is `DEP_DEL15` (departure delay ≥ 15 min). Models train on 2024 and are evaluated on all of 2025 as an out-of-sample holdout.

| Layer | Features |
|---|---|
| A | BTS flight data only |
| B | BTS + Meteostat weather |
| C | BTS + GDELT news events |
| D | BTS + Meteostat + GDELT |

---

## Repository structure

> TODO: check these names against the actual repo and remove anything that doesn't exist.

```
thesis/
├── README.md
├── .gitignore
├── environment.yml            # conda env "thesis" (Python 3.12.4)
├── requirements.txt           # pip freeze of the same env (used on Colab)
├── scripts/
│   ├── config.py              # central paths + GCP settings (only file to edit per machine)
│   ├── process_bts.py
│   ├── download_meteostat.py
│   ├── process_meteostat.py
│   ├── download_gdelt.py
│   ├── process_gdelt.py
│   ├── merge_layers.py
│   ├── subsample.py
│   ├── feature_engineering.py
│   └── modeling.py
├── notebooks/
│   ├── thesisrfmodeling.ipynb     # Random Forest, full + subsampled
│   ├── thesisfmmodeling.ipynb     # TabPFN zero-shot (Colab GPU)
│   ├── <finetune notebook>.ipynb  # TabPFN fine-tuning (Colab GPU)   TODO: name
│   └── results.ipynb              # combines results, DeLong tests, figures
└── data/                          # NOT committed, see .gitignore
    ├── raw/                       # bts/, meteostat/, gdelt/
    ├── processed/                 # bts/ (incl. retained_airports.csv), meteostat/, gdelt/
    └── final/
        ├── layer_*.parquet        # merged + engineered layer files
        └── modeling_results/
            ├── rf/                # RF subsampled summaries + *_probs.npz
            ├── rf_full/           # RF full-data summary
            └── fm/                # TabPFN zero-shot + fine-tuned summaries + *_y_proba.npy
```

All scripts read their paths from `scripts/config.py` (`raw_data_path`, `processed_data_path`, `final_data_path`, `gcp_project_id`, `bq_dataset_id`). Nothing else in the code hard-codes a path.

Each script follows the same pattern: **load → audit → filter → save**. Scripts use `# %%` cell markers, so they can be run as a whole or cell by cell in VS Code.

---

## Setup

```bash
conda env create -f environment.yml
conda activate thesis
```

1. Edit `scripts/config.py` so the data paths point to your local folders.
2. For GDELT (BigQuery), authenticate once with `gcloud auth application-default login`, then set `gcp_project_id` and `bq_dataset_id` in `config.py`.
3. For TabPFN on Colab, sync the final subsampled layer files to Google Drive and install from `requirements.txt` in the Colab runtime. Optionally set `HF_TOKEN` to avoid Hugging Face rate limits when the model weights download.

---

## Execution pipeline

Run the steps in this order. Each step reads only outputs from earlier steps.

> TODO: check that the order of steps 7 and 8 (subsample vs. feature engineering) matches your code.

| # | Step | Runs on | Input → Output |
|---|---|---|---|
| 1 | `process_bts.py` | local | Monthly BTS TranStats zips (`/PREZIP/`, 2024–2025) → cleaned flight table. Keeps airports with **at least 10 flights on every day** (114 of 358 airports, 86.8% of rows) and writes `processed/bts/retained_airports.csv` |
| 2 | `download_meteostat.py` | local | `retained_airports.csv` → hourly weather per airport (nearest station) |
| 3 | `process_meteostat.py` | local | Raw weather → cleaned hourly weather per airport |
| 4 | `download_gdelt.py` | local + BigQuery | `retained_airports.csv` → GDELT events within 50 km of each airport (CAMEO roots 14/17/18/20, `ActionGeo_Type IN (3,4)`), 2023-12-25 to 2025-12-31 (includes a 7-day lag buffer) |
| 5 | `process_gdelt.py` | local | Raw events → daily airport-level event features |
| 6 | `merge_layers.py` | local | BTS + weather + GDELT → the four layer files A–D |
| 7 | `subsample.py` | local | Layer files → stratified ~1M-row training subsample (same flights across all layers) |
| 8 | `feature_engineering.py` | local | Target encoding of airport (fit on 2024 only), sin/cos encoding of month / weekday / hour, lag and rolling features for weather and GDELT |
| 9 | `thesisrfmodeling.ipynb` | local | Random Forest on full data and on the subsample → `modeling_results/rf/`, `rf_full/` |
| 10 | `thesisfmmodeling.ipynb` | Colab GPU | TabPFN zero-shot on the subsample → `modeling_results/fm/` |
| 11 | `<finetune notebook>` | Colab GPU | Fine-tuned TabPFN on the subsample → `modeling_results/fm/fm_tuned_summary.csv` + tuned probabilities |
| 12 | `results.ipynb` | local | Combined comparison table, DeLong tests (model vs. model and layer vs. layer, Bonferroni-corrected), thesis figures |

**Class imbalance:** handled with class weighting during training plus a decision threshold tuned after training. No oversampling is used.

**Evaluation:** F1, precision, recall, AUC-ROC, AUC-PR and MCC on the 2025 holdout. DeLong's test is used for pairwise AUC-ROC significance.

---

## What is not committed (`.gitignore`)

Raw data acquisition is documented here rather than committed. Data and local clutter stay out of Git:

```gitignore
data/raw/
data/processed/
*.csv
*.parquet
.DS_Store
__pycache__/
.ipynb_checkpoints/
.env
```

| Ignored | Why | How to recreate it |
|---|---|---|
| `data/raw/` | BTS alone is ~16M rows; Meteostat and GDELT are re-downloadable | Steps 1–5 |
| `data/processed/` | Derived intermediates | Steps 1–5 |
| `*.parquet` | Merged, subsampled and engineered layer files | Steps 6–8 |
| `*.csv` | All tabular outputs, including `retained_airports.csv` and the model summary CSVs in `modeling_results/` | Steps 1–12 |
| `.env` | Local secrets and tokens (e.g. `HF_TOKEN`) | Create locally |
| `.DS_Store`, `__pycache__/`, `.ipynb_checkpoints/` | OS, Python and Jupyter clutter | — |

Because `*.csv` is ignored everywhere, the result tables are not in the repo. To reproduce the thesis tables, re-run steps 9–12. GCP credentials are stored outside the repo by `gcloud auth application-default login`, so they need no ignore rule.
