# Information Layers and Modeling Approaches in US Flight Delay Prediction

Bachelor's thesis, Computational Business Analytics, Frankfurt School of Finance & Management.

**Design:** The study is an ablation over four information layers, each run with Random Forest (on the full data and on a ~1M-row subsample) and TabPFN (zero-shot and fine-tuned, on the subsample only). The target is `dep_del15` (departure delay ≥ 15 min). Models train on 2024 and are evaluated on all of 2025 as an out-of-sample holdout.

| Layer | Features |
|---|---|
| A | BTS flight data only |
| B | BTS + Meteostat weather |
| C | BTS + GDELT news events |
| D | BTS + Meteostat + GDELT |

---

## Repository structure

```
thesis/
├── README.md
├── .gitignore
├── requirements.txt
├── documentation.md
├── scripts/
│   ├── config.py              # central paths + tokens (NOT committed, create locally, see Setup)
│   ├── appendix_utils.py      # shared helper for appendix tables (CSV + LaTeX)
│   ├── download_bts.py
│   ├── process_bts.py
│   ├── download_meteostat.py
│   ├── process_meteostat.py
│   ├── download_gdelt.py
│   ├── process_gdelt.py
│   ├── merge_layers.py
│   ├── subsample.py
│   ├── feature_engineering.py
│   └── summary_stats.py
├── notebooks/
│   ├── classical_ML_full.ipynb         # Random Forest, full
│   ├── classical_ML_subsample.ipynb    # Random Forest, subsampled
│   ├── FM_zero-shot.ipynb              # TabPFN zero-shot
│   ├── fine_tuning_epochs.ipynb        # TabPFN fine-tuning learning-rate search
│   ├── FM_fine_tuned.ipynb             # TabPFN fine-tuning
│   └── results.ipynb                   # combines results, DeLong tests, figures
└── data/                          # NOT committed, see .gitignore
    ├── raw/                       # bts/, meteostat/, gdelt/
    ├── processed/                 # bts/ (incl. retained_airports.csv), meteostat/, gdelt/
    └── final/
        ├── layer_*.parquet        # merged + engineered layer files
        └── modeling_results/
            ├── rf/                # RF subsampled summaries + *_probs.npz
            ├── rf_full/           # RF full-data summary
            ├── fm/                # TabPFN zero-shot summaries + *_y_proba.npy
            └── fm_tuned/          # TabPFN fine-tuned summaries + *_y_proba.npy + learning-rate search results
```

All scripts read their paths from `scripts/config.py` (`raw_data_path`, `processed_data_path`, `final_data_path`, `gcp_project_id`, `bq_dataset_id`). Nothing else in the code hard-codes a path.

Each script follows the same pattern: **load → audit → filter → save**. Scripts use `# %%` cell markers, so they can be run as a whole or cell by cell in VS Code.

---

## Setup

```bash
conda env create -f environment.yml
conda activate thesis
```

1. Create and edit `scripts/config.py` so the data paths point to your local folders, following this structure:

```python
# Raw data: purely local
raw_data_path = xxx
processed_data_path = xxx
final_data_path = xxx

# GDELT: cloud storage and BigQuery
gcp_project_id = xxx
bq_dataset_id = xxx
hf_token = xxx
```

2. For GDELT (BigQuery), authenticate once with `gcloud auth application-default login`, then set `gcp_project_id` and `bq_dataset_id` in `config.py`.

---

## Execution pipeline

Run the steps in this order. Each step reads only outputs from earlier steps. `appendix_utils.py` is not a step; the processing scripts import it to save appendix tables.

| # | Step | Runs on | Input → Output |
|---|---|---|---|
| 1 | `download_bts.py` | local | Monthly BTS TranStats zips (`/PREZIP/`, 2024–2025) → `data/raw/bts/` |
| 2 | `process_bts.py` | local | Raw BTS zips → cleaned flight table. Keeps airports with **at least 10 flights on every day** (114 of 358 airports, 86.8% of rows) and writes `processed/bts/retained_airports.csv` |
| 3 | `download_meteostat.py` | local | `retained_airports.csv` → hourly weather per airport (nearest station) |
| 4 | `process_meteostat.py` | local | Raw weather → cleaned hourly weather per airport |
| 5 | `download_gdelt.py` | local + BigQuery | `retained_airports.csv` → GDELT events within 50 km of each airport (CAMEO roots 14/17/18/20, `ActionGeo_Type IN (3,4)`), 2023-12-25 to 2025-12-31 (includes a 7-day lag buffer) |
| 6 | `process_gdelt.py` | local | Raw events → daily airport-level event features |
| 7 | `merge_layers.py` | local | BTS + weather + GDELT → the four layer files A–D |
| 8 | `subsample.py` | local | Layer files → stratified ~1M-row training subsample (same flights across all layers) |
| 9 | `feature_engineering.py` | local | Target encoding of airport (fit on 2024 only), sin/cos encoding of month / weekday / hour, lag and rolling features for weather and GDELT |
| 10 | `summary_stats.py` | local | Layer D → descriptive tables and figures for the data chapter |
| 11 | `classical_ML_subsample.ipynb` | local | Random Forest on the subsampled data → `modeling_results/rf/` |
| 12 | `classical_ML_full.ipynb` | local | Random Forest on the full data → `modeling_results/rf_full/` |
| 13 | `FM_zero-shot.ipynb` | Kaggle, GPU | TabPFN zero-shot on the subsample → `modeling_results/fm/` |
| 14 | `fine_tuning_epochs.ipynb` | Kaggle, GPU | Learning-rate search for fine-tuning (layer D, 100k sample) → `modeling_results/fm_tuned/` |
| 15 | `FM_fine_tuned.ipynb` | Kaggle, GPU | Fine-tuned TabPFN on the subsample, using the learning rate from step 14 → `modeling_results/fm_tuned/` |
| 16 | `results.ipynb` | local | Combined comparison table, DeLong tests (model vs. model and layer vs. layer), figures |

**Class imbalance:** handled with class weighting during training plus a decision threshold tuned after training. No oversampling is used.

**Evaluation:** F1, precision, recall, AUC-ROC, AUC-PR and MCC on the 2025 holdout. DeLong's test is used for pairwise AUC-ROC significance.

### GPU requirement and Kaggle

Steps 13–15 (TabPFN) need an NVIDIA GPU with CUDA. They do not run on a CPU or on Apple Silicon. They were run on Kaggle, not locally, and their setup cells must be adapted to the structure there:

- **Data:** upload the `*_subsampled_features.parquet` files from `data/final/` as a Kaggle dataset. They appear under `/kaggle/input/<dataset-name>/`.
- **Paths:** `scripts/config.py` is not in the repo, so set the input path to `/kaggle/input/<dataset-name>/` and the output path to `/kaggle/working/` in the notebook's first cell.
- **Token:** store the Hugging Face token as a Kaggle Secret named `HF_TOKEN` and set it as an environment variable in the first cell. The model weights (~876 MB) are downloaded from Hugging Face.
- **Settings:** turn on a GPU accelerator and internet access in the notebook settings.
- **Results:** download the outputs from `/kaggle/working/` and copy them into `data/final/modeling_results/fm/` or `fm_tuned/` before running `results.ipynb` locally.

---

## What is not committed (`.gitignore`)

Raw data acquisition is documented here rather than committed. Data, model artefacts, credentials and local clutter stay out of Git:

```gitignore
# --- Data (re-creatable from the pipeline) ---
data/raw/
data/processed/
data/final/
scripts/config.py
*.csv
*.parquet
*.zip
*.gz

# --- Model artefacts ---
*.npz
*.npy
*.pkl
*.joblib
*.ckpt
*.pt
*.pth
*.log

# --- Credentials / secrets ---
.env
*credentials*.json
*service-account*.json

# --- Python / Jupyter ---
__pycache__/
*.pyc
.ipynb_checkpoints/

# --- Editor / OS ---
.vscode/
.DS_Store
```

| Ignored | Why | How to recreate it |
|---|---|---|
| `data/raw/` | BTS alone is ~16M rows; Meteostat and GDELT are re-downloadable | Steps 1–6 |
| `data/processed/` | Derived intermediates | Steps 2–6 |
| `data/final/` | Layer files, model results and figures | Steps 7–16 |
| `scripts/config.py` | Machine-specific paths and the Hugging Face token | Create locally (see Setup) |
| `*.csv`, `*.parquet`, `*.zip`, `*.gz` | Tabular data and archives anywhere in the repo, incl. `retained_airports.csv` and result summaries | Steps 1–16 |
| `*.npz`, `*.npy` | Per-flight test probabilities (up to ~100 MB per file, above GitHub's limit) | Steps 11–15 |
| `*.pkl`, `*.joblib`, `*.ckpt`, `*.pt`, `*.pth`, `*.log` | Saved models, fine-tuning checkpoints and logs | Steps 11–15 |
| `__pycache__/`, `*.pyc`, `.ipynb_checkpoints/`, `.vscode/`, `.DS_Store` | Python, Jupyter, editor and OS clutter | — |

Because all data and results are ignored, the result tables are not in the repo. To reproduce the thesis tables, re-run steps 11–16. GCP credentials are stored outside the repo by `gcloud auth application-default login`, so they need no ignore rule.