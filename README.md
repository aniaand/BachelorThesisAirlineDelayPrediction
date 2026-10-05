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
│   ├── fine_tuning_lr_epochs.ipynb        # TabPFN fine-tuning learning-rate search
│   ├── FM_fine_tuned_layer_a.ipynb     # TabPFN fine-tuning, layer A  ┐ identical except for the
│   ├── FM_fine_tuned_layer_b.ipynb     # TabPFN fine-tuning, layer B  │ layer they run; started
│   ├── FM_fine_tuned_layer_c.ipynb     # TabPFN fine-tuning, layer C  │ in parallel on Kaggle
│   ├── FM_fine_tuned_layer_d.ipynb     # TabPFN fine-tuning, layer D  ┘
│   ├── fm_finetuned_results.ipynb      # combines the four fine-tuning runs into one set of tables
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
            └── fm_tuned/          # TabPFN fine-tuned, see below
```

`fm_tuned/` holds the output of steps 14–16:

```
fm_tuned/
├── finetune_lr_search_layer_d.csv            # step 14: learning-rate search
├── fm_subsampled_summary_layer_{a,b,c,d}.csv # step 15: one summary per layer
├── layer_*_subsampled_finetuned_y_proba.npy  # step 15: test probabilities per layer
├── layer_*_subsampled/                       # step 15: fine-tuning checkpoints (.pth) per layer
├── fm_finetuned_summary_all_layers.csv       # step 16: the four summaries combined
├── fm_finetuned_checkpoints_all_layers.csv   # step 16: checkpoint inventory
└── fm_finetuned_best_checkpoints.csv         # step 16: best checkpoint per layer
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

Run the steps in this order. Each step reads only outputs from earlier steps, except that the four notebooks in step 15 are independent of each other and run in parallel. `appendix_utils.py` is not a step; the processing scripts import it to save appendix tables.

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
| 14 | `fine_tuning_lr_epochs.ipynb` | Kaggle, GPU | Learning-rate search for fine-tuning (layer D, 100k sample; learning rates 3e-5 to 3e-7 in half-decade steps, up to 30 epochs, early-stopping patience 8) → `modeling_results/fm_tuned/finetune_lr_search_layer_d.csv` |
| 15 | `FM_fine_tuned_layer_{a,b,c,d}.ipynb` | Kaggle, GPU, **4 notebooks in parallel** | Fine-tuned TabPFN on the subsample, one notebook per layer, all with the learning rate and early-stopping settings from step 14 → per layer: summary CSV, test probabilities and checkpoint folder in `modeling_results/fm_tuned/` |
| 16 | `fm_finetuned_results.ipynb` | local (CPU) | The four per-layer summaries and checkpoint folders → one combined summary table, a checkpoint inventory, and a consistency check between the two |
| 17 | `results.ipynb` | local | Combined comparison table (including the combined fine-tuned summary from step 16), DeLong tests (model vs. model and layer vs. layer), figures |

**Class imbalance:** handled with class weighting during training plus a decision threshold tuned after training. No oversampling is used. TabPFN has no class-weighting option, so for TabPFN only the tuned threshold applies.

**Evaluation:** F1, precision, recall, AUC-ROC, AUC-PR and MCC on the 2025 holdout. DeLong's test is used for pairwise AUC-ROC significance.

### GPU requirement and Kaggle

Steps 13–15 (TabPFN) need an NVIDIA GPU with CUDA. They do not run on a CPU or on Apple Silicon. They were run on Kaggle, not locally, and their setup cells must be adapted to the structure there:

- **Data:** upload the `*_subsampled_features.parquet` files from `data/final/` as a Kaggle dataset. They appear under `/kaggle/input/<dataset-name>/`.
- **Paths:** `scripts/config.py` is not in the repo, so set the input path to `/kaggle/input/<dataset-name>/` and the output path to `/kaggle/working/` in the notebook's first cell.
- **Tokens:** store the tokens as Kaggle Secrets (`HF_TOKEN` for the Hugging Face model download, `TABPFN_TOKEN` for TabPFN) and set them as environment variables in the first cell. Never paste a token into a cell, because it ends up in the saved notebook. The model weights (~876 MB) are downloaded from Hugging Face.
- **Settings:** turn on a GPU accelerator and internet access in the notebook settings.
- **Results:** download the outputs from `/kaggle/working/` and copy them into `data/final/modeling_results/fm/` or `fm_tuned/` before running the local notebooks.

#### Running the four fine-tuning notebooks in parallel (step 15)

Fine-tuning one layer takes several hours, so the four layers are run as four separate notebooks started at the same time instead of one loop over all layers. Each notebook is a copy of the same code with only the layer in `layers_to_run` changed.

- **Start all four with "Save & Run All"** (a committed run). It keeps running when the browser is closed or the computer sleeps, and its output is saved with the version.
- **Parallel runs save wall-clock time, not GPU quota.** Each session is counted against the weekly GPU quota separately. Estimated at roughly 5–9 hours per layer (about 20 minutes per epoch, plus about 2 hours of test-set prediction), all four layers use about 20–36 GPU-hours, which can exceed the weekly quota of about 30 hours. Check the remaining quota before starting.
- **Kaggle limits concurrent GPU sessions.** If fewer than four can start at once, start the rest as sessions finish. Quota already reserved by running sessions may also block a new session from starting.
- **Give every notebook its own output.** The checkpoint folder (`output_dir`) is named after the layer. `fit()` resumes from `checkpoint_*_best.pth` if one already exists in the folder it is given, so never point a run at a folder left over from a different learning rate. Output from earlier runs with a different configuration would otherwise be silently continued.
- **Keep the settings identical across the four notebooks** (learning rate, epoch cap, early-stopping patience, batch sizes). Step 16 warns if the saved settings differ between layers.
- **After all four finish,** download each notebook's output, copy the files into `data/final/modeling_results/fm_tuned/` (the layer-named folders and file names do not collide), and run `fm_finetuned_results.ipynb`.

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
| `data/final/` | Layer files, model results and figures | Steps 7–17 |
| `scripts/config.py` | Machine-specific paths and the Hugging Face token | Create locally (see Setup) |
| `*.csv`, `*.parquet`, `*.zip`, `*.gz` | Tabular data and archives anywhere in the repo, incl. `retained_airports.csv` and result summaries | Steps 1–17 |
| `*.npz`, `*.npy` | Per-flight test probabilities (up to ~100 MB per file, above GitHub's limit) | Steps 11–15 |
| `*.pkl`, `*.joblib`, `*.ckpt`, `*.pt`, `*.pth`, `*.log` | Saved models, fine-tuning checkpoints and logs | Steps 11–15 |
| `__pycache__/`, `*.pyc`, `.ipynb_checkpoints/`, `.vscode/`, `.DS_Store` | Python, Jupyter, editor and OS clutter | — |

Because all data and results are ignored, the result tables are not in the repo. To reproduce the thesis tables, re-run steps 11–17. GCP credentials are stored outside the repo by `gcloud auth application-default login`, so they need no ignore rule.