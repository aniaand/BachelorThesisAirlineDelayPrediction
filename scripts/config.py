import os

# Raw data: purely local, no cloud sync (too large for current Drive quota)
RAW_DATA_PATH = "/Users/aniaandruszkiewicz/FS/Thesis/data/raw"

# Processed data: small enough to sync via Drive for backup + Colab access
DATA_ROOT = "/Users/aniaandruszkiewicz/FS/Thesis_Data"
PROCESSED_DATA_PATH = os.path.join(DATA_ROOT, "processed")