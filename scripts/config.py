import os

# Raw data: purely local, no cloud sync (too large for current Drive quota)
raw_data_path = "/Users/aniaandruszkiewicz/FS/Thesis/data/raw"
processed_data_path = "/Users/aniaandruszkiewicz/FS/Thesis/data/processed"
# Processed data: small enough to sync via Drive for backup + Colab access
data_root = "/Users/aniaandruszkiewicz/FS/Thesis_Data"
modeling_data_path = os.path.join(data_root, "modeling")