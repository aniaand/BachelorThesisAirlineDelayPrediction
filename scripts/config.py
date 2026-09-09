import os

# Raw data: purely local, no cloud sync
raw_data_path = "/Users/aniaandruszkiewicz/FS/Thesis/data/raw"
processed_data_path = "/Users/aniaandruszkiewicz/FS/Thesis/data/processed"
# Processed data: small enough to sync via Drive for backup
data_root = "/Users/aniaandruszkiewicz/FS/Thesis_Data"
modeling_data_path = os.path.join(data_root, "modeling")
# GCP project and BigQuery dataset for GDELT download
gcp_project_id = "thesis-507219"
bq_dataset_id = "thesis_data"