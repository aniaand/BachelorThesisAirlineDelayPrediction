import os

# Raw data: purely local, no cloud sync
raw_data_path = "/Users/aniaandruszkiewicz/FS/Thesis/data/raw"
processed_data_path = "/Users/aniaandruszkiewicz/FS/Thesis/data/processed"
final_data_path = "/Users/aniaandruszkiewicz/FS/Thesis/data/final"

#GDELT: cloud storage and BigQuery
gcp_project_id = "thesis-507219"
bq_dataset_id = "thesis_data"
hf_token = "tabpfn_sk_fOIuKkH8Vz_XIP5KqKN4SuqPjVEDY0D4MoX20NdfkRE"