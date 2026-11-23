from huggingface_hub import hf_hub_download, snapshot_download

snapshot_download(
    repo_id="Qwen/Qwen3.5-0.8B-Base",
    local_dir="base_model",
)

hf_hub_download(
    repo_id="yahma/alpaca-cleaned",
    repo_type="dataset",
    filename="alpaca_data_cleaned.json",
    local_dir="data",
)

print("Model and dataset downloaded.")
