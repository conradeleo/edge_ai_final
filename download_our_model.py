from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="Fang77777/EAI_Final_model",
    local_dir="models/EAI_Final_model",
    local_dir_use_symlinks=False
)