from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="meta-llama/Llama-3.2-3B-Instruct",
    local_dir="models/Llama-3.2-3B-Instruct",
    local_dir_use_symlinks=False
)
