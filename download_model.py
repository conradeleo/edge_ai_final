from huggingface_hub import snapshot_download

# === 將模型下載到指定資料夾 === #
snapshot_download(
    repo_id="meta-llama/Llama-3.2-3B-Instruct",
    local_dir="models/Llama-3.2-3B-Instruct",
    local_dir_use_symlinks=False
)

# === 下載需要載入進行量化的資料集（終端機使用）=== #
# wget https://huggingface.co/datasets/Salesforce/wikitext/resolve/main/wikitext-2-raw-v1/train-00000-of-00001.parquet

# === 將原生模型進行量化 （終端機使用）=== #
# python ../exllamav2/convert.py -i ../models/Llama-3.2-3B-Instruct -o ../models/Llama-3.2-3B-Instruct-Quan -c ../train-00000-of-00001.parquet -cf ../models/Llama-3.2-3B-Instruct-Quan-all -b 4.0