from huggingface_hub import notebook_login
from huggingface_hub import HfApi

api = HfApi()

api.upload_folder(
    repo_id="your repo name",  # Change to your own repo
    folder_path="models/Llama-3.2-3B-Instruct-Quan-2.8bits-1100-all",  # The quantized model folder to upload
)
