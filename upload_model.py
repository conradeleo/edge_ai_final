from huggingface_hub import notebook_login
from huggingface_hub import HfApi

api = HfApi()

api.upload_folder(
    repo_id=f"Fang77777/Llama-3.2-3B-Instruct-GPTQ-all-exllama", # 修改成自己的repo
    folder_path="models/Llama-3.2-3B-Instruct-GPTQ-all", # 進行量化的模型
)