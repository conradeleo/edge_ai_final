# Final Project

This is the final project for the **Edge AI** course.
Our task is to **accelerate inference speed** for the `Llama3.2-3B-Instruct` model using various optimization techniques.

## Team
- **Team 3**

## Objective

Optimize the performance of `Llama3.2-3B-Instruct` for fast and efficient inference without significantly compromising model quality (e.g., perplexity or generation quality).

## Technical Highlights

- Model: `Llama3.2-3B-Instruct`
- Optimization techniques may include:
  - GPTQ (Quantization)
  - ExllamaV2 framework

##  Methodology

1. **Model & Dataset Preparation**  
   - Downloaded the `Llama3.2-3B-Instruct` model  
   - Downloaded the **WikiText-2** training dataset as the calibration dataset for GPTQ quantization

2. **Quantization with GPTQ**  
   - Performed **2.8-bit quantization** via the **ExLlamaV2 GPTQ converter**  
   - Used **1100 samples** from WikiText-2, each with **2048+ tokens** for calibration  

3. **Inference with ExLlamaV2**  
   - Loaded the quantized model into the **ExLlamaV2** framework
   - Performed inference using the built-in **dynamic generator** provided by ExLlamaV2

4. **Results**  
   - **Throughput**: 108.9 tokens/s
   - **Perplexity**: 10.95 (≤ 11.5)
   
## 重新復現實驗過程

### Project Directory Structure
Lab4/
├── exllamav2/ # 會使用 git 進行下載
├── models/
│   ├── EAI_Final_model/ # 如果直接下載我們已經量化的模型
│   └── Llama-3.2-3B-Instruct/ # 這個是 base_model
├── download_base_model.py # 用來下載 base_model
├── download_model.py # 用來下載 EAI_Final_model
├── inference_dynamic.py # 用我們的模型進行推論
└── upload_model.py # 上傳我們量化的模型到 huggingface 使用

<pre lang="markdown"> ```plaintext Lab4/ ├── exllamav2/ # Cloned from GitHub ├── models/ │ ├── EAI_Final_model/ # Pre-quantized model (optional) │ └── Llama-3.2-3B-Instruct/ # Base model ├── download_base_model.py # Script to download the base model ├── download_model.py # Script to download the quantized model ├── inference_dynamic.py # Script for inference using ExLlamaV2 └── upload_model.py # Script to upload model to Hugging Face ``` </pre>

### 使用硬體設施
NVIDIA T4 (16GB VRAM) 就是助教提供的設備

### 環境設置
建議先建一個虛擬環境我們使用 .venv
再進行下面設定
```
# Install dependencies
pip install datasets
git clone https://github.com/turboderp/exllamav2.git
cd exllamav2
pip install -r requirements.txt
pip install .
cd .. # turn back to Lab4 folder
```
### 量化的步驟
接下來所有步驟都在 Lab4 folder 執行

### 下載 base_model
```
python download_base_model.py
```

### 下載 calibration dataset
```
wget https://huggingface.co/datasets/Salesforce/wikitext/resolve/main/wikitext-2-raw-v1/train-00000-of-00001.parquet
```

### 使用 exllamav2 提供的 convert.py 去進行量化跟轉為 EXL2 格式
```
python exllamav2/convert.py \
    -i models/Llama-3.2-3B-Instruct \                        # Path to the original (non-quantized) LLaMA model
    -o models/Llama-3.2-3B-Instruct-Quan-temp \              # Temporary output directory for intermediate quantization results
    -c train-00000-of-00001.parquet \                        # Calibration dataset used for quantization (in .parquet format)
    -cf models/Llama-3.2-3B-Instruct-Quan-2.8bits-1100-all \ # Final destination folder to store the fully quantized model
    -b 2.8 \                                                 # Bit precision for quantization (2.8 bits in this case)
    -r 1100                                                  # The number of sample used in quantization; affects accuracy vs. efficiency tradeoff
```
### 上傳至已經建立好的 huggingface 
裡面的 repo_id 以及 folder_path 要記得更改
```
python upload_model.py
```

**Install dependencies**:

```bash
pip install -r requirements.txt
