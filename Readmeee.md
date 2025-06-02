# Final Project

This is the final project for the **Edge AI** course.
Our task is to **accelerate inference speed** for the `Llama3.2-3B-Instruct` model using various optimization techniques.

## Team
- **Team 3**
- **Team Members**
  - 110701018 張周芳
  - 110705009 陳重光
  - 111550029 蔡奕庠
  - 111705069 劉冠言

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

## Project Directory Structure
```bash
Lab4/
├── exllamav2/ # Cloned from GitHub
├── models/
│ ├── EAI_Final_model/ # Our Pre-quantized model (Download using `download_base_model.py`)
│ └── Llama-3.2-3B-Instruct/ # Base model (Download using `download_our_model.py`)
├── download_base_model.py # Script to download the base model
├── download_our_model.py # Script to download the our quantized model (EAL_final_model)
├── inference_dynamic.py # Script for inference using ExLlamaV2 dynamic generator
└── upload_model.py # Script to upload model to HuggingFace
```

## Hardware Used
- **NVIDIA T4 (16GB VRAM)** — Provided by the TA

## Reproducing the Experiment

### Environment Setup
We recommend creating a virtual environment (e.g., `.venv`) before installation.
```bash
# Install dependencies
pip install datasets
git clone https://github.com/turboderp/exllamav2.git
cd exllamav2
pip install -r requirements.txt
pip install .
cd .. # turn back to Lab4 folder
```
Then, login to Hugging Face:
```bash
huggingface-cli login
# Enter your Hugging Face access token when prompted
```
-------
### Quantization Steps
All commands below should be executed inside the Lab4 folder.

#### 1. Download the Base Model
```bash
python download_base_model.py
```

#### 2. Download the Calibration Dataset
```bash
wget https://huggingface.co/datasets/Salesforce/wikitext/resolve/main/wikitext-2-raw-v1/train-00000-of-00001.parquet
```

#### 3. Quantize the Model Using ExLlamaV2
```bash
python exllamav2/convert.py \
    -i models/Llama-3.2-3B-Instruct \                        # Path to the original (non-quantized) LLaMA model
    -o models/Llama-3.2-3B-Instruct-Quan-temp \              # Temporary output directory for intermediate quantization results
    -c train-00000-of-00001.parquet \                        # Calibration dataset used for quantization (in .parquet format)
    -cf models/Llama-3.2-3B-Instruct-Quan-2.8bits-1100-all \ # Final destination folder to store the fully quantized model
    -b 2.8 \                                                 # Bit precision for quantization (2.8 bits in this case)
    -r 1100                                                  # The number of sample used in quantization; affects accuracy vs. efficiency tradeoff
```

#### 4. Upload to Hugging Face
Make sure to update `repo_id` and `folder_path` in the `upload_model.py` script before running.

**Note:** The `folder_path` should match the directory you specified in the `-cf` argument above.
```bash
python upload_model.py
```
-------
### Inference Steps
All the following steps should be executed in the `Lab4` directory using Bash. Make sure the environment is set up properly. 

In this case, we will demonstrate how to run inference using our pre-quantized model `EAI_Final_model`.

#### 1. Download the Our Pre-quantized Model (EAI_final_model)
```bash
python download_our_model.py
```
After running the command, you should see a folder named `EAI_Final_model` appear under the `models/` directory.

#### 2. Running Inference
Since we are using an NVIDIA T4 GPU, which does not support ExLlamaV2's paged attention,
you must disable flash attention by setting the `EXLLAMA_NO_FLASH_ATTN=1` environment variable:
```bash
EXLLAMA_NO_FLASH_ATTN=1 python inference_dynamic.py
```
## Resources
- **Original Model**: [Meta Llama3.2-3B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct)
- **ExLlamaV2**: [GitHub](https://github.com/turboderp/exllamav2)
- **WikiText-2 Dataset**: https://huggingface.co/datasets/Salesforce/wikitext
- 
