# Final Project

## Project Overview

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
   - Downloaded the **WikiText-2** training dataset as the calibration set for GPTQ quantization

2. **Quantization with GPTQ**  
   - Performed **2.8-bit quantization** via the **ExLlamaV2 GPTQ converter**  
   - Used **1100 samples** from WikiText-2, each with **2048+ tokens** for calibration  

3. **Inference with ExLlamaV2**  
   - Loaded the quantized model into the **ExLlamaV2** framework
   - Performed inference using the built-in **dynamic generator** provided by ExLlamaV2

4. **Results**  
   - **Throughput**: 108.9 tokens/s
   - **Perplexity**: 10.95 (≤ 11.5)
   
## 📦 Environment Setup

**Requirements**:
- Python >= 3.9
- CUDA >= 11.8
- PyTorch >= 2.1
- [Optional] GPU with at least 12GB VRAM recommended

**Install dependencies**:

```bash
pip install -r requirements.txt
