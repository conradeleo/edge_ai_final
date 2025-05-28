# LLM Acceleration Final Project

**Team**: 3
**Task**: Accelerate Llama3.2-3B-Instruct inference speed

## 🏆 Results

| Metric | Our Result |
|--------|------------|
| **Throughput** | **87.69 toks/s** |
| **Perplexity** | **8.58** | ≤ 11.5 (required) |

## 🎯 Method

We use **ExLlamaV2 + GPTQ 4-bit quantization** to accelerate inference:

1. Download Llama3.2-3B-Instruct model
2. Apply GPTQ 4-bit quantization with WikiText-2 calibration  
3. Load quantized model with ExLlamaV2 framework
4. Achieve 87.69 toks/s throughput with 8.58 perplexity

## 🚀 Quick Start

### Prerequisites
- NVIDIA GPU with CUDA
- Python 3.8+

### Installation
```bash
# Clone repository
git clone https://github.com/[username]/llm-acceleration-final.git
cd llm-acceleration-final

# Install dependencies
pip install -r requirements.txt

# Install ExLlamaV2
git clone https://github.com/turboderp/exllamav2
cd exllamav2 && pip install -e . && cd ..
```

### Run Evaluation
```bash
# Option 1: Use our pre-quantized model (recommended)
python src/inference.py --use_pretrained

# Option 2: Full pipeline (quantize yourself)
python src/download_model.py
bash scripts/quantize_model.sh
python src/inference.py
```

### Expected Output
```
Throughput: 87.69 toks/s
Perplexity (PPL): 8.58
Results saved to result.csv
```

## 📁 Files

- `src/download_model.py` - Download base model
- `src/inference.py` - Main evaluation script  
- `src/upload_model.py` - Upload quantized model
- `scripts/quantize_model.sh` - GPTQ quantization
- `requirements.txt` - Python dependencies

## 🔗 Resources

- **Original Model**: [Meta Llama3.2-3B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct)
- **ExLlamaV2**: [GitHub](https://github.com/turboderp/exllamav2)

## 📊 Technical Details

- **Quantization**: GPTQ 4-bit post-training quantization
- **Calibration**: WikiText-2 dataset
- **Framework**: ExLlamaV2 with optimized CUDA kernels
- **Hardware**: NVIDIA T4 (16GB VRAM)
- **Model Size**: ~1.6GB (75% reduction from 6.4GB)

## 👥 Team Members

- [成員1姓名]: [負責工作，例如：模型量化]
- [成員2姓名]: [負責工作，例如：效能優化]  
- [成員3姓名]: [負責工作，例如：文檔撰寫]

---

**Competition Results**: 87.69 toks/s throughput, 8.58 perplexity  
