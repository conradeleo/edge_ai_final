# Final Project

## Project Overview

This is the final project for the **Edge AI** course.
Our task is to **accelerate inference speed** for the `Llama3.2-3B-Instruct` model using various optimization techniques.

## 👥 Team

- Team 3

---

## 🎯 Objective

Optimize the performance of `Llama3.2-3B-Instruct` for fast and efficient inference without significantly compromising model quality (e.g., perplexity or generation quality).

---

## ⚙️ Technical Highlights

- ✅ Model: `Llama3.2-3B-Instruct`
- ✅ Optimization techniques may include:
  - GPTQ (Quantization)
  - Exllama or ExllamaV2 backend
  - TensorRT / ONNX acceleration
  - KV Cache / FlashAttention
  - Batch size tuning, token parallelism

> 🔧 Techniques used in our final implementation will be documented in detail below.

---

## 📦 Environment Setup

**Requirements**:
- Python >= 3.9
- CUDA >= 11.8
- PyTorch >= 2.1
- [Optional] GPU with at least 12GB VRAM recommended

**Install dependencies**:

```bash
pip install -r requirements.txt
