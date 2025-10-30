# 🚀 Google Colab Setup Guide

## 📓 How to Run Nash-Swarm Tests in Google Colab

### Method 1: Direct Python Script (Easiest!)

**Step 1:** Open [Google Colab](https://colab.research.google.com/)

**Step 2:** Create a new notebook

**Step 3:** Copy-paste this code into cells:

---

### 🦅 GPT-2 Demo (5-10 minutes)

```python
# Cell 1: Setup
!git clone https://github.com/sukruuzun/dilmodeli.git
%cd dilmodeli
!pip install -q torch transformers datasets numpy

print("✅ Setup complete!")
```

```python
# Cell 2: Run Test
!python tests/compare_simple_quantization.py
```

**Expected Output:**
```
Method          Loss    Δ Loss    Compression    Size
────────────────────────────────────────────────────
Baseline        5.414   -         0%             475 MB
Uniform 4-bit   6.689   +23.56%   87.5%          59 MB
Nash-Swarm      5.421   +0.12%    90.7%          44 MB  ✅

🎯 Nash-Swarm is 196× better!
```

---

### 🦙 LLaMA Demo (30-60 minutes, GPU recommended)

```python
# Cell 1: Setup (Enable GPU!)
# Runtime → Change runtime type → GPU (T4)
!nvidia-smi  # Check GPU

!git clone https://github.com/sukruuzun/dilmodeli.git
%cd dilmodeli
!pip install -q torch transformers datasets numpy

print("✅ GPU setup complete!")
```

```python
# Cell 2: Run LLaMA Test
!python tests/test_llama_quantization.py
```

**Note:** Free Colab has 12-16 GB RAM + GPU, enough for TinyLlama 1.1B!

---

### ⚡ Speed Benchmark (Optional)

```python
# Run speed test
!python tests/benchmark_speed.py
```

---

## 🎨 Customize Your Test

### Test Different Models

```python
# Edit model name before running
!sed -i 's/model_name = "gpt2"/model_name = "gpt2-medium"/' tests/compare_simple_quantization.py
!python tests/compare_simple_quantization.py
```

### Change Sample Size

```python
# Edit sample size (default: 30 for LLaMA, 50 for GPT-2)
!sed -i 's/text_samples\[:30\]/text_samples[:10]/' tests/test_llama_quantization.py
!python tests/test_llama_quantization.py
```

---

## 📊 Save Your Results

```python
# Save output to file
!python tests/compare_simple_quantization.py > my_results.txt

# Download results
from google.colab import files
files.download('my_results.txt')
```

---

## 🐛 Troubleshooting

### "Out of Memory" Error
- Enable GPU: Runtime → Change runtime type → GPU
- Reduce sample size (see "Change Sample Size" above)
- Use smaller model (gpt2 instead of gpt2-medium)

### "Module not found" Error
- Re-run Cell 1 (Setup)
- Make sure you're in `/content/dilmodeli` directory

### Slow Execution
- Enable GPU if testing LLaMA
- GPT-2 runs fine on CPU (~5-10 mins)

---

## 🎯 Quick Links

- 📄 **Paper:** [ArXiv:XXXX.XXXXX](https://arxiv.org/abs/XXXX.XXXXX)
- 💻 **GitHub:** [dilmodeli](https://github.com/sukruuzun/dilmodeli)
- 📧 **Contact:** sukru@yes.tools

---

## 📝 Creating a Persistent Notebook

Want to save your Colab notebook for later?

1. Follow steps above to create cells
2. File → Save a copy in Drive
3. Share the link!

**Template notebook coming soon!**

