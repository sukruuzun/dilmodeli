# 📚 Nash-Swarm Optimization Tutorial

Complete guide to using, testing, and understanding Nash-Swarm quantization.

---

## 🎯 Table of Contents

1. [Quick Start](#quick-start)
2. [Understanding the Results](#understanding-the-results)
3. [Testing Different Models](#testing-different-models)
4. [Customizing Quantization](#customizing-quantization)
5. [Using Google Colab](#using-google-colab)
6. [Troubleshooting](#troubleshooting)

---

## 🚀 Quick Start

### Local Installation (5 minutes)

```bash
# 1. Clone repository
git clone https://github.com/sukruuzun/dilmodeli
cd dilmodeli

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run your first test (GPT-2 124M)
python tests/compare_simple_quantization.py
```

**Expected output:**
```
🔬 TESTING: GPT-2 (124M parameters)
⏱️  Estimated time: 5-8 minutes

[1/5] Loading model... ✓
[2/5] Loading dataset... ✓
[3/5] Baseline test... ✓ Loss: 5.414
[4/5] Uniform 4-bit... ✓ Loss: 6.689 (+23.56%)
[5/5] Nash-Swarm... ✓ Loss: 5.421 (+0.12%)

🎯 Nash-Swarm is 196× better!
```

---

## 📊 Understanding the Results

### What Each Metric Means

**1. Loss (Lower is Better)**
- Baseline: 5.414 → Model's natural performance
- Uniform 4-bit: 6.689 → Worse (degraded)
- Nash-Swarm: 5.421 → Almost same as baseline! ✅

**2. Δ Loss (Closer to 0% is Better)**
- +0.12% → Minimal accuracy loss
- +23.56% → Significant degradation

**3. Compression (Higher is Better)**
- 90.7% → Saved 90.7% of original size
- 87.5% → Standard 4-bit compression
- Difference: 3.2% more savings!

**4. Size**
- Baseline: 475 MB (FP32)
- Nash-Swarm: 44 MB (compressed)
- Ratio: 10.7× smaller

### Why Nash-Swarm Works

**Adaptive Bit Allocation:**
```
Critical weights (10%) → 8-bit precision
Important weights (20%) → 4-bit precision  
Less important (70%) → 2-bit precision
───────────────────────────────────────────
Average: ~3 bits per weight
```

**Result:** Preserves accuracy while achieving superior compression.

---

## 🔬 Testing Different Models

### GPT-2 Family

```bash
# GPT-2 (124M) - Fast, good for testing
python tests/compare_simple_quantization.py
# Runtime: ~5 minutes, RAM: 2-3 GB

# GPT-2 Medium (355M) - Moderate
# Edit model_name in script or:
sed -i 's/gpt2"/gpt2-medium"/' tests/compare_simple_quantization.py
python tests/compare_simple_quantization.py
# Runtime: ~10 minutes, RAM: 4-5 GB

# GPT-2 Large (774M) - Larger
# Runtime: ~20 minutes, RAM: 8-10 GB

# GPT-2 XL (1.5B) - Largest
# Runtime: ~40 minutes, RAM: 12-16 GB
```

### LLaMA Models (Requires more resources)

```bash
# TinyLlama 1.1B (Use Colab with GPU!)
python tests/test_llama_quantization.py
# Runtime: ~30 minutes, RAM: 16+ GB
```

**Recommendation:** Use Google Colab for LLaMA (see [Using Google Colab](#using-google-colab))

---

## ⚡ Speed Benchmark

Want to measure inference speed?

```bash
python tests/benchmark_speed.py
```

**Expected results:**
```
Model           Baseline    Nash-Swarm    Overhead
─────────────────────────────────────────────────
GPT-2 (124M)    18.24 ms    15.46 ms      0.85× (15% faster!)
GPT-2 Medium    28.61 ms    28.81 ms      1.01× (comparable)

Average: 0.93× overhead (7% faster!)
```

**Why faster?** Smaller quantized models → better cache efficiency!

---

## 🎨 Customizing Quantization

### Change Sample Size

```python
# Edit tests/compare_simple_quantization.py
# Line ~320:
text_samples = text_samples[:50]  # Change 50 to your desired size

# Smaller = faster but less accurate measurement
# Larger = slower but more reliable
```

### Adjust Bit Allocation

```python
# Edit quantize_nash_swarm_adaptive() function
# Default thresholds:
q_90 = torch.quantile(importance, 0.90)  # Top 10% → 8-bit
q_70 = torch.quantile(importance, 0.70)  # Top 30% → 4-bit
# Bottom 70% → 2-bit

# More aggressive (higher compression, lower accuracy):
q_95 = torch.quantile(importance, 0.95)  # Top 5% → 8-bit
q_80 = torch.quantile(importance, 0.80)  # Top 20% → 4-bit

# More conservative (lower compression, higher accuracy):
q_80 = torch.quantile(importance, 0.80)  # Top 20% → 8-bit
q_50 = torch.quantile(importance, 0.50)  # Top 50% → 4-bit
```

---

## 🌐 Using Google Colab

Perfect for testing without local resources!

### Setup (2 minutes)

1. Open [Google Colab](https://colab.research.google.com/)
2. Create new notebook
3. Follow [COLAB_SETUP.md](notebooks/COLAB_SETUP.md)

### Quick Commands

```python
# Cell 1: Setup
!git clone https://github.com/sukruuzun/dilmodeli.git
%cd dilmodeli
!pip install -q torch transformers datasets numpy

# Cell 2: Run test
!python tests/compare_simple_quantization.py

# Cell 3 (Optional): Download results
from google.colab import files
files.download('tests/RESULTS.txt')
```

### Enable GPU (for LLaMA)

1. Runtime → Change runtime type
2. Hardware accelerator → GPU (T4)
3. Save
4. Run cells

**Free tier limits:**
- 12-16 GB RAM
- ~12 hours runtime
- GPU access (limited)

---

## 🐛 Troubleshooting

### "Out of Memory" Error

**Symptoms:**
```
RuntimeError: [enforce fail at alloc_cpu.cpp:75] 
```

**Solutions:**
1. Use smaller model (gpt2 instead of gpt2-xl)
2. Reduce sample size (30 → 10)
3. Use Google Colab with GPU
4. Close other applications

### "Model not found" Error

**Symptoms:**
```
OSError: USERNAME/model-name is not a valid model identifier
```

**Solutions:**
1. Check internet connection
2. Verify model name spelling
3. Try: `huggingface-cli login` if using gated models

### Slow Performance

**If taking too long:**
- Enable GPU (Colab: Runtime → GPU)
- Reduce sample size
- Use smaller model for testing
- Check CPU usage (close background apps)

### "Module not found" Error

**Symptoms:**
```
ModuleNotFoundError: No module named 'transformers'
```

**Solution:**
```bash
pip install -r requirements.txt
# or manually:
pip install torch transformers datasets numpy
```

---

## 📊 Interpreting Results

### Good Results (Expected)

```
✅ Compression > 90%
✅ Δ Loss < 5% (small models)
✅ Δ Loss < 2% (medium/large models)
✅ Nash-Swarm better than Uniform
```

### Unexpected Results

**If Nash-Swarm is worse:**
- Normal for very large models (1B+)
- Quantization provides regularization
- Both methods work, uniform just happens to be better
- Still valid research finding!

**If compression is low (<85%):**
- Check quantization code
- Verify bit allocation thresholds
- May need to adjust parameters

---

## 💡 Best Practices

### 1. Start Small
- Test on GPT-2 (124M) first
- Verify everything works
- Then scale up

### 2. Use Sufficient Samples
- Minimum: 20 samples
- Recommended: 50 samples
- More = better statistics

### 3. Document Your Setup
- Note hardware specs
- Record runtime
- Save full output

### 4. Compare Fairly
- Same dataset for all methods
- Same sample size
- Same random seed (if applicable)

---

## 🎓 Advanced Usage

### Batch Testing Multiple Models

```bash
# Create script: test_all.sh
for model in gpt2 gpt2-medium gpt2-large; do
    echo "Testing $model..."
    sed -i "s/model_name = .*/model_name = \"$model\"/" tests/compare_simple_quantization.py
    python tests/compare_simple_quantization.py > results_$model.txt
done
```

### Custom Model Integration

```python
from transformers import AutoModelForCausalLM
from tests.compare_simple_quantization import quantize_nash_swarm_adaptive

# Load your model
model = AutoModelForCausalLM.from_pretrained("your/model")

# Apply Nash-Swarm quantization
quantized_model, info = quantize_nash_swarm_adaptive(model)

print(f"Compression: {info['compression_ratio']:.1f}%")
print(f"Avg bits: {info['avg_bits']:.2f}")
```

---

## 📝 Citation

If you use this work:

```bibtex
@article{uzun2024nashswarm,
  title={Nash-Swarm Optimization: A Game-Theoretic and Bio-Inspired Framework for Large Language Model Compression},
  author={Uzun, Şükrü},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2024}
}
```

---

## 🤝 Getting Help

**Issues:** [GitHub Issues](https://github.com/sukruuzun/dilmodeli/issues)
**Email:** sukru@yes.tools
**Website:** [yes.tools](https://yes.tools)

---

## 🎯 Next Steps

1. ✅ Run basic test
2. ✅ Try different models
3. ✅ Measure speed
4. 🔬 Apply to your own models
5. 📊 Share your results!

**Happy quantizing!** 🦅

