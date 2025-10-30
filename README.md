# 🦅 Nash-Swarm Optimization for LLMs

[![Paper](https://img.shields.io/badge/ArXiv-2024-red)](https://arxiv.org/abs/XXXX.XXXXX)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

## 📖 Overview

Nash-Swarm Optimization combines **Game Theory (Nash Equilibrium)** and **Swarm Intelligence (Starling Murmuration)** for efficient LLM compression. This framework achieves adaptive quantization with superior compression and accuracy preservation.

**Key Results:**
- ✅ **90.5-90.7% compression** across GPT-2 family (124M-1.5B)
- ✅ **196× better accuracy** for small models vs uniform quantization
- ✅ **0.93× inference overhead** (7% faster than baseline)
- ✅ **Model-specific behavior** validated across 4 scales

## 🎯 Temel Kavramlar

### Sığırcık Sürüleri (Sürü Zekası)
- **İlke**: Topluluk Uyumu ve Lokal Etkileşim
- **LLM Karşılığı**: MoE Yönlendirme ve Kuantizasyon
- Her birey sadece en yakın komşularına tepki verir, ancak tüm sürü uyumlu hareket eder

### Nash Dengesi (Oyun Teorisi)
- **İlke**: Stabilite ve Rasyonel Karar
- **LLM Karşılığı**: Eğitim/İnce Ayar Kararlılığı
- Tüm oyuncuların stratejilerini değiştirmek için teşvikinin olmadığı optimum nokta

## 🚀 Özellikler

### 1. Nash-Sürü MoE Yönlendirme
- Lokal uzman grupları ile verimli routing
- CPU önbellek optimizasyonu
- Dinamik iş yükü dengeleme

### 2. Dinamik Kuantizasyon ve Budama
- Komşu ağırlık bloklarının etkileşimi
- Nash dengesine dayalı budama kararları
- Gerçek zamanlı adaptasyon

### 3. Dengeleyici Kayıp Fonksiyonu
```
L_Nash-Sürü = L_ÇaprazEntropi + λ₁ · L_Dengeleme - λ₂ · R_CPU-Önbellek
```

## 📁 Project Structure

```
dilmodeli/
├── src/
│   ├── core/              # Core math and algorithm engine
│   ├── quantization/      # Adaptive quantization logic
│   ├── optimization/      # CPU optimization layer
│   └── visualization/     # Analysis and visualization
├── tests/                 # Comprehensive test suite
│   ├── compare_simple_quantization.py  # Main quantization tests
│   ├── benchmark_speed.py              # Speed benchmarks
│   └── comprehensive_speed_test.py     # Multi-model validation
├── paper/                 # Academic paper (LaTeX)
├── docs/                  # Documentation
└── requirements.txt       # Python dependencies
```

## 🛠️ Kurulum

```bash
pip install -r requirements.txt
```

## 💡 Quick Start

### Option 1: Local Installation (Recommended for GPT-2)

```bash
# Clone repository
git clone https://github.com/sukruuzun/dilmodeli
cd dilmodeli

# Install dependencies
pip install -r requirements.txt

# Run GPT-2 test (~5 minutes)
python tests/compare_simple_quantization.py

# Optional: Run speed benchmark
python tests/benchmark_speed.py
```

**System Requirements:**
- Python 3.8+
- 8GB RAM (GPT-2 models)
- 16GB+ RAM (larger models)
- CPU or GPU (MPS/CUDA)

### Option 2: Google Colab (No Installation!) ⭐

**Run in your browser with free GPU:**

📓 **Quick Demo (5 mins):** [See COLAB_SETUP.md](notebooks/COLAB_SETUP.md) for copy-paste instructions

```python
# Just 2 cells in Colab:
# 1. Setup
!git clone https://github.com/sukruuzun/dilmodeli.git && cd dilmodeli && pip install -q -r requirements.txt

# 2. Run
!python tests/compare_simple_quantization.py
```

🦙 **LLaMA Test:** Use Colab with GPU for larger models (see [Tutorial](TUTORIAL.md))

### Option 3: Quick Test (Single Command)

```bash
# Fast test on GPT-2 with 10 samples (~2 minutes)
python -c "from tests.compare_simple_quantization import main; main()"
```

## 📊 Performance Results

### Compression (Universal)
| Model | Baseline | Uniform 4-bit | Nash-Swarm | Advantage |
|-------|----------|---------------|------------|-----------|
| GPT-2 (124M) | 475 MB | 87.5% | **90.7%** | +3.2% |
| GPT-2 Medium (355M) | 1,354 MB | 87.5% | **90.7%** | +3.2% |
| GPT-2 Large (774M) | 2,953 MB | 87.5% | **90.5%** | +3.0% |
| GPT-2 XL (1.5B) | 5,942 MB | 87.5% | **90.6%** | +3.1% |

### Accuracy (Model-Specific)
| Model | Uniform 4-bit | Nash-Swarm | Advantage |
|-------|---------------|------------|-----------|
| GPT-2 (124M) | +23.56% loss ❌ | **+0.12%** ✅ | **196× better!** |
| GPT-2 Medium (355M) | +4.91% ⚠️ | **-0.15%** ✅ | Baseline+ |
| GPT-2 Large (774M) | +1.83% | **+1.71%** ✅ | Comparable |
| GPT-2 XL (1.5B) | -0.66% | +0.41% | Both good |

### Speed (Fast!)
| Model | Baseline | Nash-Swarm | Overhead |
|-------|----------|------------|----------|
| GPT-2 (124M) | 18.24 ms | **15.46 ms** | **0.85× (15% faster!)** |
| GPT-2 Medium (355M) | 28.61 ms | 28.81 ms | 1.01× (comparable) |
| **Average** | - | - | **0.93× (7% faster)** |

## 🔬 Research & Paper

**"Nash-Swarm Optimization: A Game-Theoretic and Bio-Inspired Framework for Large Language Model Compression"**

📄 **Paper:** [ArXiv:XXXX.XXXXX](https://arxiv.org/abs/XXXX.XXXXX) *(Coming soon)*  
📊 **Results:** See `FINAL_RESULTS_50_SAMPLES.md` for comprehensive analysis  
⚡ **Speed:** See `PAPER_SPEED_UPDATE.md` for benchmark details  
📚 **Tutorial:** See [TUTORIAL.md](TUTORIAL.md) for detailed usage guide

### Key Insights
- ✅ **Compression is scale-invariant** (90.5-90.7% across all models)
- ✅ **Accuracy is model-specific** (adaptive methods crucial for small models)
- ✅ **Speed without compromise** (faster inference + better accuracy)
- ✅ **Adaptive quantization is essential** ("one-size-fits-all" fails)

### For Researchers

**Reproducing Results:**
1. 🖥️ **Local:** Clone repo + run tests (requires 8-16GB RAM)
2. ☁️ **Colab:** Copy-paste [COLAB_SETUP.md](notebooks/COLAB_SETUP.md) (free GPU!)
3. 📊 **Custom:** See [TUTORIAL.md](TUTORIAL.md) for advanced usage

**Testing Your Model:**
```python
from tests.compare_simple_quantization import quantize_nash_swarm_adaptive

# Apply to any Hugging Face model
quantized_model, info = quantize_nash_swarm_adaptive(your_model)
print(f"Compression: {info['compression_ratio']:.1f}%")
```

**Expected Runtime:**
- GPT-2 (124M): ~5 minutes (CPU)
- GPT-2 Medium (355M): ~10 minutes (CPU)
- GPT-2 XL (1.5B): ~40 minutes (CPU/MPS)
- LLaMA (1B+): Use Colab with GPU

### Citation
```bibtex
@article{uzun2024nashswarm,
  title={Nash-Swarm Optimization: A Game-Theoretic and Bio-Inspired Framework for Large Language Model Compression},
  author={Uzun, Şükrü},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2024}
}
```

## 📝 License

MIT License

## 🤝 Contributing

Contributions welcome! Please feel free to submit a Pull Request.

## 📧 Contact

For questions and suggestions, please open an issue.

