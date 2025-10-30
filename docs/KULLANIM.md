# DilModeli Kullanım Kılavuzu

## 📦 Kurulum

### Gereksinimler

```bash
Python >= 3.8
PyTorch >= 2.0.0
NumPy >= 1.24.0
```

### Kurulum Adımları

```bash
# 1. Repository'yi klonlayın
git clone https://github.com/your-username/dilmodeli.git
cd dilmodeli

# 2. Bağımlılıkları yükleyin
pip install -r requirements.txt

# 3. Test edin
python -c "import src; print('Kurulum başarılı!')"
```

## 🚀 Hızlı Başlangıç

### 1. Nash-Sürü MoE Kullanımı

```python
from src.moe.nash_suru_router import NashSuruMoE
from src.core.nash_suru_teorem import NashSuruParametreleri
import torch

# Nash-Sürü parametrelerini ayarla
nash_params = NashSuruParametreleri(
    lokal_grup_boyutu=7,        # Lokal grup boyutu
    lambda_dengeleme=0.1,        # İş yükü dengeleme ağırlığı
    lambda_cpu_onbellek=0.05     # CPU önbellek ödül ağırlığı
)

# MoE modeli oluştur
moe = NashSuruMoE(
    giris_boyutu=512,
    uzman_sayisi=8,
    uzman_gizli_boyutu=2048,
    lokal_grup_boyutu=7,
    top_k=2,
    nash_suru_params=nash_params
)

# Forward pass
x = torch.randn(4, 16, 512)  # [batch, seq_len, hidden_dim]
output, routing_info = moe(x, return_routing_info=True)

# Routing istatistiklerini incele
print(f"Sürü Uyumu: {routing_info['ortalama_suru_uyumu']:.4f}")
print(f"Nash Yakınsama: {routing_info['nash_yakinsama_orani']:.4f}")
```

### 2. Dinamik Kuantizasyon

```python
from src.quantization.dinamik_kuantizasyon import DinamikKuantizasyon, KuantizasyonKonfigurasyonu
import torch.nn as nn

# Model oluştur
model = nn.Sequential(
    nn.Linear(512, 1024),
    nn.ReLU(),
    nn.Linear(1024, 512)
)

# Kuantizasyon konfigürasyonu
config = KuantizasyonKonfigurasyonu(
    bit_genisligi=4,             # 4-bit kuantizasyon
    blok_boyutu=64,              # Blok boyutu
    komsu_boyutu=5,              # Komşu sayısı
    nash_alpha=0.2,              # Nash sürü etkisi
    budama_orani_hedefi=0.3      # %30 budama
)

# Kuantize et
quantizer = DinamikKuantizasyon(config)
kuantize_model, bilgi = quantizer.model_kuantize_et(model)

print(f"Bellek tasarrufu: {bilgi['ortalama_bellek_tasarrufu'] * 100:.1f}%")
```

### 3. CPU Optimizasyonu

```python
from src.optimization.cpu_optimizer import CPUOptimizer, CPUKonfigurasyonu

# CPU optimizer oluştur
config = CPUKonfigurasyonu(
    l1_onbellek_mb=0.5,
    l2_onbellek_mb=4.0,
    l3_onbellek_mb=16.0
)

optimizer = CPUOptimizer(config)

# Modeli CPU için optimize et
optimize_model = optimizer.model_optimizasyonu_uygula(model)

# Optimal batch boyutunu hesapla
optimal_batch = optimizer.batch_boyutu_optimize_et(
    model_parametre_sayisi=1_000_000,
    sekans_uzunlugu=128
)

print(f"Önerilen batch boyutu: {optimal_batch}")
```

### 4. Dengeleyici Kayıp Fonksiyonu

```python
from src.core.kayip_fonksiyonu import NashSuruKayipFonksiyonu, KayipAgirliklari

# Kayıp fonksiyonu oluştur
kayip_agirliklari = KayipAgirliklari(
    lambda_temel=1.0,
    lambda_dengeleme=0.1,
    lambda_cpu_onbellek=0.05,
    lambda_suru_uyumu=0.03,
    lambda_nash_dengesi=0.02
)

kayip_fn = NashSuruKayipFonksiyonu(agirliklar=kayip_agirliklari)

# Kayıp hesapla
ek_bilgiler = {
    'is_yuku_dagilimi': torch.tensor([10, 12, 9, 11, 8, 13, 10, 12]),
    'onbellek_durumu': torch.ones(8),
    'secilen_indeksler': torch.randint(0, 8, (64,))
}

kayiplar = kayip_fn(tahminler, hedefler, ek_bilgiler)
print(f"Toplam kayıp: {kayiplar['toplam_kayip'].item():.4f}")
```

## 🎯 Tam Örnek: LLM Eğitimi

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.moe.nash_suru_router import NashSuruMoE
from src.core.kayip_fonksiyonu import NashSuruKayipFonksiyonu
from src.optimization.cpu_optimizer import CPUOptimizer

# Model tanımla
class NashSuruLLM(nn.Module):
    def __init__(self, vocab_size=10000, hidden_dim=512, num_layers=6):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.moe_layers = nn.ModuleList([
            NashSuruMoE(hidden_dim, uzman_sayisi=8)
            for _ in range(num_layers)
        ])
        self.output = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, x):
        x = self.embedding(x)
        routing_infos = []
        
        for moe_layer in self.moe_layers:
            x, info = moe_layer(x, return_routing_info=True)
            routing_infos.append(info)
        
        logits = self.output(x)
        return logits, routing_infos

# Model ve optimizer
model = NashSuruLLM()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
kayip_fn = NashSuruKayipFonksiyonu()
cpu_opt = CPUOptimizer()

# Eğitim döngüsü
for epoch in range(num_epochs):
    for batch in train_loader:
        input_ids, labels = batch
        
        # Forward
        logits, routing_infos = model(input_ids)
        
        # Ek bilgileri hazırla
        ek_bilgiler = {
            'is_yuku_dagilimi': torch.tensor([
                info['uzman_kullanim_dagilimi'] 
                for info in routing_infos
            ]).mean(0),
            # ... diğer bilgiler
        }
        
        # Kayıp hesapla
        kayiplar = kayip_fn(
            logits.reshape(-1, logits.size(-1)),
            labels.reshape(-1),
            ek_bilgiler
        )
        
        # Backward
        optimizer.zero_grad()
        kayiplar['toplam_kayip'].backward()
        optimizer.step()
        
        # CPU önbellek güncelle
        cpu_opt.cpu_onbellek_durumunu_guncelle()
```

## 📊 Demo Scriptleri

Proje ile birlikte gelen demo scriptlerini çalıştırabilirsiniz:

```bash
# Tüm demo'ları çalıştır
bash run_demos.sh

# Veya tek tek:
python examples/demo_nash_suru_moe.py
python examples/demo_kuantizasyon.py
python examples/demo_cpu_optimizer.py
python examples/demo_tam_sistem.py
```

## 🔧 İleri Seviye Kullanım

### Özel Budama Stratejisi

```python
from src.quantization.budama_stratejisi import BudamaStratejisi
import numpy as np

class OzelBudamaStratejisi(BudamaStratejisi):
    def budama_skorlarini_hesapla(self, agirliklar):
        # Özel skorlama mantığınız
        return np.abs(agirliklar.cpu().numpy())
    
    def budama_maskesi_olustur(self, skorlar, budama_orani):
        # Özel maskeleme mantığınız
        esik = np.percentile(skorlar, budama_orani * 100)
        return skorlar > esik

# Kullanım
strateji = OzelBudamaStratejisi()
model, metrikler = model_buda(model, strateji, budama_orani=0.3)
```

### Görselleştirme

```python
from src.visualization.suru_visualizer import SuruVisualizasyonu
from src.visualization.metrik_visualizer import MetrikVisualizasyonu

# Sürü davranışı görselleştir
suru_viz = SuruVisualizasyonu()
suru_viz.suru_hareketini_ciz(
    pozisyon_gecmisi,
    kaydet_yol='suru_hareketi.png'
)

# Metrik görselleştir
metrik_viz = MetrikVisualizasyonu()
metrik_viz.kayip_egrileri_ciz(
    kayip_gecmisi,
    kaydet_yol='kayip_egrileri.png'
)
```

## 🐛 Sorun Giderme

### Bellek Hatası
```python
# Batch boyutunu küçültün
optimal_batch = cpu_opt.batch_boyutu_optimize_et(...)

# Veya daha agresif kuantizasyon
config.bit_genisligi = 2
config.budama_orani_hedefi = 0.5
```

### Yavaş Inference
```python
# CPU thread sayısını artırın
import torch
torch.set_num_threads(8)

# Önbellek boyutunu artırın
config.l3_onbellek_mb = 32.0
```

### Düşük Doğruluk
```python
# Kuantizasyonu gevşetin
config.bit_genisligi = 8
config.budama_orani_hedefi = 0.1

# Nash ağırlıklarını azaltın
nash_params.lambda_dengeleme = 0.01
```

## 📚 Ek Kaynaklar

- [Mimari Dokümantasyonu](MIMARISI.md)
- [API Referansı](API.md)
- [Araştırma Makalesi](PAPER.md)
- [FAQ](FAQ.md)

## 💬 Destek

Sorularınız için:
- GitHub Issues: [github.com/your-username/dilmodeli/issues](https://github.com)
- Email: your-email@example.com

