"""
CPU Optimizasyon Motoru

Bu modül, LLM'leri CPU'da verimli çalıştırmak için
önbellek yönetimi ve optimizasyon araçları sağlar.
"""

import numpy as np
import torch
import psutil
import os
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass
from collections import OrderedDict
import time


@dataclass
class CPUKonfigurasyonu:
    """CPU optimizasyon konfigürasyonu"""
    
    # Önbellek boyutları (MB cinsinden)
    l1_onbellek_mb: float = 0.5  # L1 cache (~512KB per core)
    l2_onbellek_mb: float = 4.0  # L2 cache (~4MB per core)
    l3_onbellek_mb: float = 16.0  # L3 cache (shared, ~16MB)
    
    # Optimizasyon parametreleri
    prefetch_enabled: bool = True  # Prefetch kullan
    cache_line_size: int = 64  # Cache line boyutu (bytes)
    numa_aware: bool = False  # NUMA-aware optimizasyonlar
    
    # İş parçacığı yönetimi
    max_threads: Optional[int] = None  # None = auto-detect
    thread_affinity: bool = False  # Thread affinity kullan


class OnbellekYoneticisi:
    """
    Önbellek Yöneticisi
    
    CPU önbelleğini verimli kullanmak için ağırlık ve aktivasyonları yönetir.
    """
    
    def __init__(self, config: Optional[CPUKonfigurasyonu] = None):
        """
        Args:
            config: CPU konfigürasyonu
        """
        self.config = config or CPUKonfigurasyonu()
        
        # Önbellek state
        self._l1_cache = OrderedDict()  # LRU cache
        self._l2_cache = OrderedDict()
        self._l3_cache = OrderedDict()
        
        # Cache hit/miss istatistikleri
        self._cache_stats = {
            'l1_hits': 0,
            'l1_misses': 0,
            'l2_hits': 0,
            'l2_misses': 0,
            'l3_hits': 0,
            'l3_misses': 0
        }
        
        # Önbellek boyut limitleri (bytes)
        self._l1_limit = int(self.config.l1_onbellek_mb * 1024 * 1024)
        self._l2_limit = int(self.config.l2_onbellek_mb * 1024 * 1024)
        self._l3_limit = int(self.config.l3_onbellek_mb * 1024 * 1024)
        
    def _tensor_boyutu_hesapla(self, tensor: torch.Tensor) -> int:
        """Tensor'un bellek boyutunu hesapla (bytes)"""
        return tensor.element_size() * tensor.numel()
    
    def _onbellek_boyutu_hesapla(self, cache: OrderedDict) -> int:
        """Önbellek boyutunu hesapla"""
        return sum(
            self._tensor_boyutu_hesapla(tensor)
            for tensor in cache.values()
        )
    
    def _lru_bosalt(self, cache: OrderedDict, limit: int):
        """LRU ile önbelleği boşalt"""
        while self._onbellek_boyutu_hesapla(cache) > limit:
            if not cache:
                break
            cache.popitem(last=False)  # En eski öğeyi çıkar
    
    def onbellege_ekle(
        self,
        key: str,
        tensor: torch.Tensor,
        seviye: str = "l3"
    ):
        """
        Tensor'u önbelleğe ekle
        
        Args:
            key: Tensor anahtarı (unique id)
            tensor: Tensor verisi
            seviye: Önbellek seviyesi ('l1', 'l2', 'l3')
        """
        tensor_boyutu = self._tensor_boyutu_hesapla(tensor)
        
        if seviye == "l1":
            if tensor_boyutu <= self._l1_limit:
                self._l1_cache[key] = tensor.clone()
                self._lru_bosalt(self._l1_cache, self._l1_limit)
        elif seviye == "l2":
            if tensor_boyutu <= self._l2_limit:
                self._l2_cache[key] = tensor.clone()
                self._lru_bosalt(self._l2_cache, self._l2_limit)
        else:  # l3
            if tensor_boyutu <= self._l3_limit:
                self._l3_cache[key] = tensor.clone()
                self._lru_bosalt(self._l3_cache, self._l3_limit)
    
    def onbellekten_al(self, key: str) -> Optional[torch.Tensor]:
        """
        Önbellekten tensor al (L1 -> L2 -> L3 sırasıyla)
        
        Args:
            key: Tensor anahtarı
            
        Returns:
            tensor: Bulunan tensor veya None
        """
        # L1'de ara
        if key in self._l1_cache:
            self._cache_stats['l1_hits'] += 1
            # LRU güncellemesi için move to end
            self._l1_cache.move_to_end(key)
            return self._l1_cache[key]
        
        self._cache_stats['l1_misses'] += 1
        
        # L2'de ara
        if key in self._l2_cache:
            self._cache_stats['l2_hits'] += 1
            self._l2_cache.move_to_end(key)
            # L1'e promote et (eğer yer varsa)
            tensor = self._l2_cache[key]
            if self._tensor_boyutu_hesapla(tensor) <= self._l1_limit:
                self.onbellege_ekle(key, tensor, seviye="l1")
            return tensor
        
        self._cache_stats['l2_misses'] += 1
        
        # L3'te ara
        if key in self._l3_cache:
            self._cache_stats['l3_hits'] += 1
            self._l3_cache.move_to_end(key)
            # L2'ye promote et
            tensor = self._l3_cache[key]
            if self._tensor_boyutu_hesapla(tensor) <= self._l2_limit:
                self.onbellege_ekle(key, tensor, seviye="l2")
            return tensor
        
        self._cache_stats['l3_misses'] += 1
        return None
    
    def get_cache_hit_rate(self) -> Dict[str, float]:
        """Cache hit rate'leri hesapla"""
        hit_rates = {}
        
        for seviye in ['l1', 'l2', 'l3']:
            hits = self._cache_stats[f'{seviye}_hits']
            misses = self._cache_stats[f'{seviye}_misses']
            total = hits + misses
            
            if total > 0:
                hit_rates[f'{seviye}_hit_rate'] = hits / total
            else:
                hit_rates[f'{seviye}_hit_rate'] = 0.0
        
        # Toplam hit rate
        total_hits = sum(self._cache_stats[k] for k in ['l1_hits', 'l2_hits', 'l3_hits'])
        total_misses = sum(self._cache_stats[k] for k in ['l1_misses', 'l2_misses', 'l3_misses'])
        total_accesses = total_hits + total_misses
        
        if total_accesses > 0:
            hit_rates['toplam_hit_rate'] = total_hits / total_accesses
        else:
            hit_rates['toplam_hit_rate'] = 0.0
        
        return hit_rates
    
    def istatistikleri_sifirla(self):
        """İstatistikleri sıfırla"""
        for key in self._cache_stats:
            self._cache_stats[key] = 0
    
    def onbellegi_temizle(self):
        """Tüm önbelleği temizle"""
        self._l1_cache.clear()
        self._l2_cache.clear()
        self._l3_cache.clear()
        self.istatistikleri_sifirla()


class CPUOptimizer:
    """
    CPU Optimizasyon Motoru
    
    LLM'leri CPU'da verimli çalıştırmak için çeşitli optimizasyonlar sağlar:
    1. Önbellek yönetimi
    2. SIMD/AVX kullanımı
    3. İş parçacığı yönetimi
    4. Bellek bant genişliği optimizasyonu
    """
    
    def __init__(self, config: Optional[CPUKonfigurasyonu] = None):
        """
        Args:
            config: CPU konfigürasyonu
        """
        self.config = config or CPUKonfigurasyonu()
        self.onbellek_yoneticisi = OnbellekYoneticisi(config)
        
        # CPU bilgilerini al
        self._cpu_bilgileri = self._cpu_bilgilerini_al()
        
        # Thread sayısını ayarla
        if self.config.max_threads is None:
            self.config.max_threads = os.cpu_count()
        
        torch.set_num_threads(self.config.max_threads)
        
    def _cpu_bilgilerini_al(self) -> Dict[str, Any]:
        """Sistem CPU bilgilerini al"""
        try:
            fiziksel = psutil.cpu_count(logical=False)
        except:
            fiziksel = os.cpu_count() or 1
        
        try:
            mantiksal = psutil.cpu_count(logical=True)
        except:
            mantiksal = os.cpu_count() or 1
        
        try:
            cpu_freq = psutil.cpu_freq().current if psutil.cpu_freq() and hasattr(psutil.cpu_freq(), 'current') else 0
        except:
            cpu_freq = 0
        
        cpu_info = {
            'cekirdek_sayisi': os.cpu_count() or 1,
            'fiziksel_cekirdek': fiziksel,
            'mantiksal_cekirdek': mantiksal,
            'toplam_bellek_gb': psutil.virtual_memory().total / (1024**3),
            'kullanilabilir_bellek_gb': psutil.virtual_memory().available / (1024**3),
            'cpu_frekansi_mhz': cpu_freq
        }
        
        return cpu_info
    
    def model_optimizasyonu_uygula(
        self,
        model: torch.nn.Module,
        optimize_inference: bool = True
    ) -> torch.nn.Module:
        """
        Model için CPU optimizasyonları uygula
        
        Args:
            model: PyTorch modeli
            optimize_inference: Inference optimizasyonları uygula
            
        Returns:
            optimize_model: Optimize edilmiş model
        """
        # Model'i CPU'ya taşı
        model = model.cpu()
        
        # Eval moduna al (inference için)
        if optimize_inference:
            model.eval()
        
        # PyTorch JIT (TorchScript) ile optimize et
        try:
            # Dummy input oluştur (model yapısına göre ayarlanmalı)
            # Bu örnek için temel bir tensor
            if optimize_inference:
                model = torch.jit.optimize_for_inference(
                    torch.jit.script(model)
                )
        except Exception as e:
            print(f"JIT optimizasyonu uygulanamadı: {e}")
        
        # MKLDNN backend kullan (Intel CPU'lar için)
        try:
            if hasattr(torch.backends, 'mkldnn') and torch.backends.mkldnn.is_available():
                # MKLDNN format'a dönüştür
                for module in model.modules():
                    if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
                        module.to_mkldnn()
        except Exception as e:
            print(f"MKLDNN optimizasyonu uygulanamadı: {e}")
        
        return model
    
    def batch_boyutu_optimize_et(
        self,
        model_parametre_sayisi: int,
        sekans_uzunlugu: int
    ) -> int:
        """
        CPU için optimal batch boyutunu hesapla
        
        Args:
            model_parametre_sayisi: Model parametre sayısı
            sekans_uzunlugu: Giriş sekans uzunluğu
            
        Returns:
            optimal_batch: Optimal batch boyutu
        """
        # Kullanılabilir bellek
        kullanilabilir_bellek = psutil.virtual_memory().available
        
        # Model boyutu tahmini (float32)
        model_boyutu = model_parametre_sayisi * 4
        
        # Aktivasyon boyutu tahmini
        aktivasyon_boyutu_per_sample = sekans_uzunlugu * 1024 * 4  # Yaklaşık tahmin
        
        # Güvenli marj ile optimal batch
        guvenli_marj = 0.7  # %70 bellek kullanımı
        maksimum_batch = int(
            (kullanilabilir_bellek * guvenli_marj - model_boyutu) / aktivasyon_boyutu_per_sample
        )
        
        # CPU için makul batch boyutları: 1, 2, 4, 8, 16
        makul_batch_boyutlari = [1, 2, 4, 8, 16, 32]
        optimal_batch = max(
            [b for b in makul_batch_boyutlari if b <= maksimum_batch],
            default=1
        )
        
        return optimal_batch
    
    def prefetch_agirliklari(
        self,
        model: torch.nn.Module,
        katman_isimleri: List[str]
    ):
        """
        Belirtilen katman ağırlıklarını önbelleğe yükle
        
        Args:
            model: PyTorch modeli
            katman_isimleri: Önbelleğe yüklenecek katman isimleri
        """
        for name, param in model.named_parameters():
            if any(katman in name for katman in katman_isimleri):
                # Ağırlıkları önbelleğe ekle
                key = f"weight_{name}"
                self.onbellek_yoneticisi.onbellege_ekle(key, param.data)
    
    def bellek_kullanimi_analiz_et(self) -> Dict[str, Any]:
        """
        Mevcut bellek kullanımını analiz et
        
        Returns:
            analiz: Bellek kullanım analizi
        """
        vm = psutil.virtual_memory()
        
        analiz = {
            'toplam_bellek_gb': vm.total / (1024**3),
            'kullanilan_bellek_gb': vm.used / (1024**3),
            'kullanilabilir_bellek_gb': vm.available / (1024**3),
            'bellek_kullanim_yuzdesi': vm.percent,
            'cache_hit_rates': self.onbellek_yoneticisi.get_cache_hit_rate(),
            'cpu_kullanim_yuzdesi': psutil.cpu_percent(interval=0.1)
        }
        
        return analiz
    
    def performans_profili_olustur(
        self,
        model: torch.nn.Module,
        ornek_girdi: torch.Tensor,
        iterasyon_sayisi: int = 10
    ) -> Dict[str, Any]:
        """
        Model için performans profili oluştur
        
        Args:
            model: PyTorch modeli
            ornek_girdi: Örnek girdi tensoru
            iterasyon_sayisi: Profil için iterasyon sayısı
            
        Returns:
            profil: Performans profili
        """
        model.eval()
        
        # Warm-up
        with torch.no_grad():
            for _ in range(3):
                _ = model(ornek_girdi)
        
        # Profiling
        sureler = []
        bellek_kullanimlari = []
        
        with torch.no_grad():
            for _ in range(iterasyon_sayisi):
                baslangic_zamani = time.time()
                baslangic_bellek = psutil.Process().memory_info().rss
                
                _ = model(ornek_girdi)
                
                bitis_zamani = time.time()
                bitis_bellek = psutil.Process().memory_info().rss
                
                sureler.append(bitis_zamani - baslangic_zamani)
                bellek_kullanimlari.append((bitis_bellek - baslangic_bellek) / (1024**2))
        
        profil = {
            'ortalama_inference_suresi_ms': np.mean(sureler) * 1000,
            'std_inference_suresi_ms': np.std(sureler) * 1000,
            'min_inference_suresi_ms': np.min(sureler) * 1000,
            'max_inference_suresi_ms': np.max(sureler) * 1000,
            'ortalama_bellek_artisi_mb': np.mean(bellek_kullanimlari),
            'throughput_samples_per_sec': 1.0 / np.mean(sureler),
            'cpu_bilgileri': self._cpu_bilgileri
        }
        
        return profil
    
    def get_cpu_bilgileri(self) -> Dict[str, Any]:
        """CPU bilgilerini döndür"""
        return self._cpu_bilgileri.copy()
    
    def get_onbellek_istatistikleri(self) -> Dict[str, Any]:
        """Önbellek istatistiklerini döndür"""
        return {
            'cache_hit_rates': self.onbellek_yoneticisi.get_cache_hit_rate(),
            'cache_stats': self.onbellek_yoneticisi._cache_stats.copy()
        }

