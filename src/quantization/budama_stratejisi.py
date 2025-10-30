"""
Budama Stratejileri

Nash-Sürü teoremi ile dinamik budama stratejileri
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Optional, Dict, Any, List, Callable, Tuple
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class BudamaMetrikleri:
    """Budama metrikleri"""
    budama_orani: float
    dogruluk_kaybi: float
    bellek_tasarrufu: float
    hiz_artisi: float
    budanan_parametre_sayisi: int


class BudamaStratejisi(ABC):
    """
    Budama stratejisi arayüzü
    """
    
    @abstractmethod
    def budama_skorlarini_hesapla(
        self,
        agirliklar: torch.Tensor
    ) -> np.ndarray:
        """
        Ağırlıklar için budama skorlarını hesapla
        
        Args:
            agirliklar: Ağırlık tensoru
            
        Returns:
            budama_skorlari: Her ağırlık için budama skoru (düşük = buda)
        """
        pass
    
    @abstractmethod
    def budama_maskesi_olustur(
        self,
        budama_skorlari: np.ndarray,
        budama_orani: float
    ) -> np.ndarray:
        """
        Budama maskesi oluştur
        
        Args:
            budama_skorlari: Budama skorları
            budama_orani: Hedef budama oranı [0, 1]
            
        Returns:
            budama_maskesi: True = koru, False = buda
        """
        pass


class MagnitudeBudama(BudamaStratejisi):
    """
    Magnitude-based budama: Küçük ağırlıkları buda
    """
    
    def budama_skorlarini_hesapla(
        self,
        agirliklar: torch.Tensor
    ) -> np.ndarray:
        """Magnitude skorları (L1 norm)"""
        return torch.abs(agirliklar).cpu().numpy()
    
    def budama_maskesi_olustur(
        self,
        budama_skorlari: np.ndarray,
        budama_orani: float
    ) -> np.ndarray:
        """En küçük magnitude'leri buda"""
        esik_idx = int(len(budama_skorlari.flatten()) * budama_orani)
        skorlar_flat = budama_skorlari.flatten()
        esik_deger = np.sort(skorlar_flat)[esik_idx]
        
        return budama_skorlari > esik_deger


class YapısalBudama(BudamaStratejisi):
    """
    Yapısal budama: Tüm nöron/kanal/filter'ları buda
    """
    
    def __init__(self, yapisal_birim: str = "filter"):
        """
        Args:
            yapisal_birim: 'filter', 'channel', 'neuron'
        """
        self.yapisal_birim = yapisal_birim
    
    def budama_skorlarini_hesapla(
        self,
        agirliklar: torch.Tensor
    ) -> np.ndarray:
        """Yapısal birim skorları (L2 norm)"""
        if self.yapisal_birim == "filter" and agirliklar.ndim == 4:
            # Conv filter: [out_channels, in_channels, H, W]
            # Her filter için norm
            skorlar = torch.norm(agirliklar.reshape(agirliklar.shape[0], -1), p=2, dim=1)
        elif self.yapisal_birim == "neuron" and agirliklar.ndim == 2:
            # Linear neuron: [out_features, in_features]
            # Her nöron için norm
            skorlar = torch.norm(agirliklar, p=2, dim=1)
        else:
            # Fallback: magnitude budama
            skorlar = torch.abs(agirliklar)
        
        return skorlar.cpu().numpy()
    
    def budama_maskesi_olustur(
        self,
        budama_skorlari: np.ndarray,
        budama_orani: float
    ) -> np.ndarray:
        """En düşük skorlu yapısal birimleri buda"""
        N_birim = len(budama_skorlari)
        budanacak_sayi = int(N_birim * budama_orani)
        
        esik_idx = budanacak_sayi
        esik_deger = np.sort(budama_skorlari)[esik_idx] if esik_idx < N_birim else np.max(budama_skorlari)
        
        return budama_skorlari > esik_deger


class NashSuruBudama(BudamaStratejisi):
    """
    Nash-Sürü Budama Stratejisi
    
    Ağırlıkları oyuncu olarak görür ve:
    1. Her ağırlık kendi önemine bakar (Rasyonel Çıkar)
    2. Komşu ağırlıkların önemine bakar (Sürü Etkisi)
    3. Nash dengesi: Doğruluk vs bellek trade-off
    """
    
    def __init__(
        self,
        komsu_boyutu: int = 5,
        suru_agirligi: float = 0.3,
        blok_boyutu: int = 64
    ):
        """
        Args:
            komsu_boyutu: Komşu ağırlık sayısı
            suru_agirligi: Sürü etkisi ağırlığı [0, 1]
            blok_boyutu: Blok boyutu (lokal etkileşim için)
        """
        self.komsu_boyutu = komsu_boyutu
        self.suru_agirligi = suru_agirligi
        self.blok_boyutu = blok_boyutu
    
    def budama_skorlarini_hesapla(
        self,
        agirliklar: torch.Tensor
    ) -> np.ndarray:
        """
        Nash-Sürü skorları: Kendi önemi + komşu etkisi
        """
        agirliklar_flat = agirliklar.flatten()
        N = len(agirliklar_flat)
        
        # Temel önem: Magnitude
        temel_skorlar = torch.abs(agirliklar_flat).cpu().numpy()
        
        # Bloklara böl
        N_blok = (N + self.blok_boyutu - 1) // self.blok_boyutu
        blok_skorlari = np.zeros(N)
        
        for i in range(N_blok):
            baslangic = i * self.blok_boyutu
            bitis = min((i + 1) * self.blok_boyutu, N)
            blok_indeksleri = list(range(baslangic, bitis))
            
            # Bu bloğun ortalaması
            blok_ortalama = np.mean(temel_skorlar[blok_indeksleri])
            
            # Komşu blokları bul
            komsu_blok_indeksleri = []
            yaricap = self.komsu_boyutu // 2
            for j in range(max(0, i - yaricap), min(N_blok, i + yaricap + 1)):
                if j != i:
                    b_start = j * self.blok_boyutu
                    b_end = min((j + 1) * self.blok_boyutu, N)
                    komsu_blok_indeksleri.extend(range(b_start, b_end))
            
            # Komşu etkisi
            if komsu_blok_indeksleri:
                komsu_ortalama = np.mean(temel_skorlar[komsu_blok_indeksleri])
                suru_etkisi = self.suru_agirligi * komsu_ortalama
            else:
                suru_etkisi = 0.0
            
            # Blok içindeki tüm ağırlıklara sürü etkisi ekle
            for idx in blok_indeksleri:
                blok_skorlari[idx] = temel_skorlar[idx] + suru_etkisi
        
        return blok_skorlari.reshape(agirliklar.shape)
    
    def budama_maskesi_olustur(
        self,
        budama_skorlari: np.ndarray,
        budama_orani: float
    ) -> np.ndarray:
        """
        Nash dengesi ile budama maskesi
        
        En düşük skorlu ağırlıkları buda, ancak komşu etkilerini dikkate al
        """
        skorlar_flat = budama_skorlari.flatten()
        N = len(skorlar_flat)
        budanacak_sayi = int(N * budama_orani)
        
        # En düşük skorları buda
        esik_idx = budanacak_sayi
        if esik_idx >= N:
            return np.zeros_like(budama_skorlari, dtype=bool)  # Hepsini buda
        
        esik_deger = np.sort(skorlar_flat)[esik_idx]
        budama_maskesi = budama_skorlari > esik_deger
        
        return budama_maskesi


class AdaptifBudama(BudamaStratejisi):
    """
    Adaptif Budama: Katman duyarlı budama
    
    Her katman için farklı budama oranı uygular
    """
    
    def __init__(
        self,
        temel_strateji: BudamaStratejisi,
        katman_hassasiyetleri: Optional[Dict[str, float]] = None
    ):
        """
        Args:
            temel_strateji: Temel budama stratejisi
            katman_hassasiyetleri: Her katman için hassasiyet çarpanı
        """
        self.temel_strateji = temel_strateji
        self.katman_hassasiyetleri = katman_hassasiyetleri or {}
    
    def budama_skorlarini_hesapla(
        self,
        agirliklar: torch.Tensor
    ) -> np.ndarray:
        """Temel strateji skorlarını kullan"""
        return self.temel_strateji.budama_skorlarini_hesapla(agirliklar)
    
    def budama_maskesi_olustur(
        self,
        budama_skorlari: np.ndarray,
        budama_orani: float
    ) -> np.ndarray:
        """Adaptif budama oranı ile maske oluştur"""
        return self.temel_strateji.budama_maskesi_olustur(
            budama_skorlari, budama_orani
        )
    
    def katmana_ozel_budama_orani(
        self,
        katman_adi: str,
        temel_budama_orani: float
    ) -> float:
        """
        Katmana özel budama oranı hesapla
        
        Args:
            katman_adi: Katman adı
            temel_budama_orani: Temel budama oranı
            
        Returns:
            katman_budama_orani: Bu katman için budama oranı
        """
        hassasiyet = self.katman_hassasiyetleri.get(katman_adi, 1.0)
        return temel_budama_orani * hassasiyet


def model_buda(
    model: nn.Module,
    strateji: BudamaStratejisi,
    budama_orani: float,
    hedef_katmanlar: Optional[List[str]] = None
) -> Tuple[nn.Module, Dict[str, BudamaMetrikleri]]:
    """
    Tüm modeli belirtilen stratejiyle buda
    
    Args:
        model: PyTorch modeli
        strateji: Budama stratejisi
        budama_orani: Hedef budama oranı [0, 1]
        hedef_katmanlar: Budanacak katman isimleri (None ise tümü)
        
    Returns:
        budanmis_model: Budanmış model
        metrikler: Her katman için budama metrikleri
    """
    metrikler = {}
    
    for name, param in model.named_parameters():
        # Hedef katman kontrolü
        if hedef_katmanlar and not any(target in name for target in hedef_katmanlar):
            continue
        
        # Sadece weight parametrelerini buda
        if 'weight' not in name:
            continue
        
        orijinal_agirliklar = param.data.clone()
        
        # Budama skorlarını hesapla
        budama_skorlari = strateji.budama_skorlarini_hesapla(param.data)
        
        # Budama maskesi oluştur
        budama_maskesi = strateji.budama_maskesi_olustur(budama_skorlari, budama_orani)
        
        # Budamayı uygula
        param.data = param.data * torch.tensor(
            budama_maskesi, dtype=param.dtype, device=param.device
        )
        
        # Metrikler
        toplam_parametre = param.numel()
        budanan_parametre = int(np.sum(~budama_maskesi))
        gercek_budama_orani = budanan_parametre / toplam_parametre
        bellek_tasarrufu = gercek_budama_orani  # Sıfır ağırlıklar için
        
        metrikler[name] = BudamaMetrikleri(
            budama_orani=gercek_budama_orani,
            dogruluk_kaybi=0.0,  # Bu ayrıca ölçülmeli
            bellek_tasarrufu=bellek_tasarrufu,
            hiz_artisi=0.0,  # Bu ayrıca ölçülmeli
            budanan_parametre_sayisi=budanan_parametre
        )
    
    return model, metrikler


def iteratif_budama(
    model: nn.Module,
    strateji: BudamaStratejisi,
    hedef_budama_orani: float,
    adim_sayisi: int = 5,
    dogrulama_fonksiyonu: Optional[Callable] = None,
    maksimum_dogruluk_kaybi: float = 0.02
) -> Tuple[nn.Module, List[Dict[str, BudamaMetrikleri]]]:
    """
    İteratif budama: Yavaş yavaş budama oranını artır
    
    Args:
        model: PyTorch modeli
        strateji: Budama stratejisi
        hedef_budama_orani: Hedef budama oranı
        adim_sayisi: Budama adım sayısı
        dogrulama_fonksiyonu: Doğruluk ölçme fonksiyonu
        maksimum_dogruluk_kaybi: Maksimum kabul edilebilir doğruluk kaybı
        
    Returns:
        budanmis_model: Budanmış model
        adim_metrikleri: Her adım için metrikler
    """
    adim_metrikleri = []
    baslangic_dogrulugu = None
    
    if dogrulama_fonksiyonu:
        baslangic_dogrulugu = dogrulama_fonksiyonu(model)
    
    for adim in range(adim_sayisi):
        # Budama oranını kademeli artır
        mevcut_budama_orani = hedef_budama_orani * (adim + 1) / adim_sayisi
        
        # Modeli buda
        model, metrikler = model_buda(model, strateji, mevcut_budama_orani)
        
        # Doğruluk kontrolü
        if dogrulama_fonksiyonu and baslangic_dogrulugu:
            yeni_dogruluk = dogrulama_fonksiyonu(model)
            dogruluk_kaybi = baslangic_dogrulugu - yeni_dogruluk
            
            # Her katman için doğruluk kaybını güncelle
            for katman_adi in metrikler:
                metrikler[katman_adi].dogruluk_kaybi = dogruluk_kaybi
            
            # Maksimum kayıp aşıldıysa dur
            if dogruluk_kaybi > maksimum_dogruluk_kaybi:
                print(f"Adım {adim + 1}: Maksimum doğruluk kaybı aşıldı ({dogruluk_kaybi:.4f}). Durduruluyor.")
                break
        
        adim_metrikleri.append(metrikler)
    
    return model, adim_metrikleri

