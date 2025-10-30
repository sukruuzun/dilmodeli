"""
Dinamik Kuantizasyon Motoru

Bu modül, Nash-Sürü teoremini kullanarak dinamik
kuantizasyon ve budama yapar.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass
from enum import Enum

from ..core.nash_suru_teorem import NashSuruTeorem, NashSuruParametreleri


class KuantizasyonTipi(Enum):
    """Kuantizasyon türleri"""
    INT8 = 8
    INT4 = 4
    INT2 = 2
    MIXED = "mixed"  # Karışık bit genişliği


@dataclass
class KuantizasyonKonfigurasyonu:
    """Kuantizasyon konfigürasyonu"""
    bit_genisligi: int = 8
    blok_boyutu: int = 64  # Kuantizasyon blok boyutu
    komsu_boyutu: int = 5  # Sürü davranışı için komşu sayısı
    nash_alpha: float = 0.2  # Nash dengesi ağırlığı
    dogruluk_esigi: float = 0.02  # Maksimum doğruluk kaybı
    dinamik_budama: bool = True
    budama_orani_hedefi: float = 0.3  # Hedef budama oranı


class DinamikKuantizasyon:
    """
    Dinamik Kuantizasyon ve Budama Motoru
    
    Bu sınıf, Nash-Sürü teoremini kullanarak:
    1. Ağırlık bloklarını dinamik olarak kuantize eder
    2. Komşu bloklarla etkileşimi dikkate alır
    3. Nash dengesi ile budama kararları verir
    
    Temel Fikir:
    - Her ağırlık bloğu bir "oyuncu"
    - Komşu blokların önemine bakarak kendi önemini belirler (Sürü)
    - Nash dengesi: Doğruluk vs bellek kullanımı trade-off
    """
    
    def __init__(
        self,
        config: Optional[KuantizasyonKonfigurasyonu] = None,
        nash_suru_params: Optional[NashSuruParametreleri] = None
    ):
        """
        Args:
            config: Kuantizasyon konfigürasyonu
            nash_suru_params: Nash-Sürü teoremi parametreleri
        """
        self.config = config or KuantizasyonKonfigurasyonu()
        self.nash_suru_teorem = NashSuruTeorem(nash_suru_params)
        
        # Kuantizasyon istatistikleri
        self.istatistikler = {
            'toplam_parametre': 0,
            'kuantize_parametre': 0,
            'budanan_parametre': 0,
            'ortalama_bit_genisligi': 0.0,
            'bellek_tasarrufu': 0.0
        }
        
    def agirlik_bloklarini_olustur(
        self,
        agirliklar: torch.Tensor
    ) -> Tuple[List[torch.Tensor], List[Tuple[int, ...]]]:
        """
        Ağırlık tensorünü bloklara böl
        
        Args:
            agirliklar: Ağırlık tensoru
            
        Returns:
            bloklar: Ağırlık blokları
            blok_pozisyonlari: Her bloğun tensor içindeki pozisyonu
        """
        orijinal_sekil = agirliklar.shape
        agirliklar_flat = agirliklar.flatten()
        
        blok_boyutu = self.config.blok_boyutu
        N_toplam = len(agirliklar_flat)
        N_blok = (N_toplam + blok_boyutu - 1) // blok_boyutu
        
        bloklar = []
        blok_pozisyonlari = []
        
        for i in range(N_blok):
            baslangic = i * blok_boyutu
            bitis = min((i + 1) * blok_boyutu, N_toplam)
            
            blok = agirliklar_flat[baslangic:bitis]
            bloklar.append(blok)
            blok_pozisyonlari.append((baslangic, bitis))
        
        return bloklar, blok_pozisyonlari
    
    def blok_onem_skorunu_hesapla(
        self,
        blok: torch.Tensor,
        metrik: str = "l2"
    ) -> float:
        """
        Bir bloğun önem skorunu hesapla
        
        Args:
            blok: Ağırlık bloğu
            metrik: Önem metriği ('l2', 'l1', 'max')
            
        Returns:
            onem_skoru: Bloğun önem skoru
        """
        if metrik == "l2":
            return float(torch.norm(blok, p=2).item())
        elif metrik == "l1":
            return float(torch.norm(blok, p=1).item())
        elif metrik == "max":
            return float(torch.max(torch.abs(blok)).item())
        else:
            return float(torch.norm(blok, p=2).item())
    
    def komsu_bloklari_bul(
        self,
        blok_idx: int,
        toplam_blok_sayisi: int
    ) -> List[int]:
        """
        Bir blok için komşu blokları bul (lokal etkileşim)
        
        Args:
            blok_idx: Blok indeksi
            toplam_blok_sayisi: Toplam blok sayısı
            
        Returns:
            komsu_indeksleri: Komşu blok indeksleri
        """
        komsu_yaricap = self.config.komsu_boyutu // 2
        
        baslangic = max(0, blok_idx - komsu_yaricap)
        bitis = min(toplam_blok_sayisi, blok_idx + komsu_yaricap + 1)
        
        komsu_indeksleri = list(range(baslangic, bitis))
        komsu_indeksleri.remove(blok_idx)  # Kendisini çıkar
        
        return komsu_indeksleri
    
    def nash_suru_budama_karari(
        self,
        blok_onem_skorlari: np.ndarray,
        budama_orani_hedefi: float
    ) -> np.ndarray:
        """
        Nash-Sürü ile budama kararı ver
        
        Her blok bir oyuncu olarak:
        1. Kendi önemine bakar (Rasyonel Çıkar)
        2. Komşu blokların önemine bakar (Sürü Etkileşimi)
        3. Nash dengesi: Toplam doğruluk kaybı vs bellek kazancı
        
        Args:
            blok_onem_skorlari: Her bloğun önem skoru [N_blok]
            budama_orani_hedefi: Hedef budama oranı [0, 1]
            
        Returns:
            budama_maskesi: Her blok için budama kararı [N_blok] (True = koru, False = buda)
        """
        N_blok = len(blok_onem_skorlari)
        
        # Her blok için sürü etkisiyle düzeltilmiş önem skoru
        duzenli_onem_skorlari = np.zeros(N_blok)
        
        for i in range(N_blok):
            kendi_onem = blok_onem_skorlari[i]
            
            # Komşu blokları bul
            komsu_indeksleri = self.komsu_bloklari_bul(i, N_blok)
            
            if len(komsu_indeksleri) > 0:
                komsu_onemleri = blok_onem_skorlari[komsu_indeksleri]
                ortalama_komsu_onem = np.mean(komsu_onemleri)
                
                # Sürü etkisi: Komşular önemliyse, bu blok da önemli sayılabilir
                suru_etkisi = self.config.nash_alpha * ortalama_komsu_onem
                duzenli_onem_skorlari[i] = kendi_onem + suru_etkisi
            else:
                duzenli_onem_skorlari[i] = kendi_onem
        
        # Nash dengesi: En önemli blokları koru, diğerlerini buda
        budanacak_blok_sayisi = int(N_blok * budama_orani_hedefi)
        
        if budanacak_blok_sayisi >= N_blok:
            return np.zeros(N_blok, dtype=bool)  # Hepsini buda
        elif budanacak_blok_sayisi <= 0:
            return np.ones(N_blok, dtype=bool)  # Hiçbirini budama
        
        # En az önemli blokları buda
        esik_idx = np.argsort(duzenli_onem_skorlari)[budanacak_blok_sayisi]
        esik_deger = duzenli_onem_skorlari[esik_idx]
        
        budama_maskesi = duzenli_onem_skorlari > esik_deger
        
        return budama_maskesi
    
    def blok_kuantize_et(
        self,
        blok: torch.Tensor,
        bit_genisligi: int
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Bir bloğu kuantize et
        
        Args:
            blok: Ağırlık bloğu
            bit_genisligi: Bit genişliği
            
        Returns:
            kuantize_blok: Kuantize edilmiş blok
            kuantizasyon_parametreleri: Scale ve zero_point
        """
        # Min-max kuantizasyon
        min_val = torch.min(blok)
        max_val = torch.max(blok)
        
        # Kuantizasyon aralığı
        n_levels = 2 ** bit_genisligi
        scale = (max_val - min_val) / (n_levels - 1)
        zero_point = min_val
        
        # Kuantize et
        kuantize_blok = torch.round((blok - zero_point) / scale)
        kuantize_blok = torch.clamp(kuantize_blok, 0, n_levels - 1)
        
        # Dequantize (floating point'e geri dön)
        dequantize_blok = kuantize_blok * scale + zero_point
        
        params = {
            'scale': scale,
            'zero_point': zero_point,
            'bit_genisligi': bit_genisligi
        }
        
        return dequantize_blok, params
    
    def agirlik_kuantize_ve_buda(
        self,
        agirliklar: torch.Tensor,
        budama_orani: Optional[float] = None
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Ağırlıkları Nash-Sürü ile kuantize ve buda
        
        Args:
            agirliklar: Ağırlık tensoru
            budama_orani: Hedef budama oranı (None ise config'den al)
            
        Returns:
            kuantize_agirliklar: Kuantize ve budanmış ağırlıklar
            bilgi: Kuantizasyon ve budama bilgileri
        """
        if budama_orani is None:
            budama_orani = self.config.budama_orani_hedefi
        
        orijinal_sekil = agirliklar.shape
        device = agirliklar.device
        
        # Blokları oluştur
        bloklar, blok_pozisyonlari = self.agirlik_bloklarini_olustur(agirliklar)
        N_blok = len(bloklar)
        
        # Her blok için önem skorunu hesapla
        blok_onem_skorlari = np.array([
            self.blok_onem_skorunu_hesapla(blok)
            for blok in bloklar
        ])
        
        # Nash-Sürü ile budama kararı
        budama_maskesi = self.nash_suru_budama_karari(
            blok_onem_skorlari,
            budama_orani
        )
        
        # Blokları kuantize et ve buda
        kuantize_bloklar = []
        kuantizasyon_bilgileri = []
        
        for i, (blok, koru) in enumerate(zip(bloklar, budama_maskesi)):
            if koru:
                # Bloğu kuantize et
                kuantize_blok, params = self.blok_kuantize_et(
                    blok, self.config.bit_genisligi
                )
                kuantize_bloklar.append(kuantize_blok)
                kuantizasyon_bilgileri.append(params)
            else:
                # Bloğu buda (sıfırla)
                kuantize_bloklar.append(torch.zeros_like(blok))
                kuantizasyon_bilgileri.append({'budandi': True})
        
        # Blokları birleştir
        kuantize_flat = torch.cat(kuantize_bloklar)
        kuantize_agirliklar = kuantize_flat.reshape(orijinal_sekil)
        
        # İstatistikler
        toplam_parametre = agirliklar.numel()
        budanan_parametre = int(np.sum(~budama_maskesi) * self.config.blok_boyutu)
        kuantize_parametre = toplam_parametre - budanan_parametre
        
        # Bellek tasarrufu hesapla
        orijinal_bellek = toplam_parametre * 4  # float32 = 4 bytes
        yeni_bellek = (
            kuantize_parametre * (self.config.bit_genisligi / 8) +
            N_blok * 8  # scale ve zero_point için
        )
        bellek_tasarrufu = 1.0 - (yeni_bellek / orijinal_bellek)
        
        self.istatistikler.update({
            'toplam_parametre': toplam_parametre,
            'kuantize_parametre': kuantize_parametre,
            'budanan_parametre': budanan_parametre,
            'ortalama_bit_genisligi': self.config.bit_genisligi,
            'bellek_tasarrufu': bellek_tasarrufu
        })
        
        bilgi = {
            'toplam_blok': N_blok,
            'korunan_blok': int(np.sum(budama_maskesi)),
            'budanan_blok': int(np.sum(~budama_maskesi)),
            'budama_orani': float(np.mean(~budama_maskesi)),
            'bellek_tasarrufu': bellek_tasarrufu,
            'orijinal_bellek_mb': orijinal_bellek / (1024 * 1024),
            'yeni_bellek_mb': yeni_bellek / (1024 * 1024)
        }
        
        return kuantize_agirliklar, bilgi
    
    def model_kuantize_et(
        self,
        model: nn.Module,
        hedef_katmanlar: Optional[List[str]] = None
    ) -> Tuple[nn.Module, Dict[str, Any]]:
        """
        Tüm modeli kuantize et
        
        Args:
            model: PyTorch modeli
            hedef_katmanlar: Kuantize edilecek katman isimleri (None ise tümü)
            
        Returns:
            kuantize_model: Kuantize edilmiş model
            toplam_bilgi: Toplam kuantizasyon bilgileri
        """
        toplam_bilgi = {
            'katman_bilgileri': {},
            'toplam_parametre': 0,
            'toplam_kuantize_parametre': 0,
            'toplam_budanan_parametre': 0,
            'ortalama_bellek_tasarrufu': 0.0
        }
        
        bellek_tasarrufları = []
        
        for name, param in model.named_parameters():
            # Hedef katman kontrolü
            if hedef_katmanlar and not any(target in name for target in hedef_katmanlar):
                continue
            
            # Sadece weight parametrelerini kuantize et
            if 'weight' not in name:
                continue
            
            # Kuantize et
            kuantize_agirlik, bilgi = self.agirlik_kuantize_ve_buda(param.data)
            param.data = kuantize_agirlik
            
            toplam_bilgi['katman_bilgileri'][name] = bilgi
            toplam_bilgi['toplam_parametre'] += self.istatistikler['toplam_parametre']
            toplam_bilgi['toplam_kuantize_parametre'] += self.istatistikler['kuantize_parametre']
            toplam_bilgi['toplam_budanan_parametre'] += self.istatistikler['budanan_parametre']
            bellek_tasarrufları.append(bilgi['bellek_tasarrufu'])
        
        if bellek_tasarrufları:
            toplam_bilgi['ortalama_bellek_tasarrufu'] = float(np.mean(bellek_tasarrufları))
        
        return model, toplam_bilgi
    
    def get_istatistikler(self) -> Dict[str, Any]:
        """İstatistikleri döndür"""
        return self.istatistikler.copy()

