"""
Nash-Sürü Dengeleyici Kayıp Fonksiyonu

L_Nash-Sürü = L_ÇaprazEntropi + λ₁ · L_Dengeleme - λ₂ · R_CPU-Önbellek
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Dict, Any, List
from dataclasses import dataclass


@dataclass
class KayipAgirliklari:
    """Kayıp fonksiyonu ağırlıkları"""
    
    # Temel kayıp (çapraz entropi vs.)
    lambda_temel: float = 1.0
    
    # İş yükü dengeleme
    lambda_dengeleme: float = 0.1
    
    # CPU önbellek ödülü
    lambda_cpu_onbellek: float = 0.05
    
    # Sürü uyum ödülü
    lambda_suru_uyumu: float = 0.03
    
    # Nash dengesi regularizasyonu
    lambda_nash_dengesi: float = 0.02


class NashSuruKayipFonksiyonu(nn.Module):
    """
    Nash-Sürü Dengeleyici Kayıp Fonksiyonu
    
    Bu kayıp fonksiyonu:
    1. Temel görev kaybını (çapraz entropi) korur
    2. İş yükü dengelemesini teşvik eder
    3. CPU önbellek verimliliğini ödüllendirir
    4. Sürü uyumunu artırır
    5. Nash dengesi yakınsamasını sağlar
    """
    
    def __init__(
        self,
        agirliklar: Optional[KayipAgirliklari] = None,
        temel_kayip_fn: Optional[nn.Module] = None
    ):
        """
        Args:
            agirliklar: Kayıp ağırlıkları
            temel_kayip_fn: Temel kayıp fonksiyonu (default: CrossEntropyLoss)
        """
        super().__init__()
        
        self.agirliklar = agirliklar or KayipAgirliklari()
        self.temel_kayip_fn = temel_kayip_fn or nn.CrossEntropyLoss()
        
        # Kayıp bileşenlerinin geçmişi
        self._kayip_gecmisi = {
            'temel': [],
            'dengeleme': [],
            'cpu_onbellek': [],
            'suru_uyumu': [],
            'nash_dengesi': [],
            'toplam': []
        }
    
    def dengeleme_kaybi_hesapla(
        self,
        is_yuku_dagilimi: torch.Tensor
    ) -> torch.Tensor:
        """
        İş yükü dengeleme kaybı
        
        Uzmanlar/birimler arası iş yükü dağılımının dengesizliğini cezalandırır.
        
        Args:
            is_yuku_dagilimi: Her birimin iş yükü [N_birim]
            
        Returns:
            dengeleme_kaybi: Dengeleme kaybı (yüksek = dengesiz)
        """
        if len(is_yuku_dagilimi) == 0:
            return torch.tensor(0.0)
        
        # Hedef: Eşit iş yükü dağılımı
        ortalama_yuk = torch.mean(is_yuku_dagilimi)
        
        # Varyans ile dengesizliği ölç
        varyans = torch.var(is_yuku_dagilimi)
        
        # Normalize et (relative variance)
        dengeleme_kaybi = varyans / (ortalama_yuk + 1e-8)
        
        return dengeleme_kaybi
    
    def cpu_onbellek_odulu_hesapla(
        self,
        onbellek_durumu: torch.Tensor,
        secilen_indeksler: torch.Tensor
    ) -> torch.Tensor:
        """
        CPU önbellek ödülü
        
        Önbellekte bulunan birimlerin seçilmesini ödüllendirir.
        
        Args:
            onbellek_durumu: Her birimin önbellek durumu [N_birim] (1=içerde, 0=dışarda)
            secilen_indeksler: Seçilen birim indeksleri [N_secim]
            
        Returns:
            cpu_odulu: CPU önbellek ödülü (yüksek = iyi)
        """
        if len(secilen_indeksler) == 0:
            return torch.tensor(0.0)
        
        # Seçilen birimlerin önbellek durumları
        secilen_onbellek_durumlari = onbellek_durumu[secilen_indeksler]
        
        # Önbellek hit rate
        hit_rate = torch.mean(secilen_onbellek_durumlari.float())
        
        return hit_rate
    
    def suru_uyum_odulu_hesapla(
        self,
        birim_stratejileri: torch.Tensor,
        komsu_indeksleri: List[List[int]]
    ) -> torch.Tensor:
        """
        Sürü uyum ödülü
        
        Birimlerin komşularıyla uyumlu stratejiler seçmesini ödüllendirir.
        
        Args:
            birim_stratejileri: Her birimin stratejisi [N_birim, D]
            komsu_indeksleri: Her birim için komşu indeksleri
            
        Returns:
            uyum_odulu: Sürü uyum ödülü (yüksek = uyumlu)
        """
        N_birim = len(birim_stratejileri)
        
        if N_birim == 0:
            return torch.tensor(0.0)
        
        uyum_skorlari = []
        
        for i in range(N_birim):
            if len(komsu_indeksleri[i]) == 0:
                continue
            
            birim_stratejisi = birim_stratejileri[i]
            komsu_stratejileri = birim_stratejileri[komsu_indeksleri[i]]
            
            # Ortalama komşu stratejisi
            ortalama_komsu = torch.mean(komsu_stratejileri, dim=0)
            
            # Cosine similarity
            birim_norm = torch.norm(birim_stratejisi)
            komsu_norm = torch.norm(ortalama_komsu)
            
            if birim_norm > 1e-8 and komsu_norm > 1e-8:
                cosine_sim = torch.dot(birim_stratejisi, ortalama_komsu) / (birim_norm * komsu_norm)
                uyum_skorlari.append(cosine_sim)
        
        if len(uyum_skorlari) == 0:
            return torch.tensor(0.0)
        
        # Ortalama uyum
        uyum_odulu = torch.mean(torch.stack(uyum_skorlari))
        
        # [0, 1] aralığına normalize
        uyum_odulu = (uyum_odulu + 1) / 2
        
        return uyum_odulu
    
    def nash_dengesi_regularizasyonu_hesapla(
        self,
        oyuncu_kazanimlari: torch.Tensor,
        mevcut_stratejiler: torch.Tensor
    ) -> torch.Tensor:
        """
        Nash dengesi regularizasyonu
        
        Stratejilerin Nash dengesine yakınsamasını teşvik eder.
        
        Args:
            oyuncu_kazanimlari: Her oyuncunun kazanımları [N_oyuncu]
            mevcut_stratejiler: Mevcut stratejiler [N_oyuncu, D]
            
        Returns:
            nash_regularizasyon: Nash dengesi regularizasyonu
        """
        N_oyuncu = len(oyuncu_kazanimlari)
        
        if N_oyuncu == 0:
            return torch.tensor(0.0)
        
        # Nash dengesi: Hiçbir oyuncu tek başına stratejisini değiştirerek
        # kazanç sağlayamazsa dengedir.
        
        # Her oyuncu için: Mevcut kazanç vs maksimum potansiyel kazanç
        mevcut_kazanclar = oyuncu_kazanimlari
        
        # Basit yaklaşım: Kazançların varyansını minimize et (adil dağılım)
        kazanc_varyansi = torch.var(mevcut_kazanclar)
        
        # Normalize et
        ortalama_kazanc = torch.mean(mevcut_kazanclar)
        nash_regularizasyon = kazanc_varyansi / (ortalama_kazanc + 1e-8)
        
        return nash_regularizasyon
    
    def forward(
        self,
        tahminler: torch.Tensor,
        hedefler: torch.Tensor,
        ek_bilgiler: Optional[Dict[str, Any]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Nash-Sürü kayıp fonksiyonu forward pass
        
        Args:
            tahminler: Model tahminleri
            hedefler: Gerçek hedefler
            ek_bilgiler: Ek bilgiler (iş yükü, önbellek durumu vs.)
            
        Returns:
            kayiplar: Kayıp bileşenleri
        """
        # Temel kayıp (çapraz entropi)
        temel_kayip = self.temel_kayip_fn(tahminler, hedefler)
        
        # Ek kayıplar/ödüller
        dengeleme_kaybi = torch.tensor(0.0)
        cpu_odulu = torch.tensor(0.0)
        suru_uyum_odulu = torch.tensor(0.0)
        nash_regularizasyon = torch.tensor(0.0)
        
        if ek_bilgiler:
            # İş yükü dengeleme
            if 'is_yuku_dagilimi' in ek_bilgiler:
                dengeleme_kaybi = self.dengeleme_kaybi_hesapla(
                    ek_bilgiler['is_yuku_dagilimi']
                )
            
            # CPU önbellek ödülü
            if 'onbellek_durumu' in ek_bilgiler and 'secilen_indeksler' in ek_bilgiler:
                cpu_odulu = self.cpu_onbellek_odulu_hesapla(
                    ek_bilgiler['onbellek_durumu'],
                    ek_bilgiler['secilen_indeksler']
                )
            
            # Sürü uyum ödülü
            if 'birim_stratejileri' in ek_bilgiler and 'komsu_indeksleri' in ek_bilgiler:
                suru_uyum_odulu = self.suru_uyum_odulu_hesapla(
                    ek_bilgiler['birim_stratejileri'],
                    ek_bilgiler['komsu_indeksleri']
                )
            
            # Nash dengesi regularizasyonu
            if 'oyuncu_kazanimlari' in ek_bilgiler and 'mevcut_stratejiler' in ek_bilgiler:
                nash_regularizasyon = self.nash_dengesi_regularizasyonu_hesapla(
                    ek_bilgiler['oyuncu_kazanimlari'],
                    ek_bilgiler['mevcut_stratejiler']
                )
        
        # Toplam kayıp
        toplam_kayip = (
            self.agirliklar.lambda_temel * temel_kayip +
            self.agirliklar.lambda_dengeleme * dengeleme_kaybi -
            self.agirliklar.lambda_cpu_onbellek * cpu_odulu -
            self.agirliklar.lambda_suru_uyumu * suru_uyum_odulu +
            self.agirliklar.lambda_nash_dengesi * nash_regularizasyon
        )
        
        # Geçmişe kaydet
        self._kayip_gecmisi['temel'].append(float(temel_kayip.item()))
        self._kayip_gecmisi['dengeleme'].append(float(dengeleme_kaybi.item()))
        self._kayip_gecmisi['cpu_onbellek'].append(float(cpu_odulu.item()))
        self._kayip_gecmisi['suru_uyumu'].append(float(suru_uyum_odulu.item()))
        self._kayip_gecmisi['nash_dengesi'].append(float(nash_regularizasyon.item()))
        self._kayip_gecmisi['toplam'].append(float(toplam_kayip.item()))
        
        return {
            'toplam_kayip': toplam_kayip,
            'temel_kayip': temel_kayip,
            'dengeleme_kaybi': dengeleme_kaybi,
            'cpu_odulu': cpu_odulu,
            'suru_uyum_odulu': suru_uyum_odulu,
            'nash_regularizasyon': nash_regularizasyon
        }
    
    def get_kayip_gecmisi(self) -> Dict[str, List[float]]:
        """Kayıp geçmişini döndür"""
        return {k: v.copy() for k, v in self._kayip_gecmisi.items()}
    
    def kayip_istatistiklerini_hesapla(self) -> Dict[str, Dict[str, float]]:
        """Kayıp istatistiklerini hesapla"""
        istatistikler = {}
        
        for bileşen, degerler in self._kayip_gecmisi.items():
            if degerler:
                istatistikler[bileşen] = {
                    'ortalama': float(np.mean(degerler)),
                    'std': float(np.std(degerler)),
                    'min': float(np.min(degerler)),
                    'max': float(np.max(degerler)),
                    'son': float(degerler[-1])
                }
            else:
                istatistikler[bileşen] = {
                    'ortalama': 0.0,
                    'std': 0.0,
                    'min': 0.0,
                    'max': 0.0,
                    'son': 0.0
                }
        
        return istatistikler
    
    def gecmisi_sifirla(self):
        """Kayıp geçmişini sıfırla"""
        for key in self._kayip_gecmisi:
            self._kayip_gecmisi[key].clear()

