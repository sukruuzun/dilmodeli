"""
Uzman Havuzu - MoE uzmanlarının yönetimi
"""

import torch
import torch.nn as nn
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from enum import Enum


class UzmanTipi(Enum):
    """Uzman türleri"""
    GENEL = "genel"  # Genel amaçlı
    KOD = "kod"  # Kod üretimi
    MATEMATIK = "matematik"  # Matematiksel akıl yürütme
    SOHBET = "sohbet"  # Konuşma
    ANALIZ = "analiz"  # Veri analizi
    OZEL = "ozel"  # Özel görev


@dataclass
class UzmanKonfigurasyonu:
    """Uzman konfigürasyonu"""
    uzman_id: int
    uzman_tipi: UzmanTipi
    giris_boyutu: int
    gizli_boyutu: int
    aktivasyon: str = "gelu"
    dropout: float = 0.1


class Uzman(nn.Module):
    """
    Tek bir Uzman modülü
    
    Basit bir FFN (Feed-Forward Network) veya daha karmaşık bir ağ olabilir.
    """
    
    def __init__(self, config: UzmanKonfigurasyonu):
        """
        Args:
            config: Uzman konfigürasyonu
        """
        super().__init__()
        
        self.config = config
        self.uzman_id = config.uzman_id
        self.uzman_tipi = config.uzman_tipi
        
        # Aktivasyon fonksiyonu
        if config.aktivasyon == "gelu":
            aktivasyon = nn.GELU()
        elif config.aktivasyon == "relu":
            aktivasyon = nn.ReLU()
        elif config.aktivasyon == "silu":
            aktivasyon = nn.SiLU()
        else:
            aktivasyon = nn.GELU()
        
        # FFN: input -> gizli -> output
        self.network = nn.Sequential(
            nn.Linear(config.giris_boyutu, config.gizli_boyutu),
            aktivasyon,
            nn.Dropout(config.dropout),
            nn.Linear(config.gizli_boyutu, config.giris_boyutu)
        )
        
        # Performans metrikleri
        self.register_buffer('toplam_calistirma', torch.tensor(0))
        self.register_buffer('toplam_sure', torch.tensor(0.0))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [batch_size, hidden_dim]
            
        Returns:
            output: Uzman çıktısı [batch_size, hidden_dim]
        """
        return self.network(x)
    
    def get_performans_metrikleri(self) -> Dict[str, Any]:
        """Performans metriklerini döndür"""
        ortalama_sure = (
            self.toplam_sure.item() / self.toplam_calistirma.item()
            if self.toplam_calistirma.item() > 0 else 0.0
        )
        
        return {
            'uzman_id': self.uzman_id,
            'uzman_tipi': self.uzman_tipi.value,
            'toplam_calistirma': int(self.toplam_calistirma.item()),
            'ortalama_sure_ms': ortalama_sure * 1000,
            'parametre_sayisi': sum(p.numel() for p in self.parameters())
        }


class UzmanHavuzu(nn.Module):
    """
    Uzman Havuzu
    
    Tüm uzmanları yöneten ve dinamik olarak aktif/pasif
    yapabilen modül.
    """
    
    def __init__(
        self,
        giris_boyutu: int,
        uzman_konfigurasyonlari: List[UzmanKonfigurasyonu]
    ):
        """
        Args:
            giris_boyutu: Input boyutu
            uzman_konfigurasyonlari: Uzman konfigürasyonları
        """
        super().__init__()
        
        self.giris_boyutu = giris_boyutu
        self.uzman_sayisi = len(uzman_konfigurasyonlari)
        
        # Uzmanları oluştur
        self.uzmanlar = nn.ModuleList([
            Uzman(config) for config in uzman_konfigurasyonlari
        ])
        
        # Aktif uzman maskesi
        self.register_buffer(
            'aktif_uzmanlar',
            torch.ones(self.uzman_sayisi, dtype=torch.bool)
        )
        
    def forward(
        self,
        x: torch.Tensor,
        uzman_indeksleri: torch.Tensor,
        agirliklar: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Seçilen uzmanları çalıştır
        
        Args:
            x: Input tensor [batch_size, hidden_dim]
            uzman_indeksleri: Seçilen uzman indeksleri [batch_size, top_k]
            agirliklar: Uzman ağırlıkları [batch_size, top_k] (opsiyonel)
            
        Returns:
            output: Ağırlıklı uzman çıktıları [batch_size, hidden_dim]
        """
        batch_size = x.shape[0]
        top_k = uzman_indeksleri.shape[1]
        
        if agirliklar is None:
            agirliklar = torch.ones_like(uzman_indeksleri, dtype=torch.float32) / top_k
        
        output = torch.zeros_like(x)
        
        for b in range(batch_size):
            for k in range(top_k):
                uzman_idx = uzman_indeksleri[b, k].item()
                
                # Uzman aktif mi kontrol et
                if not self.aktif_uzmanlar[uzman_idx]:
                    continue
                
                agirlik = agirliklar[b, k]
                uzman_cikti = self.uzmanlar[uzman_idx](x[b:b+1])
                output[b] += agirlik * uzman_cikti.squeeze(0)
        
        return output
    
    def uzmani_aktif_et(self, uzman_id: int):
        """Uzmanı aktif et"""
        if 0 <= uzman_id < self.uzman_sayisi:
            self.aktif_uzmanlar[uzman_id] = True
    
    def uzmani_pasif_et(self, uzman_id: int):
        """Uzmanı pasif et"""
        if 0 <= uzman_id < self.uzman_sayisi:
            self.aktif_uzmanlar[uzman_id] = False
    
    def get_aktif_uzman_sayisi(self) -> int:
        """Aktif uzman sayısını döndür"""
        return int(torch.sum(self.aktif_uzmanlar).item())
    
    def get_toplam_parametre_sayisi(self) -> int:
        """Toplam parametre sayısını döndür"""
        return sum(p.numel() for p in self.parameters())
    
    def get_aktif_parametre_sayisi(self) -> int:
        """Aktif uzmanların parametre sayısını döndür"""
        toplam = 0
        for i, uzman in enumerate(self.uzmanlar):
            if self.aktif_uzmanlar[i]:
                toplam += sum(p.numel() for p in uzman.parameters())
        return toplam
    
    def get_havuz_istatistikleri(self) -> Dict[str, Any]:
        """Havuz istatistiklerini döndür"""
        uzman_metrikleri = [
            uzman.get_performans_metrikleri()
            for uzman in self.uzmanlar
        ]
        
        return {
            'toplam_uzman': self.uzman_sayisi,
            'aktif_uzman': self.get_aktif_uzman_sayisi(),
            'toplam_parametre': self.get_toplam_parametre_sayisi(),
            'aktif_parametre': self.get_aktif_parametre_sayisi(),
            'uzman_metrikleri': uzman_metrikleri
        }

