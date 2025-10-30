"""
Bellek Yönetim Stratejileri

Farklı bellek yönetim stratejileri ve önbellek politikaları
"""

from typing import Any, Optional, Dict, List
from collections import OrderedDict
from abc import ABC, abstractmethod
import time


class BellekStratejisi(ABC):
    """
    Bellek yönetim stratejisi arayüzü
    """
    
    @abstractmethod
    def ekle(self, key: str, value: Any):
        """Değer ekle"""
        pass
    
    @abstractmethod
    def al(self, key: str) -> Optional[Any]:
        """Değer al"""
        pass
    
    @abstractmethod
    def sil(self, key: str):
        """Değer sil"""
        pass
    
    @abstractmethod
    def temizle(self):
        """Tüm değerleri temizle"""
        pass


class LRUOnbellek(BellekStratejisi):
    """
    LRU (Least Recently Used) Önbellek
    
    En az kullanılan öğeleri önce çıkarır.
    """
    
    def __init__(self, kapasite: int):
        """
        Args:
            kapasite: Maksimum öğe sayısı
        """
        self.kapasite = kapasite
        self._cache = OrderedDict()
        self._erisim_sayilari = {}
    
    def ekle(self, key: str, value: Any):
        """LRU önbelleğe ekle"""
        if key in self._cache:
            # Varolan öğeyi güncelle
            self._cache.move_to_end(key)
        else:
            # Yeni öğe ekle
            if len(self._cache) >= self.kapasite:
                # En eski öğeyi çıkar
                eski_key = next(iter(self._cache))
                self._cache.pop(eski_key)
                self._erisim_sayilari.pop(eski_key, None)
        
        self._cache[key] = value
        self._erisim_sayilari[key] = self._erisim_sayilari.get(key, 0)
    
    def al(self, key: str) -> Optional[Any]:
        """LRU önbellekten al"""
        if key in self._cache:
            # Erişildi, en sona taşı
            self._cache.move_to_end(key)
            self._erisim_sayilari[key] = self._erisim_sayilari.get(key, 0) + 1
            return self._cache[key]
        return None
    
    def sil(self, key: str):
        """Öğeyi sil"""
        if key in self._cache:
            self._cache.pop(key)
            self._erisim_sayilari.pop(key, None)
    
    def temizle(self):
        """Tüm önbelleği temizle"""
        self._cache.clear()
        self._erisim_sayilari.clear()
    
    def get_istatistikler(self) -> Dict[str, Any]:
        """Önbellek istatistikleri"""
        return {
            'mevcut_boyut': len(self._cache),
            'kapasite': self.kapasite,
            'doluluk_orani': len(self._cache) / self.kapasite if self.kapasite > 0 else 0,
            'toplam_erisim': sum(self._erisim_sayilari.values()),
            'en_cok_kullanilan': max(self._erisim_sayilari.items(), key=lambda x: x[1])[0] if self._erisim_sayilari else None
        }


class LFUOnbellek(BellekStratejisi):
    """
    LFU (Least Frequently Used) Önbellek
    
    En az sıklıkla kullanılan öğeleri önce çıkarır.
    """
    
    def __init__(self, kapasite: int):
        """
        Args:
            kapasite: Maksimum öğe sayısı
        """
        self.kapasite = kapasite
        self._cache = {}
        self._frekanslar = {}
        self._min_frekans = 0
        self._frekans_gruplari = {}  # frekans -> {key'ler}
    
    def _frekans_guncelle(self, key: str):
        """Bir key'in frekansını güncelle"""
        eski_frekans = self._frekanslar.get(key, 0)
        yeni_frekans = eski_frekans + 1
        
        self._frekanslar[key] = yeni_frekans
        
        # Frekans gruplarını güncelle
        if eski_frekans in self._frekans_gruplari:
            if key in self._frekans_gruplari[eski_frekans]:
                self._frekans_gruplari[eski_frekans].remove(key)
                if not self._frekans_gruplari[eski_frekans]:
                    del self._frekans_gruplari[eski_frekans]
        
        if yeni_frekans not in self._frekans_gruplari:
            self._frekans_gruplari[yeni_frekans] = set()
        self._frekans_gruplari[yeni_frekans].add(key)
        
        # Min frekansı güncelle
        if eski_frekans == self._min_frekans and not self._frekans_gruplari.get(eski_frekans):
            self._min_frekans = yeni_frekans
    
    def ekle(self, key: str, value: Any):
        """LFU önbelleğe ekle"""
        if self.kapasite <= 0:
            return
        
        if key in self._cache:
            # Varolan öğeyi güncelle
            self._cache[key] = value
            self._frekans_guncelle(key)
        else:
            # Kapasite doluysa en az kullanılanı çıkar
            if len(self._cache) >= self.kapasite:
                # En düşük frekanstaki bir key'i çıkar
                if self._min_frekans in self._frekans_gruplari:
                    cikarilacak = self._frekans_gruplari[self._min_frekans].pop()
                    del self._cache[cikarilacak]
                    del self._frekanslar[cikarilacak]
            
            # Yeni öğe ekle
            self._cache[key] = value
            self._frekanslar[key] = 1
            self._min_frekans = 1
            
            if 1 not in self._frekans_gruplari:
                self._frekans_gruplari[1] = set()
            self._frekans_gruplari[1].add(key)
    
    def al(self, key: str) -> Optional[Any]:
        """LFU önbellekten al"""
        if key in self._cache:
            self._frekans_guncelle(key)
            return self._cache[key]
        return None
    
    def sil(self, key: str):
        """Öğeyi sil"""
        if key in self._cache:
            frekans = self._frekanslar[key]
            
            del self._cache[key]
            del self._frekanslar[key]
            
            if frekans in self._frekans_gruplari:
                self._frekans_gruplari[frekans].discard(key)
                if not self._frekans_gruplari[frekans]:
                    del self._frekans_gruplari[frekans]
    
    def temizle(self):
        """Tüm önbelleği temizle"""
        self._cache.clear()
        self._frekanslar.clear()
        self._frekans_gruplari.clear()
        self._min_frekans = 0


class TTLOnbellek(BellekStratejisi):
    """
    TTL (Time To Live) Önbellek
    
    Belirli bir süre sonra öğeleri otomatik olarak çıkarır.
    """
    
    def __init__(self, kapasite: int, ttl_saniye: float = 300.0):
        """
        Args:
            kapasite: Maksimum öğe sayısı
            ttl_saniye: Yaşam süresi (saniye)
        """
        self.kapasite = kapasite
        self.ttl_saniye = ttl_saniye
        self._cache = {}
        self._zaman_damgalari = {}
    
    def _suresi_dolanlari_temizle(self):
        """Süresi dolan öğeleri temizle"""
        su_an = time.time()
        silinecekler = [
            key for key, zaman in self._zaman_damgalari.items()
            if su_an - zaman > self.ttl_saniye
        ]
        
        for key in silinecekler:
            self.sil(key)
    
    def ekle(self, key: str, value: Any):
        """TTL önbelleğe ekle"""
        self._suresi_dolanlari_temizle()
        
        if len(self._cache) >= self.kapasite and key not in self._cache:
            # En eski öğeyi çıkar
            en_eski_key = min(self._zaman_damgalari, key=self._zaman_damgalari.get)
            self.sil(en_eski_key)
        
        self._cache[key] = value
        self._zaman_damgalari[key] = time.time()
    
    def al(self, key: str) -> Optional[Any]:
        """TTL önbellekten al"""
        self._suresi_dolanlari_temizle()
        
        if key in self._cache:
            # Zaman damgasını güncelle (refresh TTL)
            self._zaman_damgalari[key] = time.time()
            return self._cache[key]
        return None
    
    def sil(self, key: str):
        """Öğeyi sil"""
        if key in self._cache:
            del self._cache[key]
            del self._zaman_damgalari[key]
    
    def temizle(self):
        """Tüm önbelleği temizle"""
        self._cache.clear()
        self._zaman_damgalari.clear()


class AdaptifOnbellek(BellekStratejisi):
    """
    Adaptif Önbellek
    
    Erişim paternlerine göre LRU ve LFU arasında otomatik geçiş yapar.
    """
    
    def __init__(self, kapasite: int, degerlendirme_penceresi: int = 1000):
        """
        Args:
            kapasite: Maksimum öğe sayısı
            degerlendirme_penceresi: Strateji değerlendirme penceresi
        """
        self.kapasite = kapasite
        self.degerlendirme_penceresi = degerlendirme_penceresi
        
        # İki stratejiyi tut
        self._lru = LRUOnbellek(kapasite)
        self._lfu = LFUOnbellek(kapasite)
        
        # Aktif strateji
        self._aktif_strateji = self._lru
        self._strateji_adi = "LRU"
        
        # Performans izleme
        self._istek_sayisi = 0
        self._lru_hit = 0
        self._lfu_hit = 0
    
    def _stratejiyi_degerlendir(self):
        """Stratejiyi değerlendir ve gerekirse değiştir"""
        if self._istek_sayisi % self.degerlendirme_penceresi == 0:
            lru_hit_rate = self._lru_hit / self.degerlendirme_penceresi
            lfu_hit_rate = self._lfu_hit / self.degerlendirme_penceresi
            
            # Daha iyi performans gösteren stratejiyi seç
            if lfu_hit_rate > lru_hit_rate and self._strateji_adi == "LRU":
                self._aktif_strateji = self._lfu
                self._strateji_adi = "LFU"
            elif lru_hit_rate > lfu_hit_rate and self._strateji_adi == "LFU":
                self._aktif_strateji = self._lru
                self._strateji_adi = "LRU"
            
            # Sayaçları sıfırla
            self._lru_hit = 0
            self._lfu_hit = 0
    
    def ekle(self, key: str, value: Any):
        """Her iki stratejiye de ekle"""
        self._lru.ekle(key, value)
        self._lfu.ekle(key, value)
    
    def al(self, key: str) -> Optional[Any]:
        """Aktif stratejiden al"""
        self._istek_sayisi += 1
        
        # Her iki stratejiden de dene (izleme için)
        lru_sonuc = self._lru.al(key)
        lfu_sonuc = self._lfu.al(key)
        
        if lru_sonuc is not None:
            self._lru_hit += 1
        if lfu_sonuc is not None:
            self._lfu_hit += 1
        
        self._stratejiyi_degerlendir()
        
        return self._aktif_strateji.al(key)
    
    def sil(self, key: str):
        """Her iki stratejiden de sil"""
        self._lru.sil(key)
        self._lfu.sil(key)
    
    def temizle(self):
        """Her iki stratejiyi de temizle"""
        self._lru.temizle()
        self._lfu.temizle()
        self._istek_sayisi = 0
        self._lru_hit = 0
        self._lfu_hit = 0
    
    def get_aktif_strateji(self) -> str:
        """Aktif strateji adını döndür"""
        return self._strateji_adi

