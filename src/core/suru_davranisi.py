"""
Sığırcık Sürü Davranışı Implementasyonu

Bu modül, sığırcık kuşlarının sürü davranışını modelleyen
algoritmaları içerir (Starling Murmuration).
"""

import numpy as np
from typing import List, Tuple, Optional, Callable
from dataclasses import dataclass


@dataclass
class SuruKurallari:
    """Sürü davranışı için temel kurallar (Boids algoritması benzeri)"""
    
    # Ayrılma (Separation): Çok yakın komşulardan uzaklaş
    ayrilma_agirligi: float = 1.0
    ayrilma_mesafesi: float = 0.5
    
    # Hizalanma (Alignment): Komşuların ortalama yönünü takip et
    hizalanma_agirligi: float = 1.0
    
    # Birleşme (Cohesion): Komşuların merkezine doğru hareket et
    birlesme_agirligi: float = 1.0
    
    # Hedefe yönelme (Goal seeking)
    hedef_agirligi: float = 2.0
    
    # Komşu algılama mesafesi
    algilama_yarıcapi: float = 3.0


class SuruDavranisi:
    """
    Sığırcık Sürü Davranışı Motoru
    
    Bu sınıf, sığırcık kuşlarının sürü davranışından esinlenerek
    LLM bileşenlerinin koordineli çalışmasını sağlar.
    
    Temel Prensipler:
    1. Lokal Algılama: Her birim sadece yakın komşuları görebilir
    2. Basit Kurallar: Ayrılma, hizalanma, birleşme
    3. Emergent Davranış: Lokal kurallardan global paternler çıkar
    """
    
    def __init__(self, kurallar: Optional[SuruKurallari] = None):
        """
        Args:
            kurallar: Sürü davranışı kuralları
        """
        self.kurallar = kurallar or SuruKurallari()
        self._hareket_gecmisi: List[np.ndarray] = []
        
    def komsu_bul(
        self,
        birim_pozisyonu: np.ndarray,
        tum_pozisyonlar: np.ndarray,
        birim_idx: int
    ) -> Tuple[np.ndarray, List[int]]:
        """
        Bir birim için algılama yarıçapı içindeki komşuları bulur
        
        Args:
            birim_pozisyonu: Birimin pozisyonu [D]
            tum_pozisyonlar: Tüm birimlerin pozisyonları [N, D]
            birim_idx: Birimin indeksi
            
        Returns:
            komsu_pozisyonlari: Komşuların pozisyonları [K, D]
            komsu_indeksleri: Komşuların indeksleri
        """
        # Mesafeleri hesapla
        mesafeler = np.linalg.norm(tum_pozisyonlar - birim_pozisyonu, axis=1)
        
        # Kendisini hariç tut
        mesafeler[birim_idx] = np.inf
        
        # Algılama yarıçapı içindeki komşuları bul
        komsu_maskesi = mesafeler < self.kurallar.algilama_yarıcapi
        komsu_indeksleri = np.where(komsu_maskesi)[0].tolist()
        komsu_pozisyonlari = tum_pozisyonlar[komsu_maskesi]
        
        return komsu_pozisyonlari, komsu_indeksleri
    
    def ayrilma_kuvveti_hesapla(
        self,
        birim_pozisyonu: np.ndarray,
        komsu_pozisyonlari: np.ndarray
    ) -> np.ndarray:
        """
        Ayrılma Kuralı: Çok yakın komşulardan uzaklaş
        
        Bu kural, birimlerin birbirlerine çok yaklaşmasını engeller,
        böylece kaynak çakışması ve gereksiz hesaplama önlenir.
        
        Args:
            birim_pozisyonu: Birimin mevcut pozisyonu [D]
            komsu_pozisyonlari: Komşu pozisyonları [K, D]
            
        Returns:
            ayrilma_vektoru: Ayrılma kuvveti vektörü [D]
        """
        if len(komsu_pozisyonlari) == 0:
            return np.zeros_like(birim_pozisyonu)
        
        # Her komşudan uzaklaşma vektörü
        farklar = birim_pozisyonu - komsu_pozisyonlari  # [K, D]
        mesafeler = np.linalg.norm(farklar, axis=1, keepdims=True) + 1e-8  # [K, 1]
        
        # Yakın komşular daha güçlü iterim uygular
        yakin_maskesi = mesafeler.flatten() < self.kurallar.ayrilma_mesafesi
        if not np.any(yakin_maskesi):
            return np.zeros_like(birim_pozisyonu)
        
        # Mesafe ile ters orantılı kuvvet (yakın = güçlü iterim)
        iterim_gücü = 1.0 / mesafeler[yakin_maskesi]  # [K', 1]
        iterim_vektorleri = farklar[yakin_maskesi] / mesafeler[yakin_maskesi]  # [K', D]
        
        # Toplam ayrılma kuvveti
        ayrilma_vektoru = np.sum(iterim_vektorleri * iterim_gücü, axis=0)
        
        # Normalize et
        norm = np.linalg.norm(ayrilma_vektoru)
        if norm > 0:
            ayrilma_vektoru = ayrilma_vektoru / norm
        
        return ayrilma_vektoru * self.kurallar.ayrilma_agirligi
    
    def hizalanma_kuvveti_hesapla(
        self,
        birim_hizi: np.ndarray,
        komsu_hizlari: np.ndarray
    ) -> np.ndarray:
        """
        Hizalanma Kuralı: Komşuların ortalama yönünü takip et
        
        Bu kural, birimlerin koordineli hareket etmesini sağlar,
        LLM'de bileşenlerin uyumlu çalışmasına karşılık gelir.
        
        Args:
            birim_hizi: Birimin mevcut hızı (strateji değişimi) [D]
            komsu_hizlari: Komşuların hızları [K, D]
            
        Returns:
            hizalanma_vektoru: Hizalanma kuvveti vektörü [D]
        """
        if len(komsu_hizlari) == 0:
            return np.zeros_like(birim_hizi)
        
        # Komşuların ortalama hızı
        ortalama_hiz = np.mean(komsu_hizlari, axis=0)
        
        # Ortalamaya doğru yönelme
        hizalanma_vektoru = ortalama_hiz - birim_hizi
        
        # Normalize et
        norm = np.linalg.norm(hizalanma_vektoru)
        if norm > 0:
            hizalanma_vektoru = hizalanma_vektoru / norm
        
        return hizalanma_vektoru * self.kurallar.hizalanma_agirligi
    
    def birlesme_kuvveti_hesapla(
        self,
        birim_pozisyonu: np.ndarray,
        komsu_pozisyonlari: np.ndarray
    ) -> np.ndarray:
        """
        Birleşme Kuralı: Komşuların merkezine doğru hareket et
        
        Bu kural, sürünün dağılmasını engeller ve grubu bir arada tutar,
        LLM'de bileşenlerin global hedefe katkıda bulunmasını sağlar.
        
        Args:
            birim_pozisyonu: Birimin mevcut pozisyonu [D]
            komsu_pozisyonlari: Komşu pozisyonları [K, D]
            
        Returns:
            birlesme_vektoru: Birleşme kuvveti vektörü [D]
        """
        if len(komsu_pozisyonlari) == 0:
            return np.zeros_like(birim_pozisyonu)
        
        # Komşuların merkezi (sürü merkezi)
        suru_merkezi = np.mean(komsu_pozisyonlari, axis=0)
        
        # Merkeze doğru yönelme
        birlesme_vektoru = suru_merkezi - birim_pozisyonu
        
        # Normalize et
        norm = np.linalg.norm(birlesme_vektoru)
        if norm > 0:
            birlesme_vektoru = birlesme_vektoru / norm
        
        return birlesme_vektoru * self.kurallar.birlesme_agirligi
    
    def hedef_kuvveti_hesapla(
        self,
        birim_pozisyonu: np.ndarray,
        hedef_pozisyonu: np.ndarray
    ) -> np.ndarray:
        """
        Hedefe Yönelme: Global hedefe doğru hareket et
        
        LLM bağlamında, bu optimal performansa doğru ilerlemeyi temsil eder.
        
        Args:
            birim_pozisyonu: Birimin mevcut pozisyonu [D]
            hedef_pozisyonu: Hedef pozisyon [D]
            
        Returns:
            hedef_vektoru: Hedefe yönelme kuvveti [D]
        """
        hedef_vektoru = hedef_pozisyonu - birim_pozisyonu
        
        # Normalize et
        norm = np.linalg.norm(hedef_vektoru)
        if norm > 0:
            hedef_vektoru = hedef_vektoru / norm
        
        return hedef_vektoru * self.kurallar.hedef_agirligi
    
    def suru_hareketini_guncelle(
        self,
        mevcut_pozisyonlar: np.ndarray,
        mevcut_hizlar: np.ndarray,
        hedef_pozisyonu: Optional[np.ndarray] = None,
        ozel_kuvvetler: Optional[List[Callable]] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Tüm sürünün hareketini bir adım günceller
        
        Bu, LLM optimizasyonunda bir iterasyona karşılık gelir.
        
        Args:
            mevcut_pozisyonlar: Tüm birimlerin pozisyonları [N, D]
            mevcut_hizlar: Tüm birimlerin hızları [N, D]
            hedef_pozisyonu: Global hedef pozisyon [D] (opsiyonel)
            ozel_kuvvetler: Ek kuvvet fonksiyonları (opsiyonel)
            
        Returns:
            yeni_pozisyonlar: Güncellenmiş pozisyonlar [N, D]
            yeni_hizlar: Güncellenmiş hızlar [N, D]
        """
        N, D = mevcut_pozisyonlar.shape
        yeni_hizlar = np.zeros_like(mevcut_hizlar)
        
        for i in range(N):
            # Komşuları bul
            komsu_pozisyonlari, komsu_indeksleri = self.komsu_bul(
                mevcut_pozisyonlar[i], mevcut_pozisyonlar, i
            )
            
            if len(komsu_indeksleri) > 0:
                komsu_hizlari = mevcut_hizlar[komsu_indeksleri]
            else:
                komsu_hizlari = np.array([])
            
            # Sürü kurallarını uygula
            ayrilma = self.ayrilma_kuvveti_hesapla(
                mevcut_pozisyonlar[i], komsu_pozisyonlari
            )
            hizalanma = self.hizalanma_kuvveti_hesapla(
                mevcut_hizlar[i], komsu_hizlari
            )
            birlesme = self.birlesme_kuvveti_hesapla(
                mevcut_pozisyonlar[i], komsu_pozisyonlari
            )
            
            # Toplam kuvvet
            toplam_kuvvet = ayrilma + hizalanma + birlesme
            
            # Hedefe yönelme (varsa)
            if hedef_pozisyonu is not None:
                hedef_kuvveti = self.hedef_kuvveti_hesapla(
                    mevcut_pozisyonlar[i], hedef_pozisyonu
                )
                toplam_kuvvet += hedef_kuvveti
            
            # Özel kuvvetler (varsa)
            if ozel_kuvvetler:
                for kuvvet_fn in ozel_kuvvetler:
                    ozel_kuvvet = kuvvet_fn(i, mevcut_pozisyonlar, komsu_indeksleri)
                    toplam_kuvvet += ozel_kuvvet
            
            # Hızı güncelle
            yeni_hizlar[i] = mevcut_hizlar[i] + toplam_kuvvet
            
            # Hız sınırlaması (stability için)
            max_hiz = 1.0
            hiz_normu = np.linalg.norm(yeni_hizlar[i])
            if hiz_normu > max_hiz:
                yeni_hizlar[i] = yeni_hizlar[i] / hiz_normu * max_hiz
        
        # Pozisyonları güncelle
        yeni_pozisyonlar = mevcut_pozisyonlar + yeni_hizlar
        
        # Geçmişi kaydet
        self._hareket_gecmisi.append(yeni_pozisyonlar.copy())
        
        return yeni_pozisyonlar, yeni_hizlar
    
    def suru_metriklerini_hesapla(
        self,
        pozisyonlar: np.ndarray,
        hizlar: np.ndarray
    ) -> dict:
        """
        Sürünün mevcut durumu hakkında metrikler hesaplar
        
        Args:
            pozisyonlar: Tüm birimlerin pozisyonları [N, D]
            hizlar: Tüm birimlerin hızları [N, D]
            
        Returns:
            metrikler: Sürü metrikleri
        """
        N = len(pozisyonlar)
        
        # Sürü merkezi
        suru_merkezi = np.mean(pozisyonlar, axis=0)
        
        # Dağılım (variance)
        merkeze_mesafeler = np.linalg.norm(pozisyonlar - suru_merkezi, axis=1)
        dagilim = np.var(merkeze_mesafeler)
        
        # Ortalama hız
        ortalama_hiz_normu = np.mean(np.linalg.norm(hizlar, axis=1))
        
        # Hizalanma derecesi (hızların ne kadar paralel olduğu)
        if ortalama_hiz_normu > 1e-6:
            normalize_hizlar = hizlar / (np.linalg.norm(hizlar, axis=1, keepdims=True) + 1e-8)
            ortalama_yon = np.mean(normalize_hizlar, axis=0)
            hizalanma_derecesi = np.linalg.norm(ortalama_yon)
        else:
            hizalanma_derecesi = 0.0
        
        # Yoğunluk (average number of neighbors)
        toplam_komsu = 0
        for i in range(N):
            _, komsu_indeksleri = self.komsu_bul(pozisyonlar[i], pozisyonlar, i)
            toplam_komsu += len(komsu_indeksleri)
        ortalama_komsu_sayisi = toplam_komsu / N if N > 0 else 0
        
        metrikler = {
            "suru_merkezi": suru_merkezi.tolist(),
            "dagilim": float(dagilim),
            "ortalama_hiz": float(ortalama_hiz_normu),
            "hizalanma_derecesi": float(hizalanma_derecesi),
            "ortalama_komsu_sayisi": float(ortalama_komsu_sayisi),
            "birim_sayisi": N
        }
        
        return metrikler
    
    def get_hareket_gecmisi(self) -> List[np.ndarray]:
        """Sürü hareket geçmişini döndür"""
        return self._hareket_gecmisi.copy()
    
    def sifirla(self):
        """Dahili durumu sıfırla"""
        self._hareket_gecmisi.clear()

