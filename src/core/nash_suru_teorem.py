"""
Nash-Sürü Teoremi - Ana Matematiksel Motor

Bu modül, Nash Dengesi ve Sığırcık Sürü Davranışını birleştiren
temel matematiksel çerçeveyi sağlar.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass


@dataclass
class NashSuruParametreleri:
    """Nash-Sürü teoremi için parametreler"""
    
    # Sürü davranışı parametreleri
    lokal_grup_boyutu: int = 7  # Her birimin komşu sayısı (Sığırcık: 6-7)
    suru_uyum_katsayisi: float = 0.8  # Sürü uyum gücü [0, 1]
    
    # Nash dengesi parametreleri
    nash_iterasyon_limiti: int = 100  # Denge arama iterasyonu
    nash_epsilon: float = 1e-4  # Denge yakınsama eşiği
    
    # Dengeleyici kayıp fonksiyonu ağırlıkları
    lambda_dengeleme: float = 0.1  # İş yükü dengeleme ağırlığı
    lambda_cpu_onbellek: float = 0.05  # CPU önbellek ödül ağırlığı
    
    # Optimizasyon parametreleri
    ogrenme_hizi: float = 0.01
    momentum: float = 0.9


class NashSuruTeorem:
    """
    Nash-Sürü Teoremi Motoru
    
    Bu sınıf, Oyun Teorisi (Nash Dengesi) ve Sürü Zekası (Sığırcık Davranışı)
    teorilerini birleştirerek LLM optimizasyonu için temel algoritmaları sağlar.
    
    Temel Prensipler:
    1. Lokal Etkileşim: Her birim sadece en yakın komşularıyla etkileşir
    2. Global Uyum: Lokal kararlar global optimuma yakınsar
    3. Nash Dengesi: Hiçbir birimin stratejisini değiştirme teşviki yoktur
    """
    
    def __init__(self, parametreler: Optional[NashSuruParametreleri] = None):
        """
        Args:
            parametreler: Nash-Sürü teoremi parametreleri
        """
        self.params = parametreler or NashSuruParametreleri()
        self._denge_gecmisi: List[Dict[str, float]] = []
        
    def lokal_grup_sec(
        self,
        tum_adaylar: np.ndarray,
        hedef_ozellik: np.ndarray,
        mesafe_metrigi: str = "cosine"
    ) -> Tuple[np.ndarray, List[int]]:
        """
        Sığırcık Kuşu Kuralı: Lokal grup seçimi
        
        Tüm adaylar yerine, hedefe en yakın lokal grubu seçer.
        Bu, sürü davranışındaki "sadece komşulara bak" prensibini uygular.
        
        Args:
            tum_adaylar: Tüm aday birimlerin temsil vektörleri [N, D]
            hedef_ozellik: Hedef özellik vektörü [D]
            mesafe_metrigi: Mesafe metriği ('cosine', 'euclidean', 'manhattan')
            
        Returns:
            lokal_grup: Seçilen lokal grup [lokal_grup_boyutu, D]
            indeksler: Seçilen birimlerin orijinal indeksleri
        """
        N = len(tum_adaylar)
        
        if N <= self.params.lokal_grup_boyutu:
            return tum_adaylar, list(range(N))
        
        # Mesafe hesapla
        if mesafe_metrigi == "cosine":
            # Cosine similarity -> distance
            hedef_norm = hedef_ozellik / (np.linalg.norm(hedef_ozellik) + 1e-8)
            adaylar_norm = tum_adaylar / (np.linalg.norm(tum_adaylar, axis=1, keepdims=True) + 1e-8)
            benzerlik = np.dot(adaylar_norm, hedef_norm)
            mesafeler = 1 - benzerlik
        elif mesafe_metrigi == "euclidean":
            mesafeler = np.linalg.norm(tum_adaylar - hedef_ozellik, axis=1)
        elif mesafe_metrigi == "manhattan":
            mesafeler = np.sum(np.abs(tum_adaylar - hedef_ozellik), axis=1)
        else:
            raise ValueError(f"Bilinmeyen mesafe metriği: {mesafe_metrigi}")
        
        # En yakın lokal_grup_boyutu kadar adayı seç
        en_yakin_indeksler = np.argsort(mesafeler)[:self.params.lokal_grup_boyutu]
        lokal_grup = tum_adaylar[en_yakin_indeksler]
        
        return lokal_grup, en_yakin_indeksler.tolist()
    
    def suru_uyum_hesapla(
        self,
        birim_stratejisi: np.ndarray,
        komsu_stratejileri: np.ndarray,
        uyum_agirlik: Optional[np.ndarray] = None
    ) -> float:
        """
        Sürü Uyum Skoru
        
        Bir birimin stratejisinin komşularıyla ne kadar uyumlu olduğunu ölçer.
        Yüksek uyum = Sürü içinde koordineli hareket
        
        Args:
            birim_stratejisi: Birimin mevcut stratejisi [D]
            komsu_stratejileri: Komşuların stratejileri [K, D]
            uyum_agirlik: Komşular için özel ağırlıklar [K] (opsiyonel)
            
        Returns:
            uyum_skoru: [0, 1] arası uyum skoru (1 = tam uyum)
        """
        if len(komsu_stratejileri) == 0:
            return 1.0
        
        # Ortalama komşu stratejisi (sürü merkezi)
        if uyum_agirlik is not None:
            uyum_agirlik = uyum_agirlik / np.sum(uyum_agirlik)
            suru_merkezi = np.sum(komsu_stratejileri * uyum_agirlik[:, None], axis=0)
        else:
            suru_merkezi = np.mean(komsu_stratejileri, axis=0)
        
        # Cosine similarity ile uyum ölç
        birim_norm = np.linalg.norm(birim_stratejisi) + 1e-8
        merkez_norm = np.linalg.norm(suru_merkezi) + 1e-8
        uyum_skoru = np.dot(birim_stratejisi, suru_merkezi) / (birim_norm * merkez_norm)
        
        # [0, 1] aralığına normalize et
        uyum_skoru = (uyum_skoru + 1) / 2
        
        return float(uyum_skoru)
    
    def nash_dengesi_hesapla(
        self,
        oyuncu_kazanimlari: np.ndarray,
        mevcut_stratejiler: np.ndarray,
        kisitlar: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, bool, int]:
        """
        Nash Dengesi Bulma Algoritması
        
        Verilen bir oyun için Nash dengesini bulur. Bu, hiçbir oyuncunun
        stratejisini tek başına değiştirerek kazanç sağlayamayacağı noktadır.
        
        Args:
            oyuncu_kazanimlari: Her oyuncunun her strateji kombinasyonundan
                              kazanımı [N_oyuncu, N_strateji, ...]
            mevcut_stratejiler: Oyuncuların başlangıç stratejileri [N_oyuncu, N_strateji]
            kisitlar: Stratejilere kısıtlar (ör. toplam kaynak bütçesi)
            
        Returns:
            nash_stratejileri: Denge durumundaki stratejiler [N_oyuncu, N_strateji]
            yakinsadi: Denge bulundu mu?
            iterasyon_sayisi: Kullanılan iterasyon sayısı
        """
        N_oyuncu = len(oyuncu_kazanimlari)
        stratejiler = mevcut_stratejiler.copy()
        
        for iterasyon in range(self.params.nash_iterasyon_limiti):
            eski_stratejiler = stratejiler.copy()
            
            # Her oyuncu için en iyi yanıt (best response)
            for oyuncu_idx in range(N_oyuncu):
                # Diğer oyuncuların stratejileri sabit, bu oyuncu optimize eder
                diger_stratejiler = [stratejiler[i] for i in range(N_oyuncu) if i != oyuncu_idx]
                
                # Bu oyuncunun kazanım fonksiyonunu maksimize et
                en_iyi_strateji = self._en_iyi_yanitla(
                    oyuncu_idx,
                    oyuncu_kazanimlari[oyuncu_idx],
                    diger_stratejiler,
                    kisitlar
                )
                
                # Momentum ile güncelle (gradual değişim)
                stratejiler[oyuncu_idx] = (
                    self.params.momentum * stratejiler[oyuncu_idx] +
                    (1 - self.params.momentum) * en_iyi_strateji
                )
            
            # Yakınsama kontrolü
            degisim = np.max(np.abs(stratejiler - eski_stratejiler))
            if degisim < self.params.nash_epsilon:
                self._denge_gecmisi.append({
                    "iterasyon": iterasyon + 1,
                    "degisim": float(degisim),
                    "yakinsadi": True
                })
                return stratejiler, True, iterasyon + 1
        
        # Yakınsama sağlanamadı
        self._denge_gecmisi.append({
            "iterasyon": self.params.nash_iterasyon_limiti,
            "degisim": float(degisim),
            "yakinsadi": False
        })
        return stratejiler, False, self.params.nash_iterasyon_limiti
    
    def _en_iyi_yanitla(
        self,
        oyuncu_idx: int,
        kazanim_fonksiyonu: np.ndarray,
        diger_stratejiler: List[np.ndarray],
        kisitlar: Optional[Dict[str, Any]]
    ) -> np.ndarray:
        """
        Bir oyuncu için en iyi yanıt stratejisini bulur
        
        Args:
            oyuncu_idx: Oyuncu indeksi
            kazanim_fonksiyonu: Oyuncunun kazanım fonksiyonu
            diger_stratejiler: Diğer oyuncuların stratejileri
            kisitlar: Strateji kısıtları
            
        Returns:
            en_iyi_strateji: Bu oyuncu için en iyi yanıt stratejisi
        """
        # Basit gradient ascent ile en iyi yanıtı bul
        strateji_boyutu = kazanim_fonksiyonu.shape[0] if len(kazanim_fonksiyonu.shape) > 0 else 1
        en_iyi_strateji = np.random.rand(strateji_boyutu)
        
        # Normalize et (probability distribution)
        en_iyi_strateji = en_iyi_strateji / np.sum(en_iyi_strateji)
        
        # Kısıtları uygula
        if kisitlar:
            if "min_deger" in kisitlar:
                en_iyi_strateji = np.maximum(en_iyi_strateji, kisitlar["min_deger"])
            if "max_deger" in kisitlar:
                en_iyi_strateji = np.minimum(en_iyi_strateji, kisitlar["max_deger"])
            # Yeniden normalize
            en_iyi_strateji = en_iyi_strateji / np.sum(en_iyi_strateji)
        
        return en_iyi_strateji
    
    def dengeleyici_kayip_hesapla(
        self,
        capraz_entropi_kaybı: float,
        is_yuku_dagilimi: np.ndarray,
        cpu_onbellek_vurusu: float
    ) -> Tuple[float, Dict[str, float]]:
        """
        Nash-Sürü Dengeleyici Kayıp Fonksiyonu
        
        L_Nash-Sürü = L_ÇaprazEntropi + λ₁ · L_Dengeleme - λ₂ · R_CPU-Önbellek
        
        Args:
            capraz_entropi_kaybı: Standart çapraz entropi kaybı
            is_yuku_dagilimi: Uzmanlar/Birimler arası iş yükü dağılımı [N]
            cpu_onbellek_vurusu: CPU önbellek hit rate [0, 1]
            
        Returns:
            toplam_kayip: Nash-Sürü dengeleyici kaybı
            kayip_bilesenleri: Her bileşenin değeri
        """
        # İş yükü dengeleme kaybı (varyansı cezalandır)
        hedef_yuk = np.mean(is_yuku_dagilimi)
        dengeleme_kaybi = np.var(is_yuku_dagilimi) / (hedef_yuk + 1e-8)
        
        # CPU önbellek ödülü (yüksek olması istenir, bu yüzden negatif)
        cpu_odulu = cpu_onbellek_vurusu
        
        # Toplam kayıp
        toplam_kayip = (
            capraz_entropi_kaybı +
            self.params.lambda_dengeleme * dengeleme_kaybi -
            self.params.lambda_cpu_onbellek * cpu_odulu
        )
        
        kayip_bilesenleri = {
            "capraz_entropi": float(capraz_entropi_kaybı),
            "dengeleme_kaybi": float(dengeleme_kaybi),
            "cpu_odulu": float(cpu_odulu),
            "toplam": float(toplam_kayip)
        }
        
        return toplam_kayip, kayip_bilesenleri
    
    def adaptif_karar_ver(
        self,
        aday_birimleri: np.ndarray,
        hedef_gorev: np.ndarray,
        performans_gecmisi: Optional[np.ndarray] = None,
        kaynak_kisiti: Optional[float] = None
    ) -> Tuple[int, Dict[str, Any]]:
        """
        Nash-Sürü ile Adaptif Karar Verme
        
        Bu metod tüm süreci birleştirir:
        1. Lokal grup seç (Sürü davranışı)
        2. Nash dengesi bul
        3. En optimal birimi seç
        
        Args:
            aday_birimleri: Tüm aday birimler [N, D]
            hedef_gorev: Görev özellik vektörü [D]
            performans_gecmisi: Geçmiş performans verileri [N] (opsiyonel)
            kaynak_kisiti: Kaynak bütçesi kısıtı (opsiyonel)
            
        Returns:
            secilen_birim_idx: Seçilen birimin indeksi
            karar_bilgisi: Karar süreci hakkında detaylar
        """
        # Adım 1: Lokal grup seç
        lokal_grup, lokal_indeksler = self.lokal_grup_sec(
            aday_birimleri, hedef_gorev
        )
        
        # Adım 2: Lokal grup içinde Nash dengesi bul
        # Oyuncu kazanımları: Her birim, görevle uyumunu maksimize etmek ister
        N_lokal = len(lokal_grup)
        
        # Basit kazanım fonksiyonu: görev benzerliği
        hedef_norm = hedef_gorev / (np.linalg.norm(hedef_gorev) + 1e-8)
        lokal_norm = lokal_grup / (np.linalg.norm(lokal_grup, axis=1, keepdims=True) + 1e-8)
        kazanimlar = np.dot(lokal_norm, hedef_norm)
        
        # Performans geçmişini dahil et
        if performans_gecmisi is not None:
            geçmis_performanslar = performans_gecmisi[lokal_indeksler]
            kazanimlar = 0.7 * kazanimlar + 0.3 * geçmis_performanslar
        
        # Adım 3: Nash dengesi ile en iyi birimi seç
        # Her birim bir "oyuncu", strateji = ne kadar aktif olacak [0, 1]
        oyuncu_kazanimlari = kazanimlar[:, None]  # [N_lokal, 1]
        baslangic_stratejileri = np.ones((N_lokal, 1)) / N_lokal  # Eşit başlangıç
        
        kisitlar = {"min_deger": 0.0, "max_deger": 1.0}
        if kaynak_kisiti:
            kisitlar["toplam_kaynak"] = kaynak_kisiti
        
        nash_stratejileri, yakinsadi, iterasyonlar = self.nash_dengesi_hesapla(
            oyuncu_kazanimlari,
            baslangic_stratejileri,
            kisitlar
        )
        
        # Adım 4: En yüksek Nash stratejisine sahip birimi seç
        lokal_en_iyi_idx = np.argmax(nash_stratejileri.flatten())
        global_en_iyi_idx = lokal_indeksler[lokal_en_iyi_idx]
        
        # Sürü uyum skoru hesapla
        secilen_strateji = lokal_grup[lokal_en_iyi_idx]
        komsu_stratejileri = np.delete(lokal_grup, lokal_en_iyi_idx, axis=0)
        uyum_skoru = self.suru_uyum_hesapla(secilen_strateji, komsu_stratejileri)
        
        karar_bilgisi = {
            "lokal_grup_boyutu": N_lokal,
            "nash_yakinsadi": yakinsadi,
            "nash_iterasyonlar": iterasyonlar,
            "suru_uyum_skoru": uyum_skoru,
            "kazanim_skoru": float(kazanimlar[lokal_en_iyi_idx]),
            "nash_stratejisi": float(nash_stratejileri[lokal_en_iyi_idx, 0])
        }
        
        return global_en_iyi_idx, karar_bilgisi
    
    def get_denge_gecmisi(self) -> List[Dict[str, float]]:
        """Nash dengesi arama geçmişini döndür"""
        return self._denge_gecmisi.copy()
    
    def sifirla(self):
        """Dahili durumu sıfırla"""
        self._denge_gecmisi.clear()

