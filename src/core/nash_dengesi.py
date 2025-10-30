"""
Nash Dengesi Hesaplama Motoru

Bu modül, oyun teorisinden Nash dengesini hesaplamak için
özelleştirilmiş algoritmalar sağlar.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Callable
from enum import Enum


class OyunTipi(Enum):
    """Oyun türleri"""
    SIMETRIK = "simetrik"  # Tüm oyuncular aynı kazanım yapısına sahip
    ASIMETRIK = "asimetrik"  # Oyuncular farklı kazanım yapılarına sahip
    SIFIR_TOPLAMLI = "sifir_toplamli"  # Bir oyuncunun kazancı diğerinin kaybı
    KOOPERATIF = "kooperatif"  # Oyuncular işbirliği yapabilir


class NashDengesiMotoru:
    """
    Nash Dengesi Hesaplama Motoru
    
    Bu sınıf, farklı oyun türleri için Nash dengesini bulma
    algoritmalarını implement eder.
    
    Nash Dengesi: Hiçbir oyuncunun tek başına stratejisini değiştirerek
    daha iyi kazanç elde edemeyeceği strateji profili.
    """
    
    def __init__(
        self,
        oyun_tipi: OyunTipi = OyunTipi.ASIMETRIK,
        epsilon: float = 1e-4,
        max_iterasyon: int = 100
    ):
        """
        Args:
            oyun_tipi: Oyun türü
            epsilon: Yakınsama eşiği
            max_iterasyon: Maksimum iterasyon sayısı
        """
        self.oyun_tipi = oyun_tipi
        self.epsilon = epsilon
        self.max_iterasyon = max_iterasyon
        self._gecmis: List[Dict] = []
        
    def saf_strateji_nash(
        self,
        kazanim_matrisi: np.ndarray
    ) -> Tuple[Optional[Tuple[int, ...]], bool]:
        """
        Saf Strateji Nash Dengesi Bulma
        
        2 oyunculu oyunlar için saf strateji Nash dengesini bulur.
        
        Args:
            kazanim_matrisi: Kazanım matrisi [N_strateji_1, N_strateji_2, 2]
                           Son boyut: [oyuncu1_kazanci, oyuncu2_kazanci]
            
        Returns:
            nash_dengesi: Nash dengesi strateji indeksleri (yoksa None)
            bulundu: Denge bulundu mu?
        """
        if kazanim_matrisi.ndim != 3 or kazanim_matrisi.shape[2] != 2:
            raise ValueError("Kazanım matrisi [N, M, 2] boyutunda olmalı")
        
        N, M, _ = kazanim_matrisi.shape
        
        # Her hücre için en iyi yanıt kontrolü
        for i in range(N):
            for j in range(M):
                # Oyuncu 1 için: j sabit, i değişken
                oyuncu1_en_iyi = True
                for i_alt in range(N):
                    if kazanim_matrisi[i_alt, j, 0] > kazanim_matrisi[i, j, 0]:
                        oyuncu1_en_iyi = False
                        break
                
                # Oyuncu 2 için: i sabit, j değişken
                oyuncu2_en_iyi = True
                for j_alt in range(M):
                    if kazanim_matrisi[i, j_alt, 1] > kazanim_matrisi[i, j, 1]:
                        oyuncu2_en_iyi = False
                        break
                
                # Her iki oyuncu için de en iyi yanıt → Nash dengesi
                if oyuncu1_en_iyi and oyuncu2_en_iyi:
                    return (i, j), True
        
        return None, False
    
    def karisik_strateji_nash(
        self,
        kazanim_matrisi: np.ndarray,
        baslangic_stratejileri: Optional[List[np.ndarray]] = None
    ) -> Tuple[List[np.ndarray], bool, int]:
        """
        Karışık Strateji Nash Dengesi (Fictitious Play)
        
        Iteratif en iyi yanıt yöntemiyle karışık strateji Nash dengesi bulur.
        
        Args:
            kazanim_matrisi: Kazanım matrisi [N, M, 2]
            baslangic_stratejileri: Başlangıç olasılık dağılımları
            
        Returns:
            nash_stratejileri: [oyuncu1_dagilimi, oyuncu2_dagilimi]
            yakinsadi: Yakınsama sağlandı mı?
            iterasyon: Kullanılan iterasyon sayısı
        """
        N, M, _ = kazanim_matrisi.shape
        
        # Başlangıç stratejileri
        if baslangic_stratejileri is None:
            strateji_1 = np.ones(N) / N
            strateji_2 = np.ones(M) / M
        else:
            strateji_1, strateji_2 = baslangic_stratejileri
        
        for iterasyon in range(self.max_iterasyon):
            eski_strateji_1 = strateji_1.copy()
            eski_strateji_2 = strateji_2.copy()
            
            # Oyuncu 1'in en iyi yanıtı (strateji_2 sabit)
            beklenen_kazanimlar_1 = np.dot(kazanim_matrisi[:, :, 0], strateji_2)
            en_iyi_1 = np.argmax(beklenen_kazanimlar_1)
            yeni_strateji_1 = np.zeros(N)
            yeni_strateji_1[en_iyi_1] = 1.0
            
            # Oyuncu 2'nin en iyi yanıtı (strateji_1 sabit)
            beklenen_kazanimlar_2 = np.dot(kazanim_matrisi[:, :, 1].T, strateji_1)
            en_iyi_2 = np.argmax(beklenen_kazanimlar_2)
            yeni_strateji_2 = np.zeros(M)
            yeni_strateji_2[en_iyi_2] = 1.0
            
            # Fictitious play: ortalama stratejileri güncelle
            alpha = 1.0 / (iterasyon + 2)
            strateji_1 = (1 - alpha) * strateji_1 + alpha * yeni_strateji_1
            strateji_2 = (1 - alpha) * strateji_2 + alpha * yeni_strateji_2
            
            # Yakınsama kontrolü
            degisim_1 = np.max(np.abs(strateji_1 - eski_strateji_1))
            degisim_2 = np.max(np.abs(strateji_2 - eski_strateji_2))
            
            if degisim_1 < self.epsilon and degisim_2 < self.epsilon:
                self._gecmis.append({
                    "iterasyon": iterasyon + 1,
                    "yakinsadi": True,
                    "degisim": max(degisim_1, degisim_2)
                })
                return [strateji_1, strateji_2], True, iterasyon + 1
        
        self._gecmis.append({
            "iterasyon": self.max_iterasyon,
            "yakinsadi": False,
            "degisim": max(degisim_1, degisim_2)
        })
        return [strateji_1, strateji_2], False, self.max_iterasyon
    
    def cok_oyunculu_nash(
        self,
        oyuncu_sayisi: int,
        kazanim_fonksiyonlari: List[Callable],
        strateji_uzaylari: List[np.ndarray],
        baslangic_stratejileri: Optional[List[np.ndarray]] = None
    ) -> Tuple[List[np.ndarray], bool, int]:
        """
        Çok Oyunculu Nash Dengesi
        
        N-oyunculu oyunlar için iteratif en iyi yanıt yöntemi.
        
        Args:
            oyuncu_sayisi: Oyuncu sayısı
            kazanim_fonksiyonlari: Her oyuncu için kazanım fonksiyonu
            strateji_uzaylari: Her oyuncu için strateji uzayı
            baslangic_stratejileri: Başlangıç stratejileri
            
        Returns:
            nash_stratejileri: Her oyuncu için Nash stratejisi
            yakinsadi: Yakınsama sağlandı mı?
            iterasyon: Kullanılan iterasyon sayısı
        """
        # Başlangıç stratejileri
        if baslangic_stratejileri is None:
            stratejiler = [
                np.ones(len(uzay)) / len(uzay)
                for uzay in strateji_uzaylari
            ]
        else:
            stratejiler = [s.copy() for s in baslangic_stratejileri]
        
        for iterasyon in range(self.max_iterasyon):
            eski_stratejiler = [s.copy() for s in stratejiler]
            max_degisim = 0.0
            
            # Her oyuncu için en iyi yanıt
            for oyuncu_idx in range(oyuncu_sayisi):
                # Diğer oyuncuların stratejileri sabit
                diger_stratejiler = [
                    stratejiler[i] for i in range(oyuncu_sayisi)
                    if i != oyuncu_idx
                ]
                
                # Bu oyuncu için en iyi stratejiyi bul
                en_iyi_kazanim = -np.inf
                en_iyi_strateji_idx = 0
                
                strateji_uzayi = strateji_uzaylari[oyuncu_idx]
                for strateji_idx in range(len(strateji_uzayi)):
                    # Bu stratejiyi dene
                    test_stratejisi = np.zeros(len(strateji_uzayi))
                    test_stratejisi[strateji_idx] = 1.0
                    
                    # Kazanım hesapla
                    tum_stratejiler = (
                        diger_stratejiler[:oyuncu_idx] +
                        [test_stratejisi] +
                        diger_stratejiler[oyuncu_idx:]
                    )
                    kazanim = kazanim_fonksiyonlari[oyuncu_idx](*tum_stratejiler)
                    
                    if kazanim > en_iyi_kazanim:
                        en_iyi_kazanim = kazanim
                        en_iyi_strateji_idx = strateji_idx
                
                # En iyi yanıtı karışık stratejiye ekle (fictitious play)
                yeni_strateji = np.zeros(len(strateji_uzayi))
                yeni_strateji[en_iyi_strateji_idx] = 1.0
                
                alpha = 1.0 / (iterasyon + 2)
                stratejiler[oyuncu_idx] = (
                    (1 - alpha) * stratejiler[oyuncu_idx] + alpha * yeni_strateji
                )
                
                # Değişimi hesapla
                degisim = np.max(np.abs(stratejiler[oyuncu_idx] - eski_stratejiler[oyuncu_idx]))
                max_degisim = max(max_degisim, degisim)
            
            # Yakınsama kontrolü
            if max_degisim < self.epsilon:
                self._gecmis.append({
                    "iterasyon": iterasyon + 1,
                    "yakinsadi": True,
                    "degisim": max_degisim,
                    "oyuncu_sayisi": oyuncu_sayisi
                })
                return stratejiler, True, iterasyon + 1
        
        self._gecmis.append({
            "iterasyon": self.max_iterasyon,
            "yakinsadi": False,
            "degisim": max_degisim,
            "oyuncu_sayisi": oyuncu_sayisi
        })
        return stratejiler, False, self.max_iterasyon
    
    def pareto_optimum_bul(
        self,
        kazanim_profilleri: np.ndarray
    ) -> Tuple[List[int], np.ndarray]:
        """
        Pareto Optimum Çözümler
        
        Hiçbir oyuncunun kazancı azalmadan başka bir oyuncunun kazancı
        artırılamayan çözümleri bulur.
        
        Args:
            kazanim_profilleri: Kazanım profilleri [N_cozum, N_oyuncu]
            
        Returns:
            pareto_indeksler: Pareto optimal çözüm indeksleri
            pareto_cozumler: Pareto optimal kazanım profilleri
        """
        N_cozum, N_oyuncu = kazanim_profilleri.shape
        pareto_maskesi = np.ones(N_cozum, dtype=bool)
        
        for i in range(N_cozum):
            for j in range(N_cozum):
                if i == j:
                    continue
                
                # j, i'yi domine ediyor mu?
                # (j tüm oyuncular için >= i ve en az bir oyuncu için > i)
                dominate_ediyor = np.all(kazanim_profilleri[j] >= kazanim_profilleri[i])
                strictly_daha_iyi = np.any(kazanim_profilleri[j] > kazanim_profilleri[i])
                
                if dominate_ediyor and strictly_daha_iyi:
                    pareto_maskesi[i] = False
                    break
        
        pareto_indeksler = np.where(pareto_maskesi)[0].tolist()
        pareto_cozumler = kazanim_profilleri[pareto_maskesi]
        
        return pareto_indeksler, pareto_cozumler
    
    def kooperatif_cozum(
        self,
        kazanim_profilleri: np.ndarray,
        adalet_agirligi: float = 0.5
    ) -> Tuple[int, np.ndarray]:
        """
        Kooperatif Çözüm (Nash Bargaining Solution)
        
        Oyuncuların işbirliği yaparak ulaşabilecekleri en adil çözümü bulur.
        
        Args:
            kazanim_profilleri: Kazanım profilleri [N_cozum, N_oyuncu]
            adalet_agirligi: Adalet vs verimlilik trade-off [0, 1]
                           (0 = sadece toplam kazanım, 1 = sadece adalet)
            
        Returns:
            en_iyi_idx: En iyi kooperatif çözüm indeksi
            en_iyi_profil: En iyi kazanım profili
        """
        N_cozum, N_oyuncu = kazanim_profilleri.shape
        
        # Nash product: Her oyuncunun kazancının çarpımı (adalet)
        nash_products = np.prod(kazanim_profilleri, axis=1)
        
        # Toplam kazanç (verimlilik)
        toplam_kazanclar = np.sum(kazanim_profilleri, axis=1)
        
        # Normalize et
        nash_products_norm = nash_products / (np.max(nash_products) + 1e-8)
        toplam_kazanclar_norm = toplam_kazanclar / (np.max(toplam_kazanclar) + 1e-8)
        
        # Ağırlıklı kombinasyon
        skorlar = (
            adalet_agirligi * nash_products_norm +
            (1 - adalet_agirligi) * toplam_kazanclar_norm
        )
        
        en_iyi_idx = int(np.argmax(skorlar))
        en_iyi_profil = kazanim_profilleri[en_iyi_idx]
        
        return en_iyi_idx, en_iyi_profil
    
    def get_gecmis(self) -> List[Dict]:
        """Hesaplama geçmişini döndür"""
        return self._gecmis.copy()
    
    def sifirla(self):
        """Dahili durumu sıfırla"""
        self._gecmis.clear()

