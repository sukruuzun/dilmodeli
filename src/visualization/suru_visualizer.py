"""
Sürü Davranışı Görselleştirme

Sığırcık sürü davranışı ve Nash dengesi görselleştirmeleri
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from typing import List, Optional, Tuple
import seaborn as sns


class SuruVisualizasyonu:
    """
    Sürü davranışı görselleştirme araçları
    """
    
    def __init__(self, stil: str = "darkgrid"):
        """
        Args:
            stil: Seaborn stili
        """
        sns.set_style(stil)
        self.renk_paleti = sns.color_palette("husl", 10)
        
    def suru_hareketini_ciz(
        self,
        pozisyon_gecmisi: List[np.ndarray],
        kaydet_yol: Optional[str] = None,
        baslik: str = "Sürü Hareketi"
    ):
        """
        Sürü hareketini 2D'de görselleştir
        
        Args:
            pozisyon_gecmisi: Her zaman adımında birimlerin pozisyonları [T, N, 2]
            kaydet_yol: Kaydetme yolu (opsiyonel)
            baslik: Grafik başlığı
        """
        if len(pozisyon_gecmisi) == 0:
            print("Görselleştirilecek veri yok!")
            return
        
        # İlk 2 boyutu al
        pozisyonlar = [p[:, :2] for p in pozisyon_gecmisi]
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Tüm yörüngeleri çiz
        N_birim = pozisyonlar[0].shape[0]
        
        for birim_idx in range(N_birim):
            yörunge = np.array([poz[birim_idx] for poz in pozisyonlar])
            ax.plot(
                yörunge[:, 0],
                yörunge[:, 1],
                alpha=0.3,
                linewidth=0.5,
                color=self.renk_paleti[birim_idx % len(self.renk_paleti)]
            )
        
        # Son pozisyonları işaretle
        son_pozisyonlar = pozisyonlar[-1]
        ax.scatter(
            son_pozisyonlar[:, 0],
            son_pozisyonlar[:, 1],
            s=100,
            c=range(N_birim),
            cmap='viridis',
            edgecolor='black',
            linewidth=1,
            alpha=0.8
        )
        
        ax.set_xlabel('X Pozisyonu', fontsize=12)
        ax.set_ylabel('Y Pozisyonu', fontsize=12)
        ax.set_title(baslik, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if kaydet_yol:
            plt.savefig(kaydet_yol, dpi=300, bbox_inches='tight')
            print(f"Grafik kaydedildi: {kaydet_yol}")
        
        plt.show()
        
    def komsu_agini_ciz(
        self,
        pozisyonlar: np.ndarray,
        komsu_listesi: List[List[int]],
        kaydet_yol: Optional[str] = None,
        baslik: str = "Komşu Ağı"
    ):
        """
        Birimlerin komşu ilişkilerini çiz
        
        Args:
            pozisyonlar: Birim pozisyonları [N, 2]
            komsu_listesi: Her birim için komşu indeksleri
            kaydet_yol: Kaydetme yolu
            baslik: Grafik başlığı
        """
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # İlk 2 boyutu al
        poz_2d = pozisyonlar[:, :2]
        N_birim = len(poz_2d)
        
        # Komşu bağlantılarını çiz
        for i, komşular in enumerate(komsu_listesi):
            for komsu_idx in komşular:
                ax.plot(
                    [poz_2d[i, 0], poz_2d[komsu_idx, 0]],
                    [poz_2d[i, 1], poz_2d[komsu_idx, 1]],
                    'gray',
                    alpha=0.2,
                    linewidth=0.5
                )
        
        # Birimleri çiz
        ax.scatter(
            poz_2d[:, 0],
            poz_2d[:, 1],
            s=200,
            c=range(N_birim),
            cmap='viridis',
            edgecolor='black',
            linewidth=2,
            alpha=0.8
        )
        
        # Birim numaralarını ekle
        for i, (x, y) in enumerate(poz_2d):
            ax.annotate(
                str(i),
                (x, y),
                fontsize=8,
                ha='center',
                va='center',
                color='white',
                fontweight='bold'
            )
        
        ax.set_xlabel('X Pozisyonu', fontsize=12)
        ax.set_ylabel('Y Pozisyonu', fontsize=12)
        ax.set_title(baslik, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if kaydet_yol:
            plt.savefig(kaydet_yol, dpi=300, bbox_inches='tight')
            print(f"Grafik kaydedildi: {kaydet_yol}")
        
        plt.show()
        
    def nash_dengesi_yakinsama_ciz(
        self,
        nash_gecmisi: List[dict],
        kaydet_yol: Optional[str] = None
    ):
        """
        Nash dengesi yakınsamasını görselleştir
        
        Args:
            nash_gecmisi: Nash dengesi arama geçmişi
            kaydet_yol: Kaydetme yolu
        """
        if not nash_gecmisi:
            print("Görselleştirilecek Nash geçmişi yok!")
            return
        
        iterasyonlar = [d['iterasyon'] for d in nash_gecmisi]
        degisimler = [d['degisim'] for d in nash_gecmisi]
        yakinsadi_mi = [d['yakinsadi'] for d in nash_gecmisi]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Değişim grafiği
        ax.plot(iterasyonlar, degisimler, marker='o', linewidth=2, markersize=6)
        ax.axhline(y=1e-4, color='r', linestyle='--', label='Yakınsama Eşiği', linewidth=2)
        
        # Yakınsanan noktaları işaretle
        for i, (it, deg, yak) in enumerate(zip(iterasyonlar, degisimler, yakinsadi_mi)):
            if yak:
                ax.scatter(it, deg, color='green', s=200, marker='*', zorder=5)
                ax.annotate(
                    'Yakınsadı!',
                    (it, deg),
                    xytext=(10, 10),
                    textcoords='offset points',
                    fontsize=10,
                    fontweight='bold',
                    color='green'
                )
        
        ax.set_xlabel('İterasyon', fontsize=12)
        ax.set_ylabel('Strateji Değişimi', fontsize=12)
        ax.set_title('Nash Dengesi Yakınsaması', fontsize=14, fontweight='bold')
        ax.set_yscale('log')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if kaydet_yol:
            plt.savefig(kaydet_yol, dpi=300, bbox_inches='tight')
            print(f"Grafik kaydedildi: {kaydet_yol}")
        
        plt.show()
        
    def uzman_yuk_dagilimi_ciz(
        self,
        uzman_yukleri: np.ndarray,
        kaydet_yol: Optional[str] = None,
        baslik: str = "Uzman İş Yükü Dağılımı"
    ):
        """
        Uzman iş yükü dağılımını görselleştir
        
        Args:
            uzman_yukleri: Her uzmanın iş yükü [N_uzman]
            kaydet_yol: Kaydetme yolu
            baslik: Grafik başlığı
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        N_uzman = len(uzman_yukleri)
        uzman_indeksleri = list(range(N_uzman))
        
        # Bar chart
        bars = ax1.bar(
            uzman_indeksleri,
            uzman_yukleri,
            color=self.renk_paleti[:N_uzman],
            edgecolor='black',
            linewidth=1.5
        )
        
        # Ortalama çizgisi
        ortalama = np.mean(uzman_yukleri)
        ax1.axhline(y=ortalama, color='red', linestyle='--', linewidth=2, label=f'Ortalama: {ortalama:.1f}')
        
        ax1.set_xlabel('Uzman ID', fontsize=12)
        ax1.set_ylabel('İş Yükü (Token Sayısı)', fontsize=12)
        ax1.set_title('Uzman İş Yükleri', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Pie chart
        ax2.pie(
            uzman_yukleri,
            labels=[f'Uzman {i}' for i in uzman_indeksleri],
            autopct='%1.1f%%',
            colors=self.renk_paleti[:N_uzman],
            startangle=90,
            textprops={'fontsize': 10}
        )
        ax2.set_title('İş Yükü Dağılımı', fontsize=14, fontweight='bold')
        
        plt.suptitle(baslik, fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if kaydet_yol:
            plt.savefig(kaydet_yol, dpi=300, bbox_inches='tight')
            print(f"Grafik kaydedildi: {kaydet_yol}")
        
        plt.show()
        
    def suru_metriklerini_ciz(
        self,
        metrik_gecmisi: List[dict],
        kaydet_yol: Optional[str] = None
    ):
        """
        Sürü metriklerinin zaman içindeki değişimini görselleştir
        
        Args:
            metrik_gecmisi: Sürü metriklerinin geçmişi
            kaydet_yol: Kaydetme yolu
        """
        if not metrik_gecmisi:
            print("Görselleştirilecek metrik yok!")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        zaman_adimi = list(range(len(metrik_gecmisi)))
        
        # Dağılım
        dagilimlar = [m['dagilim'] for m in metrik_gecmisi]
        axes[0, 0].plot(zaman_adimi, dagilimlar, linewidth=2, color='blue', marker='o')
        axes[0, 0].set_xlabel('Zaman Adımı', fontsize=12)
        axes[0, 0].set_ylabel('Dağılım (Varyans)', fontsize=12)
        axes[0, 0].set_title('Sürü Dağılımı', fontsize=14, fontweight='bold')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Ortalama hız
        hizlar = [m['ortalama_hiz'] for m in metrik_gecmisi]
        axes[0, 1].plot(zaman_adimi, hizlar, linewidth=2, color='green', marker='s')
        axes[0, 1].set_xlabel('Zaman Adımı', fontsize=12)
        axes[0, 1].set_ylabel('Ortalama Hız', fontsize=12)
        axes[0, 1].set_title('Sürü Hızı', fontsize=14, fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Hizalanma derecesi
        hizalanmalar = [m['hizalanma_derecesi'] for m in metrik_gecmisi]
        axes[1, 0].plot(zaman_adimi, hizalanmalar, linewidth=2, color='red', marker='^')
        axes[1, 0].set_xlabel('Zaman Adımı', fontsize=12)
        axes[1, 0].set_ylabel('Hizalanma Derecesi', fontsize=12)
        axes[1, 0].set_title('Sürü Hizalanması', fontsize=14, fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_ylim([0, 1.1])
        
        # Ortalama komşu sayısı
        komsu_sayilari = [m['ortalama_komsu_sayisi'] for m in metrik_gecmisi]
        axes[1, 1].plot(zaman_adimi, komsu_sayilari, linewidth=2, color='purple', marker='d')
        axes[1, 1].set_xlabel('Zaman Adımı', fontsize=12)
        axes[1, 1].set_ylabel('Ortalama Komşu Sayısı', fontsize=12)
        axes[1, 1].set_title('Sürü Yoğunluğu', fontsize=14, fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.suptitle('Sürü Davranış Metrikleri', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if kaydet_yol:
            plt.savefig(kaydet_yol, dpi=300, bbox_inches='tight')
            print(f"Grafik kaydedildi: {kaydet_yol}")
        
        plt.show()

