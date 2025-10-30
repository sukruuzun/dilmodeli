"""
Metrik Görselleştirme

Model performans metrikleri ve kayıp fonksiyonu görselleştirmeleri
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Optional, Any


class MetrikVisualizasyonu:
    """
    Performans metrikleri görselleştirme araçları
    """
    
    def __init__(self, stil: str = "darkgrid"):
        """
        Args:
            stil: Seaborn stili
        """
        sns.set_style(stil)
        self.renk_paleti = sns.color_palette("Set2", 8)
        
    def kayip_egrileri_ciz(
        self,
        kayip_gecmisi: Dict[str, List[float]],
        kaydet_yol: Optional[str] = None,
        baslik: str = "Kayıp Fonksiyonu Bileşenleri"
    ):
        """
        Kayıp fonksiyonu bileşenlerini görselleştir
        
        Args:
            kayip_gecmisi: Kayıp bileşenleri geçmişi
            kaydet_yol: Kaydetme yolu
            baslik: Grafik başlığı
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()
        
        bilesen_isimleri = {
            'temel': 'Temel Kayıp',
            'dengeleme': 'Dengeleme Kaybı',
            'cpu_onbellek': 'CPU Önbellek Ödülü',
            'suru_uyumu': 'Sürü Uyum Ödülü',
            'nash_dengesi': 'Nash Dengesi Regularizasyonu',
            'toplam': 'Toplam Kayıp'
        }
        
        for idx, (bilesen, isim) in enumerate(bilesen_isimleri.items()):
            if bilesen in kayip_gecmisi and kayip_gecmisi[bilesen]:
                degerler = kayip_gecmisi[bilesen]
                adimlar = list(range(len(degerler)))
                
                axes[idx].plot(
                    adimlar,
                    degerler,
                    linewidth=2,
                    color=self.renk_paleti[idx],
                    alpha=0.8
                )
                
                # Hareketli ortalama
                if len(degerler) > 10:
                    pencere = min(50, len(degerler) // 10)
                    hareketli_ort = np.convolve(
                        degerler,
                        np.ones(pencere) / pencere,
                        mode='valid'
                    )
                    axes[idx].plot(
                        range(pencere - 1, len(degerler)),
                        hareketli_ort,
                        linewidth=2,
                        color='red',
                        linestyle='--',
                        alpha=0.7,
                        label='Hareketli Ort.'
                    )
                    axes[idx].legend()
                
                axes[idx].set_xlabel('Adım', fontsize=10)
                axes[idx].set_ylabel('Değer', fontsize=10)
                axes[idx].set_title(isim, fontsize=12, fontweight='bold')
                axes[idx].grid(True, alpha=0.3)
        
        plt.suptitle(baslik, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if kaydet_yol:
            plt.savefig(kaydet_yol, dpi=300, bbox_inches='tight')
            print(f"Grafik kaydedildi: {kaydet_yol}")
        
        plt.show()
        
    def performans_karsilastirmasi_ciz(
        self,
        model_isimleri: List[str],
        metrikler: Dict[str, List[float]],
        kaydet_yol: Optional[str] = None,
        baslik: str = "Model Performans Karşılaştırması"
    ):
        """
        Farklı modellerin performanslarını karşılaştır
        
        Args:
            model_isimleri: Model isimleri
            metrikler: Metrik değerleri (metrik_adi -> [değerler])
            kaydet_yol: Kaydetme yolu
            baslik: Grafik başlığı
        """
        N_model = len(model_isimleri)
        N_metrik = len(metrikler)
        
        fig, axes = plt.subplots(1, N_metrik, figsize=(6 * N_metrik, 6))
        
        if N_metrik == 1:
            axes = [axes]
        
        for idx, (metrik_adi, degerler) in enumerate(metrikler.items()):
            x = np.arange(N_model)
            bars = axes[idx].bar(
                x,
                degerler,
                color=self.renk_paleti[:N_model],
                edgecolor='black',
                linewidth=1.5
            )
            
            # Değerleri bar'ların üstüne yaz
            for bar, deger in zip(bars, degerler):
                yukseklik = bar.get_height()
                axes[idx].text(
                    bar.get_x() + bar.get_width() / 2.,
                    yukseklik,
                    f'{deger:.2f}',
                    ha='center',
                    va='bottom',
                    fontsize=10,
                    fontweight='bold'
                )
            
            axes[idx].set_xticks(x)
            axes[idx].set_xticklabels(model_isimleri, rotation=45, ha='right')
            axes[idx].set_ylabel('Değer', fontsize=12)
            axes[idx].set_title(metrik_adi, fontsize=14, fontweight='bold')
            axes[idx].grid(True, alpha=0.3, axis='y')
        
        plt.suptitle(baslik, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if kaydet_yol:
            plt.savefig(kaydet_yol, dpi=300, bbox_inches='tight')
            print(f"Grafik kaydedildi: {kaydet_yol}")
        
        plt.show()
        
    def bellek_kullanimi_ciz(
        self,
        katman_isimleri: List[str],
        orijinal_bellek: List[float],
        optimize_bellek: List[float],
        kaydet_yol: Optional[str] = None
    ):
        """
        Katman bazında bellek kullanımını görselleştir
        
        Args:
            katman_isimleri: Katman isimleri
            orijinal_bellek: Orijinal bellek kullanımı (MB)
            optimize_bellek: Optimize edilmiş bellek kullanımı (MB)
            kaydet_yol: Kaydetme yolu
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        x = np.arange(len(katman_isimleri))
        genislik = 0.35
        
        # Grouped bar chart
        bars1 = ax1.bar(
            x - genislik / 2,
            orijinal_bellek,
            genislik,
            label='Orijinal',
            color='lightcoral',
            edgecolor='black',
            linewidth=1.5
        )
        bars2 = ax1.bar(
            x + genislik / 2,
            optimize_bellek,
            genislik,
            label='Optimize',
            color='lightgreen',
            edgecolor='black',
            linewidth=1.5
        )
        
        ax1.set_xlabel('Katman', fontsize=12)
        ax1.set_ylabel('Bellek Kullanımı (MB)', fontsize=12)
        ax1.set_title('Katman Bazında Bellek Kullanımı', fontsize=14, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(katman_isimleri, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Tasarruf yüzdesi
        tasarruf_yuzdesi = [(o - n) / o * 100 if o > 0 else 0 for o, n in zip(orijinal_bellek, optimize_bellek)]
        
        bars = ax2.barh(
            katman_isimleri,
            tasarruf_yuzdesi,
            color=self.renk_paleti[:len(katman_isimleri)],
            edgecolor='black',
            linewidth=1.5
        )
        
        # Değerleri bar'ların yanına yaz
        for bar, tasarruf in zip(bars, tasarruf_yuzdesi):
            genislik = bar.get_width()
            ax2.text(
                genislik,
                bar.get_y() + bar.get_height() / 2.,
                f'{tasarruf:.1f}%',
                ha='left',
                va='center',
                fontsize=10,
                fontweight='bold'
            )
        
        ax2.set_xlabel('Bellek Tasarrufu (%)', fontsize=12)
        ax2.set_title('Katman Bazında Tasarruf', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='x')
        
        plt.suptitle('Bellek Optimizasyon Analizi', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if kaydet_yol:
            plt.savefig(kaydet_yol, dpi=300, bbox_inches='tight')
            print(f"Grafik kaydedildi: {kaydet_yol}")
        
        plt.show()
        
    def latency_breakdown_ciz(
        self,
        bilesen_isimleri: List[str],
        sureler_ms: List[float],
        kaydet_yol: Optional[str] = None
    ):
        """
        Latency breakdown görselleştir
        
        Args:
            bilesen_isimleri: Bileşen isimleri
            sureler_ms: Süre (ms)
            kaydet_yol: Kaydetme yolu
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Bar chart
        bars = ax1.bar(
            bilesen_isimleri,
            sureler_ms,
            color=self.renk_paleti[:len(bilesen_isimleri)],
            edgecolor='black',
            linewidth=1.5
        )
        
        # Değerleri bar'ların üstüne yaz
        for bar, sure in zip(bars, sureler_ms):
            yukseklik = bar.get_height()
            ax1.text(
                bar.get_x() + bar.get_width() / 2.,
                yukseklik,
                f'{sure:.2f}ms',
                ha='center',
                va='bottom',
                fontsize=10,
                fontweight='bold'
            )
        
        ax1.set_ylabel('Süre (ms)', fontsize=12)
        ax1.set_title('Bileşen Süreleri', fontsize=14, fontweight='bold')
        ax1.set_xticklabels(bilesen_isimleri, rotation=45, ha='right')
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Pie chart
        ax2.pie(
            sureler_ms,
            labels=bilesen_isimleri,
            autopct='%1.1f%%',
            colors=self.renk_paleti[:len(bilesen_isimleri)],
            startangle=90,
            textprops={'fontsize': 10}
        )
        ax2.set_title('Süre Dağılımı', fontsize=14, fontweight='bold')
        
        toplam_sure = sum(sureler_ms)
        plt.suptitle(f'Latency Breakdown (Toplam: {toplam_sure:.2f}ms)', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if kaydet_yol:
            plt.savefig(kaydet_yol, dpi=300, bbox_inches='tight')
            print(f"Grafik kaydedildi: {kaydet_yol}")
        
        plt.show()

