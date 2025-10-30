"""
Nash-Sürü MoE Router

Bu modül, Nash Dengesi ve Sürü Davranışını kullanarak
MoE (Mixture of Experts) routing yapar.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
import psutil
import time

from ..core.nash_suru_teorem import NashSuruTeorem, NashSuruParametreleri


@dataclass
class UzmanIstatistikleri:
    """Her uzman için performans istatistikleri"""
    
    uzman_id: int
    toplam_token_sayisi: int = 0
    ortalama_kayip: float = 0.0
    cpu_onbellek_vurusu: float = 0.0
    son_kullanim_zamani: float = 0.0
    aktif_mi: bool = True


class UzmanKapisi(nn.Module):
    """
    Uzman Kapısı (Router/Gate Network)
    
    Token'ları uzmanlara yönlendiren ağ.
    Geleneksel softmax routing yerine Nash-Sürü algoritması kullanır.
    """
    
    def __init__(
        self,
        giris_boyutu: int,
        uzman_sayisi: int,
        lokal_grup_boyutu: int = 7,
        top_k: int = 2
    ):
        """
        Args:
            giris_boyutu: Token embedding boyutu
            uzman_sayisi: Toplam uzman sayısı
            lokal_grup_boyutu: Nash-Sürü için lokal grup boyutu
            top_k: Seçilecek uzman sayısı
        """
        super().__init__()
        
        self.giris_boyutu = giris_boyutu
        self.uzman_sayisi = uzman_sayisi
        self.lokal_grup_boyutu = min(lokal_grup_boyutu, uzman_sayisi)
        self.top_k = top_k
        
        # Uzman temsil vektörleri (her uzman bir embedding'e sahip)
        self.uzman_embeddings = nn.Parameter(
            torch.randn(uzman_sayisi, giris_boyutu) * 0.01
        )
        
        # İş yükü dengeleme için load tracking
        self.register_buffer(
            'uzman_yukleri',
            torch.zeros(uzman_sayisi)
        )
        
        # CPU önbellek durumu
        self.register_buffer(
            'onbellek_durumu',
            torch.ones(uzman_sayisi)  # 1 = önbellekte, 0 = değil
        )
        
    def forward(
        self,
        token_embeddings: torch.Tensor,
        nash_suru_teorem: NashSuruTeorem,
        egitim_modu: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        Token'lar için uzman seçimi yap
        
        Args:
            token_embeddings: Token embeddings [batch_size, seq_len, hidden_dim]
            nash_suru_teorem: Nash-Sürü teoremi motoru
            egitim_modu: Eğitim modunda mı?
            
        Returns:
            uzman_indeksleri: Seçilen uzmanlar [batch_size, seq_len, top_k]
            uzman_agirliklari: Uzman ağırlıkları [batch_size, seq_len, top_k]
            routing_bilgisi: Routing hakkında metrikler
        """
        batch_size, seq_len, hidden_dim = token_embeddings.shape
        
        # Token'ları düzleştir
        tokens_flat = token_embeddings.reshape(-1, hidden_dim)  # [B*S, H]
        N_token = tokens_flat.shape[0]
        
        # Uzman seçimi (her token için)
        secilen_uzmanlar = []
        secilen_agirliklar = []
        routing_metrikleri = {
            'suru_uyum_skorlari': [],
            'nash_yakinsama_oranlari': [],
            'cpu_onbellek_kullanimlari': []
        }
        
        for token_idx in range(N_token):
            token_emb = tokens_flat[token_idx].detach().cpu().numpy()
            
            # Uzman embeddings'lerini al
            uzman_embs = self.uzman_embeddings.detach().cpu().numpy()
            
            # Nash-Sürü algoritması ile karar ver
            secilen_idx, karar_bilgisi = nash_suru_teorem.adaptif_karar_ver(
                aday_birimleri=uzman_embs,
                hedef_gorev=token_emb,
                performans_gecmisi=self.uzman_yukleri.cpu().numpy(),
                kaynak_kisiti=None
            )
            
            # Lokal grup içinde top-k seçimi
            lokal_indeksler = karar_bilgisi.get('lokal_grup_indeksleri', [secilen_idx])
            if len(lokal_indeksler) < self.lokal_grup_boyutu:
                # Lokal grup bilgisi yoksa, benzerliğe göre seç
                benzerlikler = np.dot(uzman_embs, token_emb)
                lokal_indeksler = np.argsort(benzerlikler)[-self.lokal_grup_boyutu:].tolist()
            
            # Top-k uzman seç
            lokal_grup_embs = uzman_embs[lokal_indeksler]
            token_norm = token_emb / (np.linalg.norm(token_emb) + 1e-8)
            lokal_norm = lokal_grup_embs / (np.linalg.norm(lokal_grup_embs, axis=1, keepdims=True) + 1e-8)
            lokal_skorlar = np.dot(lokal_norm, token_norm)
            
            # CPU önbellek bonusu ekle
            onbellek_bonuslari = self.onbellek_durumu[lokal_indeksler].cpu().numpy()
            toplam_skorlar = lokal_skorlar + 0.1 * onbellek_bonuslari
            
            top_k_lokal_idx = np.argsort(toplam_skorlar)[-self.top_k:]
            top_k_global_idx = [lokal_indeksler[i] for i in top_k_lokal_idx]
            top_k_skorlar = toplam_skorlar[top_k_lokal_idx]
            
            # Softmax ağırlıklar
            agirliklar = np.exp(top_k_skorlar - np.max(top_k_skorlar))
            agirliklar = agirliklar / np.sum(agirliklar)
            
            secilen_uzmanlar.append(top_k_global_idx)
            secilen_agirliklar.append(agirliklar)
            
            # Metrikleri kaydet
            routing_metrikleri['suru_uyum_skorlari'].append(
                karar_bilgisi.get('suru_uyum_skoru', 0.0)
            )
            routing_metrikleri['nash_yakinsama_oranlari'].append(
                1.0 if karar_bilgisi.get('nash_yakinsadi', False) else 0.0
            )
            routing_metrikleri['cpu_onbellek_kullanimlari'].append(
                float(np.mean(onbellek_bonuslari))
            )
        
        # Tensor'lara dönüştür
        uzman_indeksleri = torch.tensor(secilen_uzmanlar, dtype=torch.long, device=token_embeddings.device)
        uzman_agirliklari = torch.tensor(secilen_agirliklar, dtype=torch.float32, device=token_embeddings.device)
        
        # Reshape
        uzman_indeksleri = uzman_indeksleri.reshape(batch_size, seq_len, self.top_k)
        uzman_agirliklari = uzman_agirliklari.reshape(batch_size, seq_len, self.top_k)
        
        # İş yüklerini güncelle (eğitim modunda)
        if egitim_modu:
            for idx in uzman_indeksleri.flatten():
                self.uzman_yukleri[idx] += 1
        
        # Toplam metrikler
        routing_bilgisi = {
            'ortalama_suru_uyumu': np.mean(routing_metrikleri['suru_uyum_skorlari']),
            'nash_yakinsama_orani': np.mean(routing_metrikleri['nash_yakinsama_oranlari']),
            'cpu_onbellek_kullanimi': np.mean(routing_metrikleri['cpu_onbellek_kullanimlari']),
            'uzman_yuk_varyansi': float(torch.var(self.uzman_yukleri).item())
        }
        
        return uzman_indeksleri, uzman_agirliklari, routing_bilgisi
    
    def yuk_dengeleme_kaybi_hesapla(self) -> torch.Tensor:
        """
        İş yükü dengeleme kaybını hesapla
        
        Uzmanlar arası iş yükü dağılımının dengesizliğini cezalandırır.
        """
        hedef_yuk = torch.mean(self.uzman_yukleri)
        dengeleme_kaybi = torch.var(self.uzman_yukleri) / (hedef_yuk + 1e-8)
        return dengeleme_kaybi
    
    def cpu_onbellek_durumunu_guncelle(self, bellek_mb: float = 1024.0):
        """
        CPU önbellek durumunu güncelle
        
        Args:
            bellek_mb: Kullanılabilir önbellek (MB)
        """
        # Basit LRU benzeri: En az kullanılan uzmanlar önbellekten çıkar
        toplam_yuk = torch.sum(self.uzman_yukleri)
        if toplam_yuk > 0:
            normalized_yukler = self.uzman_yukleri / toplam_yuk
        else:
            normalized_yukler = torch.ones_like(self.uzman_yukleri) / self.uzman_sayisi
        
        # Uzman başına ~50MB varsayımı
        uzman_basina_mb = 50.0
        onbellekte_tutulabilecek = int(bellek_mb / uzman_basina_mb)
        onbellekte_tutulabilecek = min(onbellekte_tutulabilecek, self.uzman_sayisi)
        
        # En çok kullanılan uzmanları önbellekte tut
        _, top_indeksler = torch.topk(self.uzman_yukleri, k=onbellekte_tutulabilecek)
        self.onbellek_durumu.zero_()
        self.onbellek_durumu[top_indeksler] = 1.0
    
    def yukleri_sifirla(self):
        """İş yüklerini sıfırla (epoch başında)"""
        self.uzman_yukleri.zero_()


class NashSuruMoE(nn.Module):
    """
    Nash-Sürü MoE (Mixture of Experts)
    
    Bu sınıf, Nash-Sürü teoremini kullanarak MoE routing yapar.
    Geleneksel MoE'den farkları:
    1. Lokal grup seçimi (Sürü davranışı)
    2. Nash dengesi ile karar verme
    3. CPU önbellek optimizasyonu
    4. Dinamik iş yükü dengeleme
    """
    
    def __init__(
        self,
        giris_boyutu: int,
        uzman_sayisi: int = 8,
        uzman_gizli_boyutu: int = 2048,
        lokal_grup_boyutu: int = 7,
        top_k: int = 2,
        nash_suru_params: Optional[NashSuruParametreleri] = None
    ):
        """
        Args:
            giris_boyutu: Input embedding boyutu
            uzman_sayisi: Toplam uzman sayısı
            uzman_gizli_boyutu: Her uzmanın gizli katman boyutu
            lokal_grup_boyutu: Nash-Sürü için lokal grup boyutu
            top_k: Token başına seçilecek uzman sayısı
            nash_suru_params: Nash-Sürü teoremi parametreleri
        """
        super().__init__()
        
        self.giris_boyutu = giris_boyutu
        self.uzman_sayisi = uzman_sayisi
        self.uzman_gizli_boyutu = uzman_gizli_boyutu
        self.top_k = top_k
        
        # Nash-Sürü teoremi motoru
        self.nash_suru_teorem = NashSuruTeorem(nash_suru_params)
        
        # Uzman kapısı (router)
        self.kapi = UzmanKapisi(
            giris_boyutu, uzman_sayisi, lokal_grup_boyutu, top_k
        )
        
        # Uzmanlar (basit FFN)
        self.uzmanlar = nn.ModuleList([
            nn.Sequential(
                nn.Linear(giris_boyutu, uzman_gizli_boyutu),
                nn.GELU(),
                nn.Linear(uzman_gizli_boyutu, giris_boyutu)
            )
            for _ in range(uzman_sayisi)
        ])
        
        # İstatistikler
        self.uzman_istatistikleri = [
            UzmanIstatistikleri(uzman_id=i)
            for i in range(uzman_sayisi)
        ]
        
    def forward(
        self,
        x: torch.Tensor,
        return_routing_info: bool = False
    ) -> Tuple[torch.Tensor, Optional[Dict[str, Any]]]:
        """
        MoE forward pass
        
        Args:
            x: Input tensor [batch_size, seq_len, hidden_dim]
            return_routing_info: Routing bilgilerini döndür mü?
            
        Returns:
            output: MoE çıktısı [batch_size, seq_len, hidden_dim]
            routing_info: Routing bilgileri (opsiyonel)
        """
        batch_size, seq_len, hidden_dim = x.shape
        
        # Uzman seçimi (Nash-Sürü routing)
        uzman_indeksleri, uzman_agirliklari, routing_bilgisi = self.kapi(
            x, self.nash_suru_teorem, egitim_modu=self.training
        )
        
        # Her token için seçilen uzmanları çalıştır
        output = torch.zeros_like(x)
        
        for b in range(batch_size):
            for s in range(seq_len):
                token_emb = x[b, s]  # [hidden_dim]
                
                # Bu token için seçilen uzmanlar
                secilen_uzmanlar = uzman_indeksleri[b, s]  # [top_k]
                agirliklar = uzman_agirliklari[b, s]  # [top_k]
                
                # Her uzmanın çıktısını ağırlıklı topla
                token_output = torch.zeros_like(token_emb)
                for k in range(self.top_k):
                    uzman_idx = secilen_uzmanlar[k].item()
                    agirlik = agirliklar[k]
                    
                    # Uzmanı çalıştır
                    uzman_cikti = self.uzmanlar[uzman_idx](token_emb.unsqueeze(0))
                    token_output += agirlik * uzman_cikti.squeeze(0)
                    
                    # İstatistikleri güncelle
                    if self.training:
                        self.uzman_istatistikleri[uzman_idx].toplam_token_sayisi += 1
                        self.uzman_istatistikleri[uzman_idx].son_kullanim_zamani = time.time()
                
                output[b, s] = token_output
        
        if return_routing_info:
            # Uzman istatistiklerini ekle
            routing_bilgisi['uzman_kullanim_dagilimi'] = [
                stat.toplam_token_sayisi for stat in self.uzman_istatistikleri
            ]
            return output, routing_bilgisi
        
        return output, None
    
    def dengeleyici_kayip_hesapla(
        self,
        capraz_entropi_kaybı: float,
        routing_bilgisi: Dict[str, Any]
    ) -> Tuple[float, Dict[str, float]]:
        """
        Nash-Sürü Dengeleyici Kayıp Fonksiyonu
        
        L_Nash-Sürü = L_ÇaprazEntropi + λ₁ · L_Dengeleme - λ₂ · R_CPU-Önbellek
        """
        # İş yükü dağılımı
        is_yuku_dagilimi = np.array([
            stat.toplam_token_sayisi for stat in self.uzman_istatistikleri
        ], dtype=np.float32)
        
        # CPU önbellek vuruş oranı
        cpu_onbellek_vurusu = routing_bilgisi.get('cpu_onbellek_kullanimi', 0.5)
        
        # Nash-Sürü teoremi ile kayıp hesapla
        toplam_kayip, kayip_bilesenleri = self.nash_suru_teorem.dengeleyici_kayip_hesapla(
            capraz_entropi_kaybı,
            is_yuku_dagilimi,
            cpu_onbellek_vurusu
        )
        
        return toplam_kayip, kayip_bilesenleri
    
    def cpu_onbellek_optimizasyonu(self, onbellek_mb: float = 1024.0):
        """CPU önbellek durumunu optimize et"""
        self.kapi.cpu_onbellek_durumunu_guncelle(onbellek_mb)
    
    def get_uzman_istatistikleri(self) -> List[UzmanIstatistikleri]:
        """Uzman istatistiklerini döndür"""
        return self.uzman_istatistikleri.copy()
    
    def istatistikleri_sifirla(self):
        """İstatistikleri sıfırla"""
        self.uzman_istatistikleri = [
            UzmanIstatistikleri(uzman_id=i)
            for i in range(self.uzman_sayisi)
        ]
        self.kapi.yukleri_sifirla()
        self.nash_suru_teorem.sifirla()

