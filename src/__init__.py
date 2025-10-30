"""
DilModeli - Nash-Sürü Teoremi ile LLM Optimizasyonu

Bu paket, Nash Dengesi ve Sığırcık Sürü Davranışı teorilerini
kullanarak LLM'leri CPU'da verimli çalıştırmak için araçlar sağlar.
"""

from .moe.nash_suru_router import NashSuruMoE
from .quantization.dinamik_kuantizasyon import DinamikKuantizasyon
from .core.nash_suru_teorem import NashSuruTeorem
from .optimization.cpu_optimizer import CPUOptimizer

__version__ = "0.1.0"
__all__ = [
    "NashSuruMoE",
    "DinamikKuantizasyon",
    "NashSuruTeorem",
    "CPUOptimizer"
]

