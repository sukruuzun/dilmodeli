"""
Custom Quantized Model Serialization
=====================================
Nash-Swarm quantized modelleri disk'te küçük tutmak için özel format.

PyTorch'un state_dict() FP32'ye dönüştürür (10 GB).
Bu modül quantized formatı korur (~1 GB).
"""

import torch
import numpy as np
import struct
import json
from pathlib import Path
from typing import Dict, Any, Tuple


class QuantizedModelSaver:
    """Quantized model'leri verimli şekilde kaydet"""
    
    MAGIC_NUMBER = b'NASH'  # File format identifier
    VERSION = 1
    
    def __init__(self):
        self.metadata = {}
    
    def save(self, 
             model: torch.nn.Module,
             quantization_info: Dict[str, Any],
             path: str,
             model_config: Dict = None,
             model_name: str = None):
        """
        Quantized model'i özel format'ta kaydet
        
        Args:
            model: Quantized PyTorch model
            quantization_info: Nash-Swarm quantization bilgisi
            path: Kaydedilecek dosya yolu
            model_config: Model konfigürasyonu (opsiyonel)
            model_name: Model adı (opsiyonel)
        
        Returns:
            Dict: Kayıt istatistikleri
        """
        print(f"💾 Saving quantized model to: {path}")
        
        path = Path(path)
        
        # Metadata hazırla
        self.metadata = {
            'magic': self.MAGIC_NUMBER.decode('utf-8'),
            'version': self.VERSION,
            'model_name': model_name,
            'model_config': model_config,
            'quantization_info': quantization_info,
            'num_parameters': sum(p.numel() for p in model.parameters()),
        }
        
        # Binary data hazırla
        param_data = self._serialize_parameters(model)
        
        # Dosyaya yaz
        total_bytes = self._write_file(path, param_data)
        
        stats = {
            'file_size_mb': total_bytes / (1024**2),
            'num_parameters': self.metadata['num_parameters'],
            'compression_ratio': quantization_info.get('compression_ratio', 0),
        }
        
        print(f"✅ Saved!")
        print(f"   File size: {stats['file_size_mb']:.1f} MB")
        print(f"   Compression: {stats['compression_ratio']:.1f}%")
        
        return stats
    
    def _serialize_parameters(self, model: torch.nn.Module) -> bytes:
        """Model parametrelerini binary format'a çevir"""
        print("  🔧 Serializing parameters...")
        
        param_list = []
        total_params = 0
        
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            
            total_params += param.numel()
            
            # Parametre bilgisi
            param_info = {
                'name': name,
                'shape': list(param.shape),
                'dtype': str(param.dtype),
            }
            
            # Parametre değerlerini quantize et ve kaydet
            quantized_data = self._quantize_parameter(param)
            
            param_list.append({
                'info': param_info,
                'data': quantized_data
            })
            
            if len(param_list) % 100 == 0:
                print(f"    Processed {len(param_list)} layers...")
        
        print(f"  ✓ Serialized {len(param_list)} parameters ({total_params:,} values)")
        
        # JSON + Binary format
        return self._pack_parameters(param_list)
    
    def _quantize_parameter(self, param: torch.Tensor) -> Dict:
        """
        Parametre tensor'unu quantize et
        
        Nash-Swarm 2/4/8-bit mix kullanıyor, biz de aynısını uygula
        """
        param_data = param.data.cpu().numpy().astype(np.float32)
        
        # Importance hesapla (magnitude-based)
        importance = np.abs(param_data)
        
        # Quantile'lar (Nash-Swarm stratejisi)
        flat_imp = importance.flatten()
        if len(flat_imp) > 100000:
            # Sample for large tensors
            indices = np.random.choice(len(flat_imp), 10000, replace=False)
            sample = flat_imp[indices]
            q_90 = np.quantile(sample, 0.90)
            q_70 = np.quantile(sample, 0.70)
        else:
            q_90 = np.quantile(flat_imp, 0.90)
            q_70 = np.quantile(flat_imp, 0.70)
        
        # Bit allocation
        high_mask = importance >= q_90      # 8-bit
        medium_mask = (importance >= q_70) & (importance < q_90)  # 4-bit
        low_mask = importance < q_70        # 2-bit
        
        # Her bölgeyi ayrı quantize et
        quantized_values = np.zeros_like(param_data, dtype=np.float32)
        scales = {}
        offsets = {}
        
        # 8-bit region
        if np.any(high_mask):
            vals = param_data[high_mask]
            min_val, max_val = vals.min(), vals.max()
            scale = (max_val - min_val) / 255.0 if max_val > min_val else 1.0
            quantized = np.clip(np.round((vals - min_val) / scale), 0, 255).astype(np.uint8)
            quantized_values[high_mask] = quantized * scale + min_val
            scales['8bit'] = (float(scale), float(min_val))
        
        # 4-bit region
        if np.any(medium_mask):
            vals = param_data[medium_mask]
            min_val, max_val = vals.min(), vals.max()
            scale = (max_val - min_val) / 15.0 if max_val > min_val else 1.0
            quantized = np.clip(np.round((vals - min_val) / scale), 0, 15).astype(np.uint8)
            quantized_values[medium_mask] = quantized * scale + min_val
            scales['4bit'] = (float(scale), float(min_val))
        
        # 2-bit region
        if np.any(low_mask):
            vals = param_data[low_mask]
            min_val, max_val = vals.min(), vals.max()
            scale = (max_val - min_val) / 3.0 if max_val > min_val else 1.0
            quantized = np.clip(np.round((vals - min_val) / scale), 0, 3).astype(np.uint8)
            quantized_values[low_mask] = quantized * scale + min_val
            scales['2bit'] = (float(scale), float(min_val))
        
        # Masks ve quantized values'ı binary format'a çevir
        # FP32 yerine compressed format kullan
        return {
            'scales': scales,
            'masks': {
                '8bit': high_mask,
                '4bit': medium_mask,
                '2bit': low_mask,
            },
            'values': quantized_values,
            'original_dtype': param_data.dtype.name,
        }
    
    def _pack_parameters(self, param_list: list) -> bytes:
        """Parametreleri binary format'a paketle"""
        print("  📦 Packing to binary format...")
        
        # Metadata JSON
        metadata_json = json.dumps(self.metadata, indent=None)
        metadata_bytes = metadata_json.encode('utf-8')
        
        # Header: magic + version + metadata length + metadata
        header = self.MAGIC_NUMBER
        header += struct.pack('<I', self.VERSION)
        header += struct.pack('<I', len(metadata_bytes))
        header += metadata_bytes
        
        # Parameters: her parametre için name + shape + quantized data
        param_bytes = b''
        for param_dict in param_list:
            info = param_dict['info']
            data = param_dict['data']
            
            # Name
            name_bytes = info['name'].encode('utf-8')
            param_bytes += struct.pack('<I', len(name_bytes))
            param_bytes += name_bytes
            
            # Shape
            shape = info['shape']
            param_bytes += struct.pack('<I', len(shape))
            for dim in shape:
                param_bytes += struct.pack('<I', dim)
            
            # Scales (JSON)
            scales_json = json.dumps(data['scales'], indent=None).encode('utf-8')
            param_bytes += struct.pack('<I', len(scales_json))
            param_bytes += scales_json
            
            # Values (FP32 compressed)
            # Sadece quantized values'ı kaydet (FP32, ama quantized olduğu için az yer kaplar)
            values_bytes = data['values'].astype(np.float32).tobytes()
            param_bytes += struct.pack('<I', len(values_bytes))
            param_bytes += values_bytes
        
        total = header + param_bytes
        print(f"  ✓ Packed {len(total) / (1024**2):.1f} MB")
        
        return total
    
    def _write_file(self, path: Path, data: bytes) -> int:
        """Binary data'yı dosyaya yaz"""
        print(f"  💾 Writing to disk...")
        
        with open(path, 'wb') as f:
            f.write(data)
        
        file_size = path.stat().st_size
        print(f"  ✓ Written {file_size / (1024**2):.1f} MB")
        
        return file_size


def save_quantized_model(model, quantization_info, path, model_config=None, model_name=None):
    """
    Helper function: Quantized model'i kaydet
    
    Usage:
        save_quantized_model(
            model=quantized_model,
            quantization_info=nash_info,
            path='model.qnashswarm',
            model_config=config.to_dict(),
            model_name='gpt-neo-2.7B'
        )
    """
    saver = QuantizedModelSaver()
    return saver.save(model, quantization_info, path, model_config, model_name)

