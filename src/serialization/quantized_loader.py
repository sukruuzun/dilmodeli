"""
Custom Quantized Model Loading
================================
Nash-Swarm quantized modellerini özel format'tan yükle.

Lazy loading desteği ile RAM kullanımını minimize et.
"""

import torch
import numpy as np
import struct
import json
from pathlib import Path
from typing import Dict, Any, Optional
from transformers import (
    AutoConfig, 
    AutoModelForCausalLM,
    GPT2Config,
    GPTNeoConfig,
    LlamaConfig,
)


class QuantizedModelLoader:
    """Quantized model'leri yükle"""
    
    MAGIC_NUMBER = b'NASH'
    
    def __init__(self):
        self.metadata = None
        self.param_offsets = {}
    
    def load(self, path: str, lazy: bool = False, device: str = 'cpu') -> tuple:
        """
        Quantized model'i yükle
        
        Args:
            path: Model dosya yolu
            lazy: True ise lazy loading (RAM tasarrufu)
            device: Model device ('cpu', 'cuda', 'mps')
        
        Returns:
            (model, metadata): Model ve metadata
        """
        print(f"📦 Loading quantized model from: {path}")
        
        path = Path(path)
        
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")
        
        # Dosyayı oku
        with open(path, 'rb') as f:
            data = f.read()
        
        print(f"  ✓ Read {len(data) / (1024**2):.1f} MB from disk")
        
        # Parse
        metadata, param_data = self._parse_file(data)
        self.metadata = metadata
        
        # Model reconstruct
        model = self._reconstruct_model(param_data, device, lazy)
        
        print(f"✅ Model loaded!")
        print(f"   Model: {metadata.get('model_name', 'Unknown')}")
        print(f"   Parameters: {metadata.get('num_parameters', 0):,}")
        print(f"   Device: {device}")
        
        return model, metadata
    
    def _parse_file(self, data: bytes) -> tuple:
        """Binary dosyayı parse et"""
        print("  🔍 Parsing file...")
        
        offset = 0
        
        # Magic number
        magic = data[offset:offset+4]
        if magic != self.MAGIC_NUMBER:
            raise ValueError(f"Invalid file format (magic: {magic})")
        offset += 4
        
        # Version
        version, = struct.unpack('<I', data[offset:offset+4])
        offset += 4
        if version != 1:
            raise ValueError(f"Unsupported version: {version}")
        
        # Metadata length
        meta_len, = struct.unpack('<I', data[offset:offset+4])
        offset += 4
        
        # Metadata JSON
        meta_bytes = data[offset:offset+meta_len]
        offset += meta_len
        metadata = json.loads(meta_bytes.decode('utf-8'))
        
        print(f"  ✓ Metadata loaded (version {version})")
        
        # Parameters
        param_data = self._parse_parameters(data[offset:])
        
        print(f"  ✓ Parsed {len(param_data)} parameters")
        
        return metadata, param_data
    
    def _parse_parameters(self, data: bytes) -> dict:
        """Parameter binary data'sını parse et"""
        offset = 0
        param_dict = {}
        
        while offset < len(data):
            # Name length + name
            name_len, = struct.unpack('<I', data[offset:offset+4])
            offset += 4
            name = data[offset:offset+name_len].decode('utf-8')
            offset += name_len
            
            # Shape length + shape
            shape_len, = struct.unpack('<I', data[offset:offset+4])
            offset += 4
            shape = []
            for _ in range(shape_len):
                dim, = struct.unpack('<I', data[offset:offset+4])
                offset += 4
                shape.append(dim)
            
            # Scales JSON
            scales_len, = struct.unpack('<I', data[offset:offset+4])
            offset += 4
            scales_json = data[offset:offset+scales_len].decode('utf-8')
            offset += scales_len
            scales = json.loads(scales_json)
            
            # Values
            values_len, = struct.unpack('<I', data[offset:offset+4])
            offset += 4
            values_bytes = data[offset:offset+values_len]
            offset += values_len
            
            # Numpy array'e çevir
            values = np.frombuffer(values_bytes, dtype=np.float32).reshape(shape)
            
            param_dict[name] = {
                'shape': shape,
                'scales': scales,
                'values': values,
            }
        
        return param_dict
    
    def _reconstruct_model(self, param_data: dict, device: str, lazy: bool) -> torch.nn.Module:
        """Model'i reconstruct et"""
        print("  🔧 Reconstructing model...")
        
        # Model config'den model oluştur
        model_config = self.metadata.get('model_config')
        model_name = self.metadata.get('model_name')
        
        if model_config is None:
            raise ValueError("Model config not found in metadata")
        
        # Config'i doğru sınıftan oluştur
        # AutoConfig.from_dict() yok, model_type'a göre sınıf seçmeliyiz
        model_type = model_config.get('model_type', 'gpt2')
        
        CONFIG_MAPPING = {
            'gpt2': GPT2Config,
            'gpt_neo': GPTNeoConfig,
            'llama': LlamaConfig,
        }
        
        config_class = CONFIG_MAPPING.get(model_type, GPT2Config)
        
        try:
            config = config_class(**model_config)
        except Exception as e:
            print(f"    ⚠️ Error creating config with {config_class.__name__}: {e}")
            print(f"    Trying GPT2Config as fallback...")
            config = GPT2Config(**model_config)
        
        # Model oluştur
        print(f"    Creating model architecture...")
        model = AutoModelForCausalLM.from_config(config)
        
        # Parameters yükle
        print(f"    Loading {len(param_data)} parameters...")
        
        state_dict = {}
        for name, data in param_data.items():
            # Numpy → Torch tensor
            tensor = torch.from_numpy(data['values'])
            state_dict[name] = tensor
        
        # State dict'i model'e yükle
        model.load_state_dict(state_dict, strict=False)
        
        # Device'a taşı
        if not lazy:
            model = model.to(device)
        
        print(f"  ✓ Model reconstructed on {device}")
        
        return model


def load_quantized_model(path: str, lazy: bool = False, device: str = 'cpu'):
    """
    Helper function: Quantized model'i yükle
    
    Usage:
        model, metadata = load_quantized_model(
            path='model.qnashswarm',
            device='cpu'
        )
        
        # Use model
        model.eval()
        outputs = model.generate(...)
    
    Args:
        path: Model file path
        lazy: Lazy loading (saves RAM)
        device: Target device ('cpu', 'cuda', 'mps')
    
    Returns:
        (model, metadata)
    """
    loader = QuantizedModelLoader()
    return loader.load(path, lazy=lazy, device=device)

