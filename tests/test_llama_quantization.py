"""
LLaMA Quantization Test
=======================
Nash-Swarm quantization testi LLaMA modelleri için.

Model: TinyLlama 1.1B (başlangıç için küçük model)
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import numpy as np
import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def quantize_uniform_4bit_theoretical(model):
    """Uniform 4-bit quantization (teorik)"""
    print("  🔧 Uniform 4-bit quantization uygulanıyor...")
    
    quantized_model = type(model)(model.config)
    quantized_model.load_state_dict(model.state_dict())
    
    total_params = 0
    quantized_params = 0
    
    for name, param in quantized_model.named_parameters():
        if param.requires_grad and len(param.shape) >= 2:
            total_params += param.numel()
            
            # 4-bit quantization
            min_val = param.data.min()
            max_val = param.data.max()
            scale = (max_val - min_val) / 15
            
            if scale > 0:
                quantized = torch.clamp(
                    torch.round((param.data - min_val) / scale),
                    0, 15
                )
                param.data = quantized * scale + min_val
                quantized_params += param.numel()
    
    original_size = total_params * 4  # FP32 = 4 bytes
    quantized_size = quantized_params * 0.5 + (total_params - quantized_params) * 4  # 4-bit = 0.5 bytes
    compression_ratio = (1 - quantized_size / original_size) * 100
    
    return quantized_model, {
        'compression_ratio': compression_ratio,
        'original_size_mb': original_size / (1024**2),
        'quantized_size_mb': quantized_size / (1024**2),
        'quantized_params': quantized_params,
        'total_params': total_params
    }


def quantize_nash_swarm_adaptive(model):
    """Nash-Swarm adaptive quantization (2/4/8-bit mix)"""
    print("  🦅 Nash-Swarm adaptive quantization uygulanıyor...")
    
    quantized_model = type(model)(model.config)
    quantized_model.load_state_dict(model.state_dict())
    
    total_params = 0
    quantized_params = 0
    bit_distribution = {2: 0, 4: 0, 8: 0}
    
    for name, param in quantized_model.named_parameters():
        if param.requires_grad and len(param.shape) >= 2:
            total_params += param.numel()
            
            # Calculate importance (magnitude + neighborhood context)
            importance = param.data.abs()
            
            # Add neighborhood context (swarm cohesion)
            if len(param.shape) == 2:
                padded = torch.nn.functional.pad(importance, (1, 1, 1, 1), mode='replicate')
                neighbor_sum = (
                    padded[:-2, 1:-1] + padded[2:, 1:-1] +  # top, bottom
                    padded[1:-1, :-2] + padded[1:-1, 2:]    # left, right
                )
                importance = importance * (1 + neighbor_sum / 4)
            
            # Sample for large tensors
            flat_importance = importance.flatten()
            if flat_importance.numel() > 100000:
                sample_size = 10000
                indices = torch.randperm(flat_importance.numel())[:sample_size]
                sample = flat_importance[indices]
                q_90 = torch.quantile(sample, 0.90)
                q_70 = torch.quantile(sample, 0.70)
            else:
                q_90 = torch.quantile(flat_importance, 0.90)
                q_70 = torch.quantile(flat_importance, 0.70)
            
            # Adaptive bit allocation
            high_importance = importance >= q_90
            medium_importance = (importance >= q_70) & (importance < q_90)
            low_importance = importance < q_70
            
            # Quantize with different bit widths
            quantized_param = param.data.clone()
            
            # 8-bit for high importance
            if high_importance.any():
                mask = high_importance
                min_val = param.data[mask].min()
                max_val = param.data[mask].max()
                scale = (max_val - min_val) / 255
                if scale > 0:
                    quantized = torch.clamp(
                        torch.round((param.data[mask] - min_val) / scale),
                        0, 255
                    )
                    quantized_param[mask] = quantized * scale + min_val
                bit_distribution[8] += mask.sum().item()
            
            # 4-bit for medium importance
            if medium_importance.any():
                mask = medium_importance
                min_val = param.data[mask].min()
                max_val = param.data[mask].max()
                scale = (max_val - min_val) / 15
                if scale > 0:
                    quantized = torch.clamp(
                        torch.round((param.data[mask] - min_val) / scale),
                        0, 15
                    )
                    quantized_param[mask] = quantized * scale + min_val
                bit_distribution[4] += mask.sum().item()
            
            # 2-bit for low importance
            if low_importance.any():
                mask = low_importance
                min_val = param.data[mask].min()
                max_val = param.data[mask].max()
                scale = (max_val - min_val) / 3
                if scale > 0:
                    quantized = torch.clamp(
                        torch.round((param.data[mask] - min_val) / scale),
                        0, 3
                    )
                    quantized_param[mask] = quantized * scale + min_val
                bit_distribution[2] += mask.sum().item()
            
            param.data = quantized_param
            quantized_params += param.numel()
    
    # Calculate compression
    avg_bits = (bit_distribution[2] * 2 + bit_distribution[4] * 4 + bit_distribution[8] * 8) / quantized_params
    original_size = total_params * 4  # FP32
    quantized_size = (bit_distribution[2] * 2 + bit_distribution[4] * 4 + bit_distribution[8] * 8) / 8 / (1024**2)
    quantized_size += (total_params - quantized_params) * 4 / (1024**2)
    compression_ratio = (1 - (quantized_size * 1024**2) / original_size) * 100
    
    return quantized_model, {
        'compression_ratio': compression_ratio,
        'original_size_mb': original_size / (1024**2),
        'quantized_size_mb': quantized_size,
        'avg_bits': avg_bits,
        'bit_distribution': bit_distribution,
        'total_params': total_params
    }


def measure_perplexity_fast(model, tokenizer, text_samples, max_length=128):
    """Hızlı perplexity measurement"""
    model.eval()
    device = next(model.parameters()).device
    total_loss = 0.0
    count = 0
    
    with torch.no_grad():
        for text in text_samples:
            if len(text.strip()) < 10:
                continue
            
            try:
                inputs = tokenizer(text, return_tensors='pt', truncation=True, 
                                 max_length=max_length, padding='max_length')
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                outputs = model(**inputs, labels=inputs['input_ids'])
                loss = outputs.loss
                
                if not torch.isnan(loss) and not torch.isinf(loss):
                    total_loss += loss.item()
                    count += 1
            except Exception as e:
                continue
    
    if count == 0:
        return float('inf')
    
    avg_loss = total_loss / count
    perplexity = np.exp(avg_loss)
    return avg_loss, perplexity


def main():
    print("\n" + "="*80)
    print("🦙 LLAMA QUANTIZATION TEST - NASH-SWARM vs UNIFORM")
    print("="*80)
    
    # Model selection
    # Option 1: TinyLlama (1.1B) - küçük, hızlı test
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    
    # Option 2: Open LLaMA (3B) - daha büyük
    # model_name = "openlm-research/open_llama_3b"
    
    # Option 3: LLaMA 2 (7B) - büyük, yavaş
    # model_name = "meta-llama/Llama-2-7b-hf"  # Access token gerekebilir
    
    print(f"\n[1/6] Loading {model_name}...")
    print("⏱️  Bu işlem 2-5 dakika sürebilir (model büyüklüğüne göre)...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    baseline_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True
    )
    
    # Device setup
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        baseline_model = baseline_model.to(device)
        print(f"  ✓ Using MPS (Metal) acceleration ⚡")
    else:
        device = torch.device("cpu")
        print(f"  ✓ Using CPU")
    
    # Model info
    total_params = sum(p.numel() for p in baseline_model.parameters())
    print(f"  ✓ Model loaded: {total_params/1e9:.2f}B parameters")
    
    # Load dataset
    print("\n[2/6] Loading WikiText-2 dataset...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text_samples = [d['text'] for d in dataset if len(d['text'].strip()) > 20][:30]  # 30 samples
    print(f"  ✓ Loaded {len(text_samples)} text samples")
    
    # Baseline perplexity
    print("\n[3/6] Measuring Baseline (FP32) perplexity...")
    print(f"  Testing on {len(text_samples)} samples...")
    baseline_loss, baseline_ppl = measure_perplexity_fast(baseline_model, tokenizer, text_samples)
    baseline_size_mb = total_params * 4 / (1024**2)
    print(f"  ✓ Baseline Loss: {baseline_loss:.3f}")
    print(f"  ✓ Baseline Perplexity: {baseline_ppl:.2f}")
    print(f"  ✓ Baseline Size: {baseline_size_mb:.1f} MB")
    
    # Uniform 4-bit quantization
    print("\n[4/6] Testing Uniform 4-bit Quantization...")
    baseline_cpu = baseline_model.cpu()
    uniform_model, uniform_info = quantize_uniform_4bit_theoretical(baseline_cpu)
    uniform_model = uniform_model.to(device)
    
    uniform_loss, uniform_ppl = measure_perplexity_fast(uniform_model, tokenizer, text_samples)
    uniform_delta = ((uniform_loss - baseline_loss) / baseline_loss) * 100
    
    print(f"  ✓ Uniform 4-bit Loss: {uniform_loss:.3f} (Δ{uniform_delta:+.2f}%)")
    print(f"  ✓ Compression: {uniform_info['compression_ratio']:.1f}%")
    print(f"  ✓ Size: {uniform_info['quantized_size_mb']:.1f} MB")
    
    # Nash-Swarm quantization
    print("\n[5/6] Testing Nash-Swarm Adaptive Quantization...")
    baseline_cpu = baseline_model.cpu()
    nash_model, nash_info = quantize_nash_swarm_adaptive(baseline_cpu)
    nash_model = nash_model.to(device)
    
    nash_loss, nash_ppl = measure_perplexity_fast(nash_model, tokenizer, text_samples)
    nash_delta = ((nash_loss - baseline_loss) / baseline_loss) * 100
    
    print(f"  ✓ Nash-Swarm Loss: {nash_loss:.3f} (Δ{nash_delta:+.2f}%)")
    print(f"  ✓ Compression: {nash_info['compression_ratio']:.1f}%")
    print(f"  ✓ Size: {nash_info['quantized_size_mb']:.1f} MB")
    print(f"  ✓ Avg bits/param: {nash_info['avg_bits']:.2f}")
    
    # Results summary
    print("\n" + "="*80)
    print("📊 LLAMA QUANTIZATION RESULTS SUMMARY")
    print("="*80)
    
    print(f"\n{'Method':<20} {'Loss':<12} {'Δ Loss':<12} {'Compression':<15} {'Size (MB)':<12}")
    print("-"*80)
    print(f"{'Baseline (FP32)':<20} {baseline_loss:>9.3f}   {'-':>10}   {'-':>13}   {baseline_size_mb:>10.1f}")
    print(f"{'Uniform 4-bit':<20} {uniform_loss:>9.3f}   {uniform_delta:>9.2f}%   {uniform_info['compression_ratio']:>12.1f}%   {uniform_info['quantized_size_mb']:>10.1f}")
    print(f"{'Nash-Swarm':<20} {nash_loss:>9.3f}   {nash_delta:>9.2f}%   {nash_info['compression_ratio']:>12.1f}%   {nash_info['quantized_size_mb']:>10.1f}")
    
    print("\n" + "="*80)
    print("🎯 KEY FINDINGS")
    print("="*80)
    
    accuracy_advantage = abs(uniform_delta / nash_delta) if nash_delta != 0 else float('inf')
    compression_advantage = nash_info['compression_ratio'] - uniform_info['compression_ratio']
    
    print(f"\n1. COMPRESSION:")
    print(f"   Nash-Swarm: {nash_info['compression_ratio']:.1f}%")
    print(f"   Uniform 4-bit: {uniform_info['compression_ratio']:.1f}%")
    print(f"   → Advantage: {compression_advantage:+.1f}%")
    
    print(f"\n2. ACCURACY:")
    print(f"   Nash-Swarm: {nash_delta:+.2f}% loss degradation")
    print(f"   Uniform 4-bit: {uniform_delta:+.2f}% loss degradation")
    if nash_delta < uniform_delta:
        print(f"   → ✅ Nash-Swarm is {accuracy_advantage:.1f}× better!")
    else:
        print(f"   → ⚠️ Uniform is better (expected for some models)")
    
    print(f"\n3. MODEL SIZE:")
    print(f"   {model_name}")
    print(f"   Parameters: {total_params/1e9:.2f}B")
    print(f"   FP32: {baseline_size_mb:.1f} MB")
    print(f"   Nash-Swarm: {nash_info['quantized_size_mb']:.1f} MB")
    
    print("\n" + "="*80)
    print("✅ LLaMA test complete!")
    print("="*80)


if __name__ == "__main__":
    main()

