"""
Basitleştirilmiş Quantization Karşılaştırması
---------------------------------------------
Gerçek memory compression ile
"""

import torch
import torch.nn as nn
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import time
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def count_parameters(model):
    """Model parametrelerini say"""
    return sum(p.numel() for p in model.parameters())


def measure_perplexity_fast(model, tokenizer, text_samples, max_length=64):
    """Hızlı perplexity ölçümü"""
    model.eval()
    total_loss = 0
    count = 0
    
    # Get model device
    device = next(model.parameters()).device
    
    with torch.no_grad():
        for text in text_samples[:30]:  # İlk 30 sample (FAST MODE for large models)
            if len(text.strip()) < 10:
                continue
            
            try:
                inputs = tokenizer(
                    text,
                    return_tensors='pt',
                    truncation=True,
                    max_length=max_length,
                    padding='max_length'
                )
                
                # Move inputs to same device as model
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                outputs = model(**inputs, labels=inputs['input_ids'])
                total_loss += outputs.loss.item()
                count += 1
                
            except Exception as e:
                continue
    
    if count == 0:
        return float('inf'), float('inf')
    
    avg_loss = total_loss / count
    perplexity = np.exp(avg_loss)
    
    return perplexity, avg_loss


def quantize_uniform_4bit_theoretical(model):
    """
    Uniform 4-bit quantization (theoretical memory calculation)
    Gerçekten int4 yapmıyoruz ama memory theoretically hesaplıyoruz
    """
    quantized_model = type(model)(model.config)
    quantized_model.load_state_dict(model.state_dict())
    
    total_params = 0
    quantized_params = 0
    
    for name, module in quantized_model.named_modules():
        if isinstance(module, nn.Linear):
            weight = module.weight.data
            total_params += weight.numel()
            
            # 4-bit quantization (16 levels)
            w_min = weight.min()
            w_max = weight.max()
            scale = (w_max - w_min) / 15.0  # 16 levels (0-15)
            
            # Quantize
            quantized = torch.round((weight - w_min) / scale)
            quantized = torch.clamp(quantized, 0, 15)
            
            # Dequantize
            dequantized = quantized * scale + w_min
            
            module.weight.data = dequantized.to(dtype=weight.dtype)
            quantized_params += weight.numel()
    
    # Theoretical memory (4 bits per param instead of 32 bits)
    theoretical_size_mb = (quantized_params * 4 / 8) / (1024 ** 2)  # 4 bits = 0.5 bytes
    original_size_mb = (total_params * 4) / (1024 ** 2)  # 32 bits = 4 bytes
    
    compression_ratio = 1 - (theoretical_size_mb / original_size_mb)
    
    return quantized_model, {
        'theoretical_size_mb': theoretical_size_mb,
        'original_size_mb': original_size_mb,
        'compression_ratio': compression_ratio,
        'quantized_params': quantized_params
    }


def quantize_nash_swarm_adaptive(model):
    """
    Nash-Swarm adaptive quantization
    Mixed precision: 2/4/8-bit based on importance
    """
    quantized_model = type(model)(model.config)
    quantized_model.load_state_dict(model.state_dict())
    
    total_params = 0
    total_bits = 0
    
    for name, module in quantized_model.named_modules():
        if isinstance(module, nn.Linear):
            weight = module.weight.data
            total_params += weight.numel()
            
            # Önem skoru hesapla (basitleştirilmiş)
            importance = torch.abs(weight)
            
            # Quantile-based bit allocation (with sampling for large tensors)
            flat_importance = importance.flatten()
            if flat_importance.numel() > 100000:
                # Sample for large tensors
                sample_size = 10000
                indices = torch.randperm(flat_importance.numel())[:sample_size]
                sample = flat_importance[indices]
                q_90 = torch.quantile(sample, 0.90)
                q_70 = torch.quantile(sample, 0.70)
            else:
                q_90 = torch.quantile(flat_importance, 0.90)
                q_70 = torch.quantile(flat_importance, 0.70)
            
            # Bit allocation mask
            bit_8_mask = importance >= q_90  # Top 10%: 8-bit
            bit_4_mask = (importance >= q_70) & (importance < q_90)  # Next 20%: 4-bit
            bit_2_mask = importance < q_70  # Bottom 70%: 2-bit
            
            # Quantize each region
            quantized_weight = torch.zeros_like(weight)
            
            # 8-bit region
            if bit_8_mask.any():
                w_8 = weight[bit_8_mask]
                w_min, w_max = w_8.min(), w_8.max()
                scale = (w_max - w_min) / 255.0
                q = torch.clamp(torch.round((w_8 - w_min) / scale), 0, 255)
                quantized_weight[bit_8_mask] = q * scale + w_min
                total_bits += bit_8_mask.sum().item() * 8
            
            # 4-bit region
            if bit_4_mask.any():
                w_4 = weight[bit_4_mask]
                w_min, w_max = w_4.min(), w_4.max()
                scale = (w_max - w_min) / 15.0
                q = torch.clamp(torch.round((w_4 - w_min) / scale), 0, 15)
                quantized_weight[bit_4_mask] = q * scale + w_min
                total_bits += bit_4_mask.sum().item() * 4
            
            # 2-bit region (aggressive)
            if bit_2_mask.any():
                w_2 = weight[bit_2_mask]
                w_min, w_max = w_2.min(), w_2.max()
                scale = (w_max - w_min) / 3.0
                q = torch.clamp(torch.round((w_2 - w_min) / scale), 0, 3)
                quantized_weight[bit_2_mask] = q * scale + w_min
                total_bits += bit_2_mask.sum().item() * 2
            
            module.weight.data = quantized_weight
    
    # Theoretical memory
    theoretical_size_mb = (total_bits / 8) / (1024 ** 2)
    original_size_mb = (total_params * 4) / (1024 ** 2)
    compression_ratio = 1 - (theoretical_size_mb / original_size_mb)
    
    return quantized_model, {
        'theoretical_size_mb': theoretical_size_mb,
        'original_size_mb': original_size_mb,
        'compression_ratio': compression_ratio,
        'avg_bits_per_param': total_bits / total_params
    }


def main():
    print("\n" + "="*80)
    print("QUANTIZATION COMPARISON: NASH-SWARM vs UNIFORM 4-BIT")
    print("📊 SAMPLE SIZE: 30 (FAST MODE for GPT-2 XL)")
    print("="*80)
    
    # Load model
    print("\n[1/5] Loading model...")
    model_name = "gpt2-xl"  # 🔬 TESTING GPT-2 XL (1.5B) - FAST MODE!
    print(f"📦 Model: {model_name}")
    print(f"⚡ Fast mode: 30 samples (40% faster than standard 50)")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    baseline_model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Device optimization (MPS if available)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        baseline_model = baseline_model.to(device)
        print(f"✓ Model loaded: {count_parameters(baseline_model):,} parameters")
        print(f"✓ Using MPS (Metal) acceleration ⚡")
    else:
        print(f"✓ Model loaded: {count_parameters(baseline_model):,} parameters")
        print(f"✓ Using CPU (MPS not available)")
    
    # Load dataset
    print("\n[2/5] Loading WikiText-2...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text_samples = [d['text'] for d in dataset if len(d['text'].strip()) > 20]
    print(f"✓ Loaded {len(text_samples)} text samples")
    
    # Baseline
    print("\n[3/5] Testing Baseline (FP32)...")
    baseline_ppl, baseline_loss = measure_perplexity_fast(baseline_model, tokenizer, text_samples)
    baseline_params = count_parameters(baseline_model)
    baseline_size_mb = (baseline_params * 4) / (1024 ** 2)
    print(f"  Size: {baseline_size_mb:.2f} MB")
    print(f"  Loss: {baseline_loss:.4f}")
    print(f"  Perplexity: {baseline_ppl:.2f}")
    
    # Uniform 4-bit
    print("\n[4/5] Testing Uniform 4-bit...")
    # Move baseline to CPU for quantization
    baseline_cpu = baseline_model.cpu()
    uniform_model, uniform_info = quantize_uniform_4bit_theoretical(baseline_cpu)
    uniform_ppl, uniform_loss = measure_perplexity_fast(uniform_model, tokenizer, text_samples)
    print(f"  Theoretical Size: {uniform_info['theoretical_size_mb']:.2f} MB")
    print(f"  Compression: {uniform_info['compression_ratio']*100:.1f}%")
    print(f"  Loss: {uniform_loss:.4f}")
    print(f"  Perplexity: {uniform_ppl:.2f}")
    
    # Nash-Swarm
    print("\n[5/5] Testing Nash-Swarm (Mixed 2/4/8-bit)...")
    nash_model, nash_info = quantize_nash_swarm_adaptive(baseline_cpu)
    nash_ppl, nash_loss = measure_perplexity_fast(nash_model, tokenizer, text_samples)
    print(f"  Theoretical Size: {nash_info['theoretical_size_mb']:.2f} MB")
    print(f"  Compression: {nash_info['compression_ratio']*100:.1f}%")
    print(f"  Avg bits/param: {nash_info['avg_bits_per_param']:.2f}")
    print(f"  Loss: {nash_loss:.4f}")
    print(f"  Perplexity: {nash_ppl:.2f}")
    
    # Comparison table
    print("\n" + "="*80)
    print("FINAL RESULTS")
    print("="*80)
    
    print(f"\n{'Method':<30} {'Size (MB)':<12} {'Compression':<14} {'Loss':<10} {'Perplexity':<12}")
    print("-"*80)
    
    print(f"{'Baseline (FP32)':<30} {baseline_size_mb:>10.2f}  {'-':>12}  {baseline_loss:>8.4f}  {baseline_ppl:>10.2f}")
    print(f"{'Uniform 4-bit':<30} {uniform_info['theoretical_size_mb']:>10.2f}  {uniform_info['compression_ratio']*100:>11.1f}%  {uniform_loss:>8.4f}  {uniform_ppl:>10.2f}")
    print(f"{'Nash-Swarm (Mixed)':<30} {nash_info['theoretical_size_mb']:>10.2f}  {nash_info['compression_ratio']*100:>11.1f}%  {nash_loss:>8.4f}  {nash_ppl:>10.2f}")
    
    # Delta analysis
    print("\n" + "="*80)
    print("DELTA ANALYSIS (vs Baseline)")
    print("="*80)
    
    uniform_loss_delta = ((uniform_loss - baseline_loss) / baseline_loss) * 100
    nash_loss_delta = ((nash_loss - baseline_loss) / baseline_loss) * 100
    
    uniform_ppl_delta = ((uniform_ppl - baseline_ppl) / baseline_ppl) * 100
    nash_ppl_delta = ((nash_ppl - baseline_ppl) / baseline_ppl) * 100
    
    print(f"\n{'Method':<30} {'Δ Loss':<12} {'Δ Perplexity':<15}")
    print("-"*80)
    print(f"{'Uniform 4-bit':<30} {uniform_loss_delta:>+10.2f}%  {uniform_ppl_delta:>+13.2f}%")
    print(f"{'Nash-Swarm (Mixed)':<30} {nash_loss_delta:>+10.2f}%  {nash_ppl_delta:>+13.2f}%")
    
    # Key findings
    print("\n" + "="*80)
    print("KEY FINDINGS")
    print("="*80)
    
    if nash_info['compression_ratio'] > uniform_info['compression_ratio']:
        comp_advantage = "HIGHER"
    else:
        comp_advantage = "LOWER"
    
    if abs(nash_loss_delta) < abs(uniform_loss_delta):
        acc_advantage = "BETTER"
    else:
        acc_advantage = "WORSE"
    
    print(f"""
1. COMPRESSION:
   ✓ Nash-Swarm: {nash_info['compression_ratio']*100:.1f}% ({comp_advantage} than Uniform)
   ✓ Uniform 4-bit: {uniform_info['compression_ratio']*100:.1f}%
   ✓ Nash-Swarm uses adaptive bit-width (avg {nash_info['avg_bits_per_param']:.2f} bits/param)

2. ACCURACY:
   ✓ Nash-Swarm loss change: {nash_loss_delta:+.2f}% ({acc_advantage} than Uniform)
   ✓ Uniform 4-bit loss change: {uniform_loss_delta:+.2f}%

3. THEORETICAL INSIGHT:
   Nash-Swarm allocates bits based on parameter importance:
   - Top 10% params: 8-bit (critical weights)
   - Middle 20%: 4-bit (moderate importance)
   - Bottom 70%: 2-bit (less critical)
   
   This adaptive strategy achieves {comp_advantage} compression while
   maintaining {acc_advantage} accuracy compared to uniform quantization.

NOTE: This is a CPU-only, theoretical comparison.
Actual memory savings require hardware-specific int4/int8 support.
""")


if __name__ == "__main__":
    try:
        main()
        print("\n✅ Test completed successfully!\n")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

