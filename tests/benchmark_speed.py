"""
Speed Benchmark: Baseline vs Nash-Swarm Quantization
----------------------------------------------------
Gerçek inference time overhead'ini ölçüyoruz
"""

import torch
import torch.nn as nn
import time
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def quantize_nash_swarm_adaptive(model):
    """Nash-Swarm adaptive quantization (same as before)"""
    quantized_model = type(model)(model.config)
    quantized_model.load_state_dict(model.state_dict())
    
    for name, module in quantized_model.named_modules():
        if isinstance(module, nn.Linear):
            weight = module.weight.data
            importance = torch.abs(weight)
            
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
            
            bit_8_mask = importance >= q_90
            bit_4_mask = (importance >= q_70) & (importance < q_90)
            bit_2_mask = importance < q_70
            
            quantized_weight = torch.zeros_like(weight)
            
            if bit_8_mask.any():
                w_8 = weight[bit_8_mask]
                w_min, w_max = w_8.min(), w_8.max()
                scale = (w_max - w_min) / 255.0
                q = torch.clamp(torch.round((w_8 - w_min) / scale), 0, 255)
                quantized_weight[bit_8_mask] = q * scale + w_min
            
            if bit_4_mask.any():
                w_4 = weight[bit_4_mask]
                w_min, w_max = w_4.min(), w_4.max()
                scale = (w_max - w_min) / 15.0
                q = torch.clamp(torch.round((w_4 - w_min) / scale), 0, 15)
                quantized_weight[bit_4_mask] = q * scale + w_min
            
            if bit_2_mask.any():
                w_2 = weight[bit_2_mask]
                w_min, w_max = w_2.min(), w_2.max()
                scale = (w_max - w_min) / 3.0
                q = torch.clamp(torch.round((w_2 - w_min) / scale), 0, 3)
                quantized_weight[bit_2_mask] = q * scale + w_min
            
            module.weight.data = quantized_weight
    
    return quantized_model


def measure_inference_speed(model, tokenizer, num_samples=20, warmup=5):
    """
    Measure inference speed
    Returns: average time per inference (seconds)
    """
    model.eval()
    device = next(model.parameters()).device
    
    # Prepare test inputs
    test_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is transforming the world of artificial intelligence.",
        "Natural language processing enables computers to understand human language.",
        "Deep learning models require significant computational resources.",
        "Quantization reduces model size while preserving accuracy."
    ] * 4  # 20 samples
    
    inputs_list = []
    for text in test_texts[:num_samples]:
        inputs = tokenizer(
            text,
            return_tensors='pt',
            truncation=True,
            max_length=32,
            padding='max_length'
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        inputs_list.append(inputs)
    
    # Warmup
    print(f"  Warmup ({warmup} iterations)...")
    with torch.no_grad():
        for i in range(warmup):
            _ = model(**inputs_list[i % len(inputs_list)])
    
    # Actual measurement
    print(f"  Measuring ({num_samples} samples)...")
    times = []
    with torch.no_grad():
        for inputs in inputs_list:
            start = time.time()
            _ = model(**inputs)
            end = time.time()
            times.append(end - start)
    
    avg_time = np.mean(times)
    std_time = np.std(times)
    
    return avg_time, std_time, times


def main():
    print("\n" + "="*80)
    print("⚡ SPEED BENCHMARK: BASELINE vs NASH-SWARM ⚡")
    print("="*80)
    
    # Test küçük model (GPT-2) - hızlı olsun
    model_name = "gpt2"
    num_samples = 20
    
    print(f"\n[1/4] Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    baseline_model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # MPS kullan
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        baseline_model = baseline_model.to(device)
        print(f"✓ Using MPS (Metal) acceleration")
    else:
        print(f"✓ Using CPU")
    
    # Baseline speed
    print(f"\n[2/4] Measuring Baseline (FP32) speed...")
    baseline_time, baseline_std, _ = measure_inference_speed(
        baseline_model, tokenizer, num_samples=num_samples
    )
    print(f"  ✓ Average: {baseline_time*1000:.2f} ms (±{baseline_std*1000:.2f} ms)")
    
    # Quantize
    print(f"\n[3/4] Quantizing model (Nash-Swarm)...")
    baseline_cpu = baseline_model.cpu()
    nash_model = quantize_nash_swarm_adaptive(baseline_cpu)
    
    # Move back to device
    if torch.backends.mps.is_available():
        nash_model = nash_model.to(device)
    
    print(f"  ✓ Quantization complete")
    
    # Nash-Swarm speed
    print(f"\n[4/4] Measuring Nash-Swarm speed...")
    nash_time, nash_std, _ = measure_inference_speed(
        nash_model, tokenizer, num_samples=num_samples
    )
    print(f"  ✓ Average: {nash_time*1000:.2f} ms (±{nash_std*1000:.2f} ms)")
    
    # Results
    print("\n" + "="*80)
    print("📊 SPEED BENCHMARK RESULTS")
    print("="*80)
    
    overhead = nash_time / baseline_time
    speedup = baseline_time / nash_time
    
    print(f"\n{'Method':<20} {'Avg Time (ms)':<15} {'Std (ms)':<12} {'Throughput (inf/s)':<15}")
    print("-"*80)
    print(f"{'Baseline (FP32)':<20} {baseline_time*1000:>13.2f}  {baseline_std*1000:>10.2f}  {1/baseline_time:>13.2f}")
    print(f"{'Nash-Swarm':<20} {nash_time*1000:>13.2f}  {nash_std*1000:>10.2f}  {1/nash_time:>13.2f}")
    
    print("\n" + "="*80)
    print("⚡ OVERHEAD ANALYSIS")
    print("="*80)
    
    if overhead > 1:
        print(f"\n❌ Nash-Swarm is {overhead:.2f}× SLOWER than baseline")
        print(f"   (Baseline: {baseline_time*1000:.2f} ms, Nash: {nash_time*1000:.2f} ms)")
    else:
        print(f"\n✅ Nash-Swarm is {speedup:.2f}× FASTER than baseline")
        print(f"   (Baseline: {baseline_time*1000:.2f} ms, Nash: {nash_time*1000:.2f} ms)")
    
    print(f"\n📝 INTERPRETATION:")
    if overhead > 100:
        print(f"   ⚠️  CRITICAL: {overhead:.0f}× overhead is UNACCEPTABLE for production")
        print(f"   → GPU/CUDA optimization MANDATORY")
    elif overhead > 10:
        print(f"   ⚠️  SIGNIFICANT: {overhead:.1f}× overhead limits practical use")
        print(f"   → Further optimization recommended")
    elif overhead > 2:
        print(f"   ⚠️  MODERATE: {overhead:.1f}× overhead is acceptable for some use cases")
        print(f"   → Optimization would still help")
    elif overhead > 1:
        print(f"   ✓ ACCEPTABLE: {overhead:.2f}× overhead is reasonable")
    else:
        print(f"   ✅ EXCELLENT: Nash-Swarm is faster! (unexpected but great)")
    
    print(f"\n💡 NOTE:")
    print(f"   This measures INFERENCE ONLY (quantization already done)")
    print(f"   Quantization itself adds one-time setup cost")
    print(f"   Results may vary with different models/hardware")
    
    print("\n" + "="*80)
    print("✅ Benchmark complete!")
    print("="*80 + "\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

