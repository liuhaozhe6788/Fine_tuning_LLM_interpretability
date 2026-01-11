"""
Logit Lens implementation for ADL analysis.

Borrows core logic from diffing-toolkit but adapted for our setup:
- Uses nnsight LanguageModel instead of StandardizedTransformer
- Works with our config structure
- Saves results to scratch space
"""
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import List, Dict, Tuple, Any
from nnsight import LanguageModel
import json
from tqdm import tqdm
import pandas as pd
import gc
import os
import time

from utils import (
    load_finqa_data,
    load_models,
    load_model,
    extract_activations,
    compute_activation_differences,
    save_json,
    save_csv,
    clear_gpu_cache,
    setup_hf_auth,
)
from utils_hf_activations import load_hf_activations
from config import ADLConfig


def logit_lens_nnsight(
    latent: torch.Tensor,
    model: LanguageModel,
    device: str = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Project a latent vector through the model's output head to get token predictions.
    
    Adapted from diffing-toolkit's logit_lens function for nnsight models.
    Uses direct access to model's norm and lm_head modules.
    
    Optimized for GPU memory: uses no_grad, clears cache, moves results to CPU immediately.
    
    Args:
        latent: Activation vector of shape [..., hidden_size]
        model: nnsight LanguageModel instance
        device: Device to run on (if None, will use model's device)
        
    Returns:
        Tuple of (probs, inv_probs) where:
        - probs: Token probabilities from projecting latent
        - inv_probs: Token probabilities from projecting -latent
        Shape: [..., vocab_size]
    """
    # Get device from model if not specified
    if device is None:
        try:
            # Try to get device from model parameters
            device = next(model.model.parameters()).device
        except:
            device = "cpu"
    
    # Clear GPU cache before starting (if on GPU)
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
    
    # Move latent to device
    latent = latent.to(device)
    
    # Get model's hidden size
    hidden_size = model.config.hidden_size
    vocab_size = model.config.vocab_size
    
    if latent.shape[-1] != hidden_size:
        raise ValueError(
            f"Latent shape {latent.shape} does not match model hidden size {hidden_size}"
        )
    
    # Reshape if needed (handle batched or single vector)
    original_shape = latent.shape[:-1]
    latent_flat = latent.reshape(-1, hidden_size)
    
    # Access model's final layer norm and language model head directly
    # For Mistral models: model.model.norm and model.lm_head
    norm_module = model.model.norm
    lm_head_module = model.lm_head
    
    # Use no_grad to avoid gradient computation overhead
    with torch.no_grad():
        # Apply layer norm
        normed_vector = norm_module(latent_flat)  # [B, hidden_size]
        
        # Project through language model head
        logits = lm_head_module(normed_vector)  # [B, vocab_size]
        probs = F.softmax(logits, dim=-1)
        
        # Move to CPU immediately to free GPU memory
        probs = probs.cpu()
        del logits, normed_vector  # Free GPU memory
        
        # Clear cache after first projection
        if device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Get inverse probabilities (from -latent)
        inv_normed_vector = norm_module(-latent_flat)
        inv_logits = lm_head_module(inv_normed_vector)
        inv_probs = F.softmax(inv_logits, dim=-1)
        
        # Move to CPU immediately
        inv_probs = inv_probs.cpu()
        del inv_logits, inv_normed_vector  # Free GPU memory
    
    # Clear cache one more time
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()
        import gc
        gc.collect()
    
    # Reshape to original shape + vocab_size
    probs = probs.reshape(*original_shape, vocab_size)
    inv_probs = inv_probs.reshape(*original_shape, vocab_size)
    
    return probs, inv_probs


def compute_logit_lens_for_position(
    mean_diff: torch.Tensor,
    base_mean: torch.Tensor,
    ft_mean: torch.Tensor,
    ft_model: LanguageModel,
    k: int = 100,
    device: str = "cuda:0"
) -> Dict[str, Any]:
    """
    Compute logit lens for a single position.
    
    Args:
        mean_diff: Mean activation difference [hidden_size]
        base_mean: Mean base activations [hidden_size]
        ft_mean: Mean fine-tuned activations [hidden_size]
        ft_model: Fine-tuned model for projection
        k: Top-k tokens to return
        device: Device to use
        
    Returns:
        Dictionary with top-k tokens and probabilities for:
        - diff: activation difference
        - base: base model activations
        - ft: fine-tuned model activations
    """
    results = {}
    
    # Project difference
    diff_probs, diff_inv_probs = logit_lens_nnsight(mean_diff.unsqueeze(0), ft_model, device)
    diff_probs = diff_probs.squeeze(0)
    diff_inv_probs = diff_inv_probs.squeeze(0)
    
    diff_top_k_probs, diff_top_k_indices = torch.topk(diff_probs, k, dim=-1)
    diff_inv_top_k_probs, diff_inv_top_k_indices = torch.topk(diff_inv_probs, k, dim=-1)
    
    results["diff"] = {
        "top_k_probs": diff_top_k_probs.cpu().tolist(),
        "top_k_indices": diff_top_k_indices.cpu().tolist(),
        "inv_top_k_probs": diff_inv_top_k_probs.cpu().tolist(),
        "inv_top_k_indices": diff_inv_top_k_indices.cpu().tolist(),
    }
    
    # Project base mean
    base_probs, base_inv_probs = logit_lens_nnsight(base_mean.unsqueeze(0), ft_model, device)
    base_probs = base_probs.squeeze(0)
    base_inv_probs = base_inv_probs.squeeze(0)
    
    base_top_k_probs, base_top_k_indices = torch.topk(base_probs, k, dim=-1)
    base_inv_top_k_probs, base_inv_top_k_indices = torch.topk(base_inv_probs, k, dim=-1)
    
    results["base"] = {
        "top_k_probs": base_top_k_probs.cpu().tolist(),
        "top_k_indices": base_top_k_indices.cpu().tolist(),
        "inv_top_k_probs": base_inv_top_k_probs.cpu().tolist(),
        "inv_top_k_indices": base_inv_top_k_indices.cpu().tolist(),
    }
    
    # Project fine-tuned mean
    ft_probs, ft_inv_probs = logit_lens_nnsight(ft_mean.unsqueeze(0), ft_model, device)
    ft_probs = ft_probs.squeeze(0)
    ft_inv_probs = ft_inv_probs.squeeze(0)
    
    ft_top_k_probs, ft_top_k_indices = torch.topk(ft_probs, k, dim=-1)
    ft_inv_top_k_probs, ft_inv_top_k_indices = torch.topk(ft_inv_probs, k, dim=-1)
    
    results["ft"] = {
        "top_k_probs": ft_top_k_probs.cpu().tolist(),
        "top_k_indices": ft_top_k_indices.cpu().tolist(),
        "inv_top_k_probs": ft_inv_top_k_probs.cpu().tolist(),
        "inv_top_k_indices": ft_inv_top_k_indices.cpu().tolist(),
    }
    
    return results


def run_logit_lens_analysis(config: ADLConfig) -> None:
    """
    Run logit lens analysis on activation differences.
    
    This is the main function that:
    1. Loads data
    2. Extracts activations from both models
    3. Computes mean differences per position
    4. Projects through logit lens
    5. Saves results
    """
    print("=" * 60)
    print("Running Logit Lens Analysis")
    print("=" * 60)
    
    setup_hf_auth(config.hf_token_env)
    
    base_model = None
    ft_model = None
    hf_loading_successful = False  # Track if HF loading succeeded
    
    try:
        # Option 1: Try loading pre-extracted activations from HuggingFace (saves GPU memory)
        # If this fails, automatically fall back to direct extraction
        if config.use_hf_activations:
            print(f"\nAttempting to load pre-extracted activations from HuggingFace...")
            print(f"  Base: {config.hf_base_acts_dataset}")
            print(f"  FT: {config.hf_ft_acts_dataset}")
            print(f"  This reuses activations from crosscoder experiments (same layer {config.layer}, seed {config.random_seed})")
            
            import os
            try:
                base_acts, ft_acts, scaling_factors = load_hf_activations(
                    base_dataset_id=config.hf_base_acts_dataset,
                    ft_dataset_id=config.hf_ft_acts_dataset,
                    hf_token=os.environ.get(config.hf_token_env),
                    num_samples=config.num_samples
                )
                
                # Truncate to sequence_length if needed
                if base_acts.shape[1] > config.sequence_length:
                    base_acts = base_acts[:, :config.sequence_length, :]
                    ft_acts = ft_acts[:, :config.sequence_length, :]
                elif base_acts.shape[1] < config.sequence_length:
                    # Pad if shorter (unlikely but handle it)
                    pad_len = config.sequence_length - base_acts.shape[1]
                    base_padding = torch.zeros(base_acts.shape[0], pad_len, base_acts.shape[2], dtype=base_acts.dtype)
                    ft_padding = torch.zeros(ft_acts.shape[0], pad_len, ft_acts.shape[2], dtype=ft_acts.dtype)
                    base_acts = torch.cat([base_acts, base_padding], dim=1)
                    ft_acts = torch.cat([ft_acts, ft_padding], dim=1)
                
                print(f"  ✅ Loaded activations: base {base_acts.shape}, ft {ft_acts.shape}")
                print(f"  Note: Activations are scaled. Differences will still be valid.")
                
                # Only need to load FT model for logit lens projections
                print("\n  Loading fine-tuned model for logit lens projections...")
                ft_model = load_model(config.ft_model_id, device=config.device, use_quantization=config.use_quantization)
                print("  ✅ Fine-tuned model loaded")
                
                hf_loading_successful = True
                
            except Exception as e:
                print(f"\n  ⚠️  Failed to load activations from HuggingFace: {type(e).__name__}: {e}")
                print(f"  Falling back to direct activation extraction...")
                hf_loading_successful = False
        
        # Option 2: Extract activations ourselves (original approach)
        # This is used if use_hf_activations=False OR if HF loading failed
        if not config.use_hf_activations or (config.use_hf_activations and not hf_loading_successful):
            # Load data
            print(f"\nLoading {config.num_samples} samples from FinQA...")
            texts = load_finqa_data(
                dataset_path=config.dataset_path,
                num_samples=config.num_samples,
                random_seed=config.random_seed
            )
            print(f"✅ Loaded {len(texts)} samples")
            
            # Load models one at a time to save memory
            print(f"\nExtracting activations from layer {config.layer}...")
            
            # Load and extract from base model
            print("  Loading base model...")
            base_model = load_model(config.base_model_id, device=config.device, use_quantization=config.use_quantization)
            print("  Base model loaded")
            
            print("  Extracting base activations...")
            using_cpu = False
            try:
                base_acts = extract_activations(
                    base_model,
                    texts,
                    layer=config.layer,
                    num_tokens=config.sequence_length,
                    batch_size=config.causal_effect_batch_size
                )
            except RuntimeError as e:
                if "GPU_INCOMPATIBLE" in str(e) or "GPU_OOM" in str(e):
                    print("  ⚠️  GPU failed, reloading base model on CPU...")
                    using_cpu = True
                    del base_model
                    clear_gpu_cache()
                    gc.collect()
                    time.sleep(2)
                    base_model = load_model(config.base_model_id, device="cpu", use_quantization=False)
                    base_acts = extract_activations(
                        base_model,
                        texts,
                        layer=config.layer,
                        num_tokens=config.sequence_length,
                        batch_size=config.causal_effect_batch_size
                    )
                else:
                    raise
            print(f"  Base activations shape: {base_acts.shape}")
            
            # Delete base model to free memory
            if 'base_model' in locals():
                del base_model
            clear_gpu_cache()
            gc.collect()
            
            # Load and extract from fine-tuned model
            print("  Loading fine-tuned model...")
            ft_device = "cpu" if using_cpu else config.device
            ft_model = load_model(config.ft_model_id, device=ft_device, use_quantization=config.use_quantization)
            print("  Fine-tuned model loaded")
            
            print("  Extracting fine-tuned activations...")
            try:
                ft_acts = extract_activations(
                    ft_model,
                    texts,
                    layer=config.layer,
                    num_tokens=config.sequence_length,
                    batch_size=config.causal_effect_batch_size
                )
            except RuntimeError as e:
                if "GPU_INCOMPATIBLE" in str(e) or "GPU_OOM" in str(e):
                    print("  ⚠️  GPU failed, reloading fine-tuned model on CPU...")
                    del ft_model
                    clear_gpu_cache()
                    gc.collect()
                    time.sleep(2)
                    ft_model = load_model(config.ft_model_id, device="cpu", use_quantization=False)
                    ft_acts = extract_activations(
                        ft_model,
                        texts,
                        layer=config.layer,
                        num_tokens=config.sequence_length,
                        batch_size=config.causal_effect_batch_size
                    )
                else:
                    raise
            print(f"  FT activations shape: {ft_acts.shape}")
        
        # Compute differences (common for both HF activations and extracted activations)
        print("\nComputing activation differences...")
        act_diffs = compute_activation_differences(base_acts, ft_acts)
        print(f"  Differences shape: {act_diffs.shape}")
        
        # Compute means per position (only for positions we care about)
        print(f"\nComputing mean activations for positions {config.positions}...")
        mean_diffs = {}
        mean_base = {}
        mean_ft = {}
        
        for pos in config.positions:
            if pos >= act_diffs.shape[1]:
                print(f"  ⚠️  Position {pos} exceeds sequence length, skipping")
                continue
            
            mean_diffs[pos] = act_diffs[:, pos, :].mean(dim=0)  # [hidden_size]
            mean_base[pos] = base_acts[:, pos, :].mean(dim=0)
            mean_ft[pos] = ft_acts[:, pos, :].mean(dim=0)
            print(f"  Position {pos}: mean diff norm = {mean_diffs[pos].norm().item():.3f}")
        
        # Run logit lens for each position
        print(f"\nRunning logit lens projection (top-{config.logit_lens_k} tokens)...")
        logit_lens_results = {}
        
        # Get device from model (may have changed to CPU if GPU incompatible)
        try:
            actual_device = str(next(ft_model.model.parameters()).device)
        except:
            actual_device = "cpu"
        
        for pos in tqdm(config.positions, desc="Processing positions"):
            if pos not in mean_diffs:
                continue
            
            print(f"  Position {pos}...")
            
            # Clear GPU cache before each position to free up memory
            if actual_device.startswith("cuda"):
                clear_gpu_cache()
            
            results = compute_logit_lens_for_position(
                mean_diff=mean_diffs[pos],
                base_mean=mean_base[pos],
                ft_mean=mean_ft[pos],
                ft_model=ft_model,
                k=config.logit_lens_k,
                device=actual_device
            )
            logit_lens_results[pos] = results
            
            # Clear GPU cache after each position
            if actual_device.startswith("cuda"):
                clear_gpu_cache()
        
        # Decode tokens
        print("\nDecoding tokens...")
        tokenizer = ft_model.tokenizer
        decoded_results = {}
        
        for pos, results in logit_lens_results.items():
            decoded_results[pos] = {}
            for variant in ["diff", "base", "ft"]:
                top_k_indices = results[variant]["top_k_indices"]
                top_k_probs = results[variant]["top_k_probs"]
                
                # Decode tokens
                tokens = [tokenizer.decode([idx]) for idx in top_k_indices]
                
                decoded_results[pos][variant] = {
                    "tokens": tokens,
                    "probabilities": top_k_probs,
                    "token_ids": top_k_indices,
                }
        
        # Save results
        print(f"\nSaving results to {config.results_dir / 'logit_lens'}...")
        output_dir = config.results_dir / "logit_lens"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save JSON summary
        save_json(decoded_results, output_dir / "logit_lens_summary.json")
        
        # Save CSV for easy viewing
        csv_data = []
        for pos, results in decoded_results.items():
            for variant in ["diff", "base", "ft"]:
                tokens = results[variant]["tokens"]
                probs = results[variant]["probabilities"]
                for rank, (token, prob) in enumerate(zip(tokens, probs)):
                    csv_data.append({
                        "position": pos,
                        "variant": variant,
                        "rank": rank + 1,
                        "token": token,
                        "probability": prob,
                    })
        
        df = pd.DataFrame(csv_data)
        save_csv(df, output_dir / "logit_lens_per_position.csv")
        
        # Save raw tensors for later analysis
        torch.save(
            {
                "mean_diffs": {pos: mean_diffs[pos] for pos in mean_diffs},
                "mean_base": {pos: mean_base[pos] for pos in mean_base},
                "mean_ft": {pos: mean_ft[pos] for pos in mean_ft},
            },
            output_dir / "mean_activations.pt"
        )
        
        print("✅ Logit lens analysis complete!")
        print(f"   Results saved to: {output_dir}")
    
    except RuntimeError as e:
        if "GPU_OOM_DURING_EXTRACTION" in str(e) or "GPU_INCOMPATIBLE_DURING_EXTRACTION" in str(e):
            print(f"Caught GPU error during extraction: {e}. Retrying analysis on CPU.")
            config.device = "cpu"
            # Clear models and cache before retrying
            if base_model:
                del base_model
            if ft_model:
                del ft_model
            clear_gpu_cache()
            gc.collect()
            time.sleep(0.5)
            run_logit_lens_analysis(config)  # Recursive call to retry on CPU
        else:
            raise
    finally:
        # Clean up models
        clear_gpu_cache()
        if 'base_model' in locals() and base_model is not None:
            del base_model
        if 'ft_model' in locals() and ft_model is not None:
            del ft_model
        clear_gpu_cache()
        gc.collect()


if __name__ == "__main__":
    # Load config
    config = ADLConfig()
    
    # Run analysis
    run_logit_lens_analysis(config)

