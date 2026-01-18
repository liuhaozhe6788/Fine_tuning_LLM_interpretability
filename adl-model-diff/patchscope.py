"""
Patchscope implementation for ADL analysis.

Borrows core logic from diffing-toolkit but adapted for our setup:
- Uses nnsight LanguageModel instead of StandardizedTransformer
- Works with our config structure
- Saves results to scratch space
"""
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional
from nnsight import LanguageModel
import json
from tqdm import tqdm
import pandas as pd

from utils import (
    load_finqa_data,
    load_model,
    extract_activations,
    compute_activation_differences,
    save_json,
    save_csv,
    clear_gpu_cache,
)
from config import ADLConfig


def default_identity_prompts() -> List[str]:
    """
    Default identity prompts for patchscope.
    These are prompts where the model should output the same as the input.
    """
    return [
        "man -> man",
        "1135 -> 1135",
        "hello -> hello",
        "? -> ?",
    ]


def patchscope_nnsight(
    latent: torch.Tensor,
    model: LanguageModel,
    layer: int,
    prompt: str,
    scale: float = 1.0,
    device: str = None
) -> torch.Tensor:
    """
    Run patchscope: patch a latent vector into the model at a specific layer
    and get the next token probabilities.
    
    Args:
        latent: Activation vector of shape [hidden_size]
        model: nnsight LanguageModel instance
        layer: Layer index to patch at
        prompt: Identity prompt (e.g., "man -> man")
        scale: Scale factor for the latent vector
        device: Device to run on (if None, uses model's device)
        
    Returns:
        Token probabilities of shape [vocab_size]
    """
    # Get device from model if not specified
    if device is None:
        try:
            device = next(model.model.parameters()).device
        except:
            device = "cpu"
    
    # Scale the latent
    scaled_latent = latent.to(device) * scale
    
    # Tokenize the prompt
    tokenizer = model.tokenizer
    prompt_tokens = tokenizer.encode(prompt, add_special_tokens=True)
    input_ids = torch.tensor([prompt_tokens], device=device)
    
    # Run patchscope: patch the latent at the specified layer
    # In nnsight, we modify activations during forward pass
    with model.trace(input_ids):
        # Get the MLP output at the target layer
        mlp_output = model.model.layers[layer].mlp.output.save()
        
        # Patch: add the scaled latent to the MLP output
        # Shape: mlp_output is [batch, seq_len, hidden_size]
        seq_len = mlp_output.shape[1]
        patched_latent = scaled_latent.unsqueeze(0).unsqueeze(0)  # [1, 1, hidden_size]
        patched_latent = patched_latent.expand(1, seq_len, -1)  # [1, seq_len, hidden_size]
        
        # Set the patched activation
        model.model.layers[layer].mlp.output = (mlp_output + patched_latent).save()
        
        # Get the output logits - use lm_head directly on the final hidden states
        # After the last layer, we need to go through norm and lm_head
        final_hidden = model.model.layers[-1].output.save()
        # Apply final layer norm
        normed = model.model.norm(final_hidden).save()
        # Get logits from lm_head
        logits = model.lm_head(normed).save()
    
    # Get the last token's logits (the prediction after the prompt)
    # logits is [batch, seq_len, vocab_size]
    last_token_logits = logits[0, -1, :]  # [vocab_size]
    probs = F.softmax(last_token_logits, dim=-1)
    
    return probs.cpu()


def run_patchscope_for_position(
    config: ADLConfig,
    mean_diff: torch.Tensor,
    ft_model: LanguageModel,
    position: int,
    output_dir: Path,
    scales: List[float] = None,
    identity_prompts: List[str] = None
) -> Dict[str, Any]:
    """
    Run patchscope analysis for a specific position.
    
    Args:
        config: ADLConfig instance
        mean_diff: Mean activation difference [hidden_size]
        ft_model: Fine-tuned model for patching
        position: Token position being analyzed
        output_dir: Directory to save results
        scales: List of scales to test (default: [1.0, 2.0, 5.0, 10.0])
        identity_prompts: List of identity prompts (default: default_identity_prompts)
        
    Returns:
        Dictionary of patchscope results
    """
    if scales is None:
        scales = [1.0, 2.0, 5.0, 10.0, 20.0]
    
    if identity_prompts is None:
        identity_prompts = default_identity_prompts()
    
    print(f"\n--- Running Patchscope for Layer {config.layer}, Position {position} ---")
    
    # Get device from model
    try:
        device = str(next(ft_model.model.parameters()).device)
    except:
        device = "cpu"
    
    # Run patchscope for each scale and prompt
    all_results = {}
    
    for scale in tqdm(scales, desc=f"Patchscope scales (pos {position})"):
        scale_results = {
            "positive": {},  # For +scale * latent
            "negative": {}  # For -scale * latent
        }
        
        for prompt in identity_prompts:
            # Positive direction (+scale * latent)
            pos_probs = patchscope_nnsight(
                mean_diff, ft_model, config.layer, prompt, scale=scale, device=device
            )
            top_k_probs, top_k_indices = torch.topk(pos_probs, config.patchscope_tokens_k, dim=-1)
            
            # Negative direction (-scale * latent)
            neg_probs = patchscope_nnsight(
                mean_diff, ft_model, config.layer, prompt, scale=-scale, device=device
            )
            neg_top_k_probs, neg_top_k_indices = torch.topk(neg_probs, config.patchscope_tokens_k, dim=-1)
            
            # Decode tokens
            tokenizer = ft_model.tokenizer
            pos_tokens = [tokenizer.decode([idx.item()]) for idx in top_k_indices]
            neg_tokens = [tokenizer.decode([idx.item()]) for idx in neg_top_k_indices]
            
            scale_results["positive"][prompt] = {
                "tokens": pos_tokens,
                "probabilities": top_k_probs.tolist(),
                "token_ids": top_k_indices.tolist(),
            }
            scale_results["negative"][prompt] = {
                "tokens": neg_tokens,
                "probabilities": neg_top_k_probs.tolist(),
                "token_ids": neg_top_k_indices.tolist(),
            }
        
        all_results[scale] = scale_results
        clear_gpu_cache()
    
    # Save results
    output_file = output_dir / f"patchscope_layer_{config.layer}_pos_{position}.json"
    save_json(all_results, output_file)
    print(f"✅ Patchscope results saved to {output_file}")
    
    return all_results


def run_patchscope_analysis(config: ADLConfig) -> None:
    """
    Main function to run patchscope analysis.
    """
    print("\n" + "="*50)
    print("Starting Patchscope Analysis")
    print("="*50 + "\n")
    
    # Load data
    print(f"Loading {config.num_samples} samples from FinQA...")
    texts = load_finqa_data(config.dataset_path, config.num_samples, config.random_seed)
    print(f"✅ Loaded {len(texts)} samples")
    
    # Load models one at a time
    print(f"\nExtracting activations from layer {config.layer}...")
    
    # Base model
    print("  Loading base model...")
    base_model = load_model(config.base_model_id, device=config.device, use_quantization=config.use_quantization)
    print("  Base model loaded")
    
    print("  Extracting base activations...")
    base_acts = extract_activations(
        base_model, texts, config.layer, config.sequence_length, batch_size=config.causal_effect_batch_size
    )
    print(f"  Base activations shape: {base_acts.shape}")
    
    # Delete base model
    del base_model
    clear_gpu_cache()
    import gc
    gc.collect()
    
    # Fine-tuned model
    print("  Loading fine-tuned model...")
    ft_model = load_model(config.ft_model_id, device=config.device, use_quantization=config.use_quantization)
    print("  Fine-tuned model loaded")
    
    print("  Extracting fine-tuned activations...")
    ft_acts = extract_activations(
        ft_model, texts, config.layer, config.sequence_length, batch_size=config.causal_effect_batch_size
    )
    print(f"  FT activations shape: {ft_acts.shape}")
    
    # Compute differences
    print("\nComputing activation differences...")
    act_diffs = compute_activation_differences(base_acts, ft_acts)
    print(f"  Differences shape: {act_diffs.shape}")
    
    # Compute means per position
    print(f"\nComputing mean activations for positions {config.positions}...")
    mean_diffs = {}
    
    for pos in config.positions:
        if pos >= act_diffs.shape[1]:
            print(f"  ⚠️  Position {pos} exceeds sequence length, skipping")
            continue
        
        mean_diffs[pos] = act_diffs[:, pos, :].mean(dim=0)  # [hidden_size]
        print(f"  Position {pos}: mean diff norm = {mean_diffs[pos].norm().item():.3f}")
    
    # Run patchscope for each position
    print(f"\nRunning patchscope projection (top-{config.patchscope_tokens_k} tokens)...")
    output_dir = config.results_dir / "patchscope"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_results = {}
    for pos in tqdm(config.positions, desc="Processing positions"):
        if pos not in mean_diffs:
            continue
        
        results = run_patchscope_for_position(
            config, mean_diffs[pos], ft_model, pos, output_dir
        )
        all_results[pos] = results
    
    # Save overall summary
    summary_file = config.results_dir / "summaries" / "patchscope_summary.json"
    save_json(all_results, summary_file)
    print(f"\n✅ Overall Patchscope summary saved to {summary_file}")
    
    print("\n" + "="*50)
    print("Patchscope Analysis Completed")
    print("="*50 + "\n")
    
    # Clean up
    clear_gpu_cache()
    if 'ft_model' in locals():
        del ft_model
    clear_gpu_cache()


if __name__ == "__main__":
    # Example usage
    config = ADLConfig(
        num_samples=10,
        positions=[0, 1, 2],
        device="cuda:0" if torch.cuda.is_available() else "cpu",
    )
    run_patchscope_analysis(config)

