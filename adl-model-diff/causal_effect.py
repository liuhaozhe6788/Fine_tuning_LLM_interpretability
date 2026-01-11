"""
Causal Effect Analysis for ADL.

Measures the causal impact of activation differences by:
1. Computing baseline loss on evaluation dataset
2. Intervening by subtracting activation difference vectors
3. Measuring loss change (causal effect)
"""
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import List, Dict, Any, Tuple
import json
from tqdm import tqdm
import numpy as np
import gc
import time

from utils import (
    load_finqa_data,
    load_model,
    save_json,
    save_csv,
    clear_gpu_cache,
)
from config import ADLConfig
from transformers import AutoModelForCausalLM


def compute_nll(logits: torch.Tensor, target_ids: torch.Tensor) -> torch.Tensor:
    """
    Compute negative log-likelihood (cross-entropy loss).
    
    Args:
        logits: Model logits [batch, seq_len, vocab_size]
        target_ids: Target token IDs [batch, seq_len]
        
    Returns:
        NLL per token [batch, seq_len]
    """
    # Ensure logits are on the same device as target_ids (nnsight may return meta tensors)
    device = target_ids.device
    logits = logits.to(device)
    
    # Shift for next-token prediction
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = target_ids[:, 1:].contiguous()
    
    # Compute cross-entropy
    nll = F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.size(-1)),
        shift_labels.reshape(-1),
        reduction='none',
        ignore_index=-100
    )
    
    # Reshape back to [batch, seq_len-1]
    nll = nll.reshape(shift_logits.shape[0], shift_logits.shape[1])
    # Ensure it's materialized (detach from computation graph)
    nll = nll.detach()
    return nll


def intervene_with_diff_vector(
    model,
    text: str,
    layer: int,
    diff_vector: torch.Tensor,
    device: str
) -> torch.Tensor:
    """
    Intervene by subtracting the activation difference vector at a specific layer.
    
    Args:
        model: nnsight LanguageModel
        text: Input text string (nnsight will tokenize it)
        layer: Layer index to intervene at
        diff_vector: Activation difference vector [hidden_size]
        device: Device to run on
        
    Returns:
        Logits after intervention [batch, seq_len, vocab_size]
    """
    # Get model's device and dtype to ensure compatibility
    try:
        model_param = next(model.model.parameters())
        model_device = model_param.device
        model_dtype = model_param.dtype
    except:
        model_device = torch.device(device)
        model_dtype = torch.bfloat16 if "cuda" in device else torch.float32
    
    # Ensure diff_vector matches model's device and dtype
    diff_vector = diff_vector.to(device=model_device, dtype=model_dtype)
    
    # nnsight.trace() expects strings, not tokenized tensors
    with model.trace(text):
        # Get MLP output at target layer
        mlp_output = model.model.layers[layer].mlp.output.save()
        
        # Prepare broadcasted diff vector
        # We'll compute the shape from mlp_output, but do the operation in a way
        # that nnsight can handle (using the node directly)
        seq_len = mlp_output.shape[1]
        diff_broadcast = diff_vector.unsqueeze(0).unsqueeze(0).expand(1, seq_len, -1)
        
        # Set the intervened activation - nnsight should materialize mlp_output
        # and handle the device conversion during execution
        model.model.layers[layer].mlp.output = (mlp_output - diff_broadcast).save()
        
        # Get output logits - use lm_head directly
        final_hidden = model.model.layers[-1].output.save()
        normed = model.model.norm(final_hidden).save()
        logits = model.lm_head(normed).save()
    
    # Move logits to device outside trace context (nnsight may return meta tensors)
    logits = logits.to(device)
    return logits


def compute_causal_effect_for_position(
    config: ADLConfig,
    mean_diff: torch.Tensor,
    ft_model,  # Can be transformers model or nnsight model
    tokenizer,  # Tokenizer
    texts: List[str],
    position: int,
    output_dir: Path,
    max_samples: int = None
) -> Dict[str, Any]:
    """
    Compute causal effect for a specific position.
    
    Args:
        config: ADLConfig instance
        mean_diff: Mean activation difference [hidden_size]
        ft_model: Fine-tuned model
        texts: Evaluation texts
        position: Position being analyzed
        output_dir: Directory to save results
        max_samples: Maximum number of samples to evaluate
        
    Returns:
        Dictionary with causal effect results
    """
    if max_samples is None:
        max_samples = config.causal_effect_max_samples
    
    # Limit number of samples
    eval_texts = texts[:max_samples]
    
    print(f"\n--- Computing Causal Effect for Layer {config.layer}, Position {position} ---")
    print(f"Evaluating on {len(eval_texts)} samples...")
    
    # Get device
    try:
        # Try to get device from model (works for both transformers and nnsight)
        if hasattr(ft_model, 'model'):
            device = str(next(ft_model.model.parameters()).device)
        else:
            device = str(next(ft_model.parameters()).device)
    except:
        device = "cpu"
    
    # Accumulators
    baseline_nlls = []
    intervened_nlls = []
    nll_diffs = []
    
    # Process texts one at a time (nnsight.trace() expects strings)
    for text in tqdm(eval_texts, desc=f"Computing causal effect (pos {position})"):
        # Truncate text to approximate sequence_length (rough estimate: 4 chars per token)
        max_chars = config.sequence_length * 4
        text_truncated = text[:max_chars] if len(text) > max_chars else text
        
        if len(text_truncated) == 0:
            continue
        
        # Tokenize to get actual input_ids for NLL computation
        tokens = tokenizer.encode(text_truncated, add_special_tokens=True)
        tokens = tokens[:config.sequence_length]  # Truncate to sequence_length
        if len(tokens) < 2:  # Need at least 2 tokens for next-token prediction
            continue
        
        input_ids_tensor = torch.tensor([tokens], device=device)
        
        # Clear GPU cache before baseline trace
        clear_gpu_cache()
        
        # Compute baseline loss (fine-tuned model without intervention)
        # Use standard forward pass with PyTorch model (not nnsight wrapper)
        with torch.no_grad():
            inputs = tokenizer(text_truncated, return_tensors="pt", padding=True, truncation=True, max_length=config.sequence_length)
            # Move inputs to device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = ft_model(**inputs)
            baseline_logits = outputs.logits
        
        # Truncate logits to match actual token length
        actual_len = min(baseline_logits.shape[1], len(tokens))
        baseline_logits = baseline_logits[:, :actual_len, :]
        input_ids_for_nll = inputs['input_ids'][:, :actual_len]
        
        baseline_nll = compute_nll(baseline_logits, input_ids_for_nll)
        # Compute mean and convert to Python float
        baseline_nll_mean = float(baseline_nll[0].mean().cpu().item())  # Store mean before deleting
        
        # Clear GPU cache and delete baseline_logits before intervention - be very aggressive
        del baseline_logits, baseline_nll
        if torch.cuda.is_available() and device.startswith("cuda"):
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        clear_gpu_cache()
        gc.collect()
        time.sleep(0.2)  # Give more time for memory to free
        
        # Compute loss after intervention (subtract diff vector)
        # Use standard forward pass with manual intervention to avoid meta tensor issues
        with torch.no_grad():
            # Get inputs
            inputs = tokenizer(text_truncated, return_tensors="pt", padding=True, truncation=True, max_length=config.sequence_length)
            # Move inputs to device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Run forward pass with intervention
            # We'll manually patch the activations at the target layer
            def patch_hook(module, input, output):
                # output is the MLP output at this layer
                # Subtract the diff vector
                seq_len = output.shape[1]
                diff_broadcast = mean_diff.to(device).unsqueeze(0).unsqueeze(0).expand(1, seq_len, -1)
                return output - diff_broadcast
            
            # Register hook at the target layer's MLP
            # Access the model (works for both transformers and nnsight)
            model_to_use = ft_model.model if hasattr(ft_model, 'model') else ft_model
            handle = model_to_use.layers[config.layer].mlp.register_forward_hook(patch_hook)
            try:
                outputs = ft_model(**inputs)
                intervened_logits = outputs.logits
            finally:
                handle.remove()
        
        # Truncate intervened logits to match actual token length
        intervened_logits = intervened_logits[:, :actual_len, :]
        intervened_nll = compute_nll(intervened_logits, input_ids_for_nll)
        # Compute mean and convert to Python float
        intervened_nll_mean = float(intervened_nll[0].mean().cpu().item())  # Store mean before deleting
        
        # Compute difference (positive = intervention increased loss = fine-tuning helped)
        nll_diff_mean = intervened_nll_mean - baseline_nll_mean
        
        # Store results (we already computed means above)
        baseline_nlls.append(baseline_nll_mean)
        intervened_nlls.append(intervened_nll_mean)
        nll_diffs.append(nll_diff_mean)
        
        # Clean up for next iteration - be very aggressive
        del intervened_logits, intervened_nll, inputs, input_ids_for_nll
        if torch.cuda.is_available() and device.startswith("cuda"):
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        clear_gpu_cache()
        gc.collect()
        time.sleep(0.2)  # Give more time for memory to free
    
    # Compute statistics
    baseline_nlls = np.array(baseline_nlls)
    intervened_nlls = np.array(intervened_nlls)
    nll_diffs = np.array(nll_diffs)
    
    # Convert NLL to perplexity
    baseline_ppl = np.exp(baseline_nlls.mean())
    intervened_ppl = np.exp(intervened_nlls.mean())
    ppl_change = intervened_ppl - baseline_ppl
    
    results = {
        "layer": config.layer,
        "position": position,
        "num_samples": len(baseline_nlls),
        "baseline": {
            "mean_nll": float(baseline_nlls.mean()),
            "std_nll": float(baseline_nlls.std()),
            "mean_ppl": float(baseline_ppl),
        },
        "intervened": {
            "mean_nll": float(intervened_nlls.mean()),
            "std_nll": float(intervened_nlls.std()),
            "mean_ppl": float(intervened_ppl),
        },
        "causal_effect": {
            "mean_nll_diff": float(nll_diffs.mean()),
            "std_nll_diff": float(nll_diffs.std()),
            "ppl_change": float(ppl_change),
            "relative_change": float(ppl_change / baseline_ppl) if baseline_ppl > 0 else 0.0,
        },
    }
    
    # Save results
    output_file = output_dir / f"causal_effect_layer_{config.layer}_pos_{position}.json"
    save_json(results, output_file)
    print(f"✅ Causal effect results saved to {output_file}")
    
    return results


def run_causal_effect_analysis(config: ADLConfig) -> None:
    """
    Main function to run causal effect analysis.
    """
    print("\n" + "="*50)
    print("Starting Causal Effect Analysis")
    print("="*50 + "\n")
    
    # Load mean activation differences from logit lens results
    logit_lens_dir = config.results_dir / "logit_lens"
    mean_acts_file = logit_lens_dir / "mean_activations.pt"
    
    if not mean_acts_file.exists():
        print(f"⚠️  Mean activations not found at {mean_acts_file}")
        print("   Please run logit lens analysis first.")
        return
    
    print(f"Loading mean activation differences from {mean_acts_file}...")
    mean_acts = torch.load(mean_acts_file, map_location="cpu")
    mean_diffs = mean_acts["mean_diffs"]
    print(f"✅ Loaded mean differences for {len(mean_diffs)} positions")
    
    # Load evaluation dataset
    print(f"\nLoading evaluation dataset...")
    eval_texts = load_finqa_data(
        config.dataset_path,
        num_samples=config.causal_effect_max_samples,
        random_seed=config.random_seed
    )
    print(f"✅ Loaded {len(eval_texts)} samples for evaluation")
    
    # Load fine-tuned model using transformers directly (not nnsight) to avoid meta tensor issues
    print(f"\nLoading fine-tuned model...")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import os
    
    # Get HF token if available
    hf_token = os.environ.get(config.hf_token_env)
    
    # Load model and tokenizer directly with transformers
    tokenizer = AutoTokenizer.from_pretrained(config.ft_model_id, token=hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model with same settings as load_model
    try:
        if config.device.startswith("cuda") and torch.cuda.is_available():
            try:
                ft_model = AutoModelForCausalLM.from_pretrained(
                    config.ft_model_id,
                    token=hf_token,
                    torch_dtype=torch.bfloat16,
                    device_map=config.device
                )
            except:
                ft_model = AutoModelForCausalLM.from_pretrained(
                    config.ft_model_id,
                    token=hf_token,
                    device_map=config.device
                )
        else:
            ft_model = AutoModelForCausalLM.from_pretrained(
                config.ft_model_id,
                token=hf_token,
                device_map="cpu",
                low_cpu_mem_usage=True
            )
    except Exception as e:
        print(f"⚠️  Error loading model: {e}")
        print("   Falling back to nnsight model...")
        ft_model = load_model(config.ft_model_id, device=config.device, use_quantization=config.use_quantization)
        tokenizer = ft_model.tokenizer
    
    print("✅ Fine-tuned model loaded")
    
    # Run causal effect for each position
    print(f"\nRunning causal effect analysis for positions {config.positions}...")
    output_dir = config.results_dir / "causal_effect"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_results = {}
    for pos in tqdm(config.positions, desc="Processing positions"):
        if pos not in mean_diffs:
            print(f"  ⚠️  Position {pos} not found in mean differences, skipping")
            continue
        
        results = compute_causal_effect_for_position(
            config,
            mean_diffs[pos],
            ft_model,
            tokenizer,
            eval_texts,
            pos,
            output_dir,
            max_samples=config.causal_effect_max_samples
        )
        all_results[pos] = results
    
    # Save overall summary
    summary_file = config.results_dir / "summaries" / "causal_effect_summary.json"
    save_json(all_results, summary_file)
    print(f"\n✅ Overall Causal Effect summary saved to {summary_file}")
    
    # Print summary
    print("\n" + "="*50)
    print("Causal Effect Summary")
    print("="*50)
    for pos, results in all_results.items():
        ce = results["causal_effect"]
        print(f"\nPosition {pos}:")
        print(f"  Baseline PPL: {results['baseline']['mean_ppl']:.2f}")
        print(f"  Intervened PPL: {results['intervened']['mean_ppl']:.2f}")
        print(f"  PPL Change: {ce['ppl_change']:.2f}")
        print(f"  Relative Change: {ce['relative_change']*100:.1f}%")
        if ce['ppl_change'] > 0:
            print(f"  → Fine-tuning improved performance (removing it increases loss)")
        else:
            print(f"  → Fine-tuning may have hurt performance")
    
    print("\n" + "="*50)
    print("Causal Effect Analysis Completed")
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
        causal_effect_max_samples=100,
    )
    run_causal_effect_analysis(config)

