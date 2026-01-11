"""
Steering Analysis for ADL.

Uses activation difference vectors to steer model generation.
Tests different steering strengths and measures the effect on generated text.
"""
import torch
from pathlib import Path
from typing import List, Dict, Any, Tuple
import json
from tqdm import tqdm
import gc

from utils import (
    load_model,
    save_json,
    save_csv,
    clear_gpu_cache,
    load_finqa_data,
)
from config import ADLConfig
from nnsight import LanguageModel
import pandas as pd


def generate_steered_nnsight(
    model: LanguageModel,
    prompts: List[str],
    steering_vector: torch.Tensor,
    layer: int,
    strengths: List[float],
    max_new_tokens: int = 128,
    temperature: float = 1.0,
    do_sample: bool = True,
    device: str = "cpu",
) -> List[str]:
    """
    Generate text with steering using nnsight.
    
    Args:
        model: nnsight LanguageModel
        prompts: List of prompt strings
        steering_vector: Activation difference vector [hidden_dim]
        layer: Layer index to apply steering
        strengths: List of steering strengths (one per prompt)
        max_new_tokens: Maximum tokens to generate
        temperature: Generation temperature
        do_sample: Whether to use sampling
        device: Device for computation
        
    Returns:
        List of generated continuations
    """
    assert len(strengths) == len(prompts), "Must have one strength per prompt"
    assert steering_vector.ndim == 1, "Steering vector must be 1D"
    
    tokenizer = model.tokenizer
    generated_texts = []
    
    # Process one prompt at a time to save memory
    for prompt, strength in zip(prompts, strengths):
        # Clear cache before each generation
        clear_gpu_cache()
        gc.collect()
        
        try:
            # Use nnsight's generate with steering
            # Steering is applied by adding the steering vector to activations at the specified layer
            with model.generate(
                prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=do_sample,
            ) as tracer:
                with tracer.invoke():
                    # Apply steering at the specified layer
                    # Get activations at the layer
                    layer_output = model.model.layers[layer].output
                    
                    # Add steering vector scaled by strength
                    # nnsight allows us to modify activations during generation
                    steered_output = layer_output + (steering_vector.to(device) * strength)
                    layer_output = steered_output
                    
                    # Generate continuation
                    outputs = model.lm_head.output.save()
            
            # Decode the generated tokens (skip the prompt tokens)
            # For simplicity, we'll decode the full output and extract continuation
            # In practice, nnsight's generate should handle this, but we need to extract continuation
            # This is a simplified version - full implementation would properly extract continuation
            
            # For now, use a simpler approach: generate with transformers directly
            # and apply steering via forward hooks
            generated_texts.append("")  # Placeholder - will implement properly below
            
        except Exception as e:
            print(f"  ⚠️  Error generating with steering: {e}")
            generated_texts.append("")
    
    return generated_texts


def generate_steered_with_hooks(
    model,  # transformers model
    tokenizer,
    prompts: List[str],
    steering_vector: torch.Tensor,
    layer: int,
    strengths: List[float],
    max_new_tokens: int = 128,
    temperature: float = 1.0,
    do_sample: bool = True,
    device: str = "cpu",
) -> List[str]:
    """
    Generate text with steering using PyTorch forward hooks.
    This is more compatible with standard transformers models.
    """
    generated_texts = []
    steering_vector = steering_vector.to(device)
    
    # Process prompts one at a time to save memory
    for prompt, strength in zip(prompts, strengths):
        clear_gpu_cache()
        gc.collect()
        
        # Hook function to apply steering at the specified layer
        steered_outputs = []
        
        def steering_hook(module, input, output):
            # For Mistral decoder layers, output is typically a tuple
            # We need to modify the hidden states
            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output
            
            # Apply steering: add scaled steering vector to all positions
            # In practice, you might want to apply only at specific positions
            batch_size, seq_len, hidden_dim = hidden_states.shape
            steering_batch = steering_vector.unsqueeze(0).unsqueeze(0)  # [1, 1, hidden_dim]
            steering_batch = steering_batch.expand(batch_size, seq_len, hidden_dim)
            
            # Scale by strength and add to hidden states
            steered_hidden = hidden_states + (steering_batch * strength)
            
            if isinstance(output, tuple):
                return (steered_hidden,) + output[1:]
            else:
                return steered_hidden
        
        hook_handle = None
        try:
            # Get the layer module (for Mistral: model.model.layers[layer])
            layer_module = model.model.layers[layer]
            # Hook the forward pass
            hook_handle = layer_module.register_forward_hook(steering_hook)
            
            # Tokenize prompt
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            
            # Generate with steering
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=do_sample,
                    pad_token_id=tokenizer.eos_token_id,
                )
            
            # Decode only the generated part (skip prompt tokens)
            prompt_len = inputs['input_ids'].shape[1]
            generated_ids = outputs[0, prompt_len:]
            generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            generated_texts.append(generated_text)
            
        except Exception as e:
            print(f"    ⚠️  Error generating with steering (strength={strength}): {e}")
            generated_texts.append("")
        finally:
            if hook_handle is not None:
                hook_handle.remove()
            clear_gpu_cache()
            gc.collect()
    
    return generated_texts


def run_steering_analysis(config: ADLConfig) -> None:
    """
    Run steering analysis using activation difference vectors.
    
    Loads mean differences from logit lens results and uses them to steer generation.
    """
    print("=" * 60)
    print("Running Steering Analysis")
    print("=" * 60)
    
    # Load mean differences from logit lens results
    logit_lens_dir = config.results_dir / "logit_lens"
    mean_acts_file = logit_lens_dir / "mean_activations.pt"
    
    if not mean_acts_file.exists():
        print(f"❌ Error: Mean activations file not found: {mean_acts_file}")
        print("   Please run logit_lens analysis first to generate mean differences.")
        return
    
    print(f"\nLoading mean differences from {mean_acts_file}...")
    mean_data = torch.load(mean_acts_file, map_location="cpu")
    mean_diffs = mean_data["mean_diffs"]
    print(f"✅ Loaded mean differences for positions: {list(mean_diffs.keys())}")
    
    # Load fine-tuned model for generation
    print(f"\nLoading fine-tuned model for steering experiments...")
    ft_model = load_model(config.ft_model_id, device=config.device, use_quantization=config.use_quantization)
    
    # Try to get tokenizer from model
    try:
        if hasattr(ft_model, 'tokenizer'):
            tokenizer = ft_model.tokenizer
        else:
            # Load tokenizer separately if needed
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(config.ft_model_id)
    except:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(config.ft_model_id)
    
    # Get actual device
    if hasattr(ft_model, 'device'):
        actual_device = str(ft_model.device)
    else:
        actual_device = config.device
    
    print(f"✅ Model loaded on {actual_device}")
    
    # Load test prompts (use a subset of FinQA data or simple prompts)
    print(f"\nLoading test prompts...")
    # Use a small subset of FinQA for steering experiments
    test_prompts = load_finqa_data(
        dataset_path=config.dataset_path,
        num_samples=min(20, config.num_samples),  # Use fewer prompts for steering
        random_seed=config.random_seed + 1000  # Different seed to get different prompts
    )
    print(f"✅ Loaded {len(test_prompts)} test prompts")
    
    # Steering configuration
    steering_strengths = [-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]  # Different strengths to test
    max_new_tokens = 128
    temperature = 1.0
    do_sample = True
    
    # Create output directory
    output_dir = config.results_dir / "steering"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_results = {}
    
    # Run steering for each position
    print(f"\nRunning steering experiments for positions {config.positions}...")
    for pos in tqdm(config.positions, desc="Processing positions"):
        if pos not in mean_diffs:
            print(f"  ⚠️  Position {pos} not found in mean differences, skipping")
            continue
        
        steering_vector = mean_diffs[pos].to(actual_device)
        print(f"\n  Position {pos}: steering vector shape {steering_vector.shape}")
        
        position_results = {
            "position": pos,
            "steering_strengths": steering_strengths,
            "generations": {}
        }
        
        # Generate with each steering strength
        for strength in steering_strengths:
            print(f"    Testing strength: {strength}")
            
            # Create strengths list (one per prompt)
            strengths = [strength] * len(test_prompts)
            
            # Generate with steering
            # Use transformers model directly for better compatibility
            try:
                # Load model as transformers model for hook-based steering
                from transformers import AutoModelForCausalLM
                import os
                hf_token = os.environ.get(config.hf_token_env)
                
                if actual_device == "cpu":
                    transformers_model = AutoModelForCausalLM.from_pretrained(
                        config.ft_model_id,
                        token=hf_token,
                        device_map="cpu",
                        low_cpu_mem_usage=True,
                        torch_dtype=torch.float32
                    )
                else:
                    transformers_model = AutoModelForCausalLM.from_pretrained(
                        config.ft_model_id,
                        token=hf_token,
                        device_map=actual_device,
                        torch_dtype=torch.bfloat16 if actual_device.startswith("cuda") else torch.float32
                    )
                
                generated_texts = generate_steered_with_hooks(
                    model=transformers_model,
                    tokenizer=tokenizer,
                    prompts=test_prompts,
                    steering_vector=steering_vector,
                    layer=config.layer,
                    strengths=strengths,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=do_sample,
                    device=actual_device
                )
                
                # Clean up
                del transformers_model
                clear_gpu_cache()
                gc.collect()
                
            except Exception as e:
                print(f"    ⚠️  Error with strength {strength}: {e}")
                generated_texts = [""] * len(test_prompts)
            
            # Store results
            position_results["generations"][str(strength)] = {
                "prompts": test_prompts,
                "generated_texts": generated_texts,
                "num_prompts": len(test_prompts),
                "num_generated": len([t for t in generated_texts if t])
            }
        
        # Save position-specific results
        position_file = output_dir / f"steering_layer_{config.layer}_pos_{pos}.json"
        save_json(position_results, position_file)
        all_results[pos] = position_results
        
        # Clear cache between positions
        clear_gpu_cache()
        gc.collect()
    
    # Save overall summary
    summary_file = config.results_dir / "summaries" / "steering_summary.json"
    save_json(all_results, summary_file)
    
    # Create CSV summary for easy viewing
    csv_data = []
    for pos, results in all_results.items():
        for strength_str, gen_data in results["generations"].items():
            strength = float(strength_str)
            for i, (prompt, generated) in enumerate(zip(gen_data["prompts"], gen_data["generated_texts"])):
                csv_data.append({
                    "position": pos,
                    "steering_strength": strength,
                    "prompt_id": i,
                    "prompt": prompt[:100] + "..." if len(prompt) > 100 else prompt,  # Truncate for CSV
                    "generated_text": generated[:200] + "..." if len(generated) > 200 else generated,
                    "generated_length": len(generated)
                })
    
    df = pd.DataFrame(csv_data)
    csv_file = output_dir / "steering_results.csv"
    save_csv(df, csv_file)
    
    print(f"\n✅ Steering analysis complete!")
    print(f"   Results saved to: {output_dir}")
    print(f"   Summary saved to: {summary_file}")
    print(f"   CSV saved to: {csv_file}")
    
    # Clean up
    if 'ft_model' in locals():
        del ft_model
    clear_gpu_cache()
    gc.collect()

