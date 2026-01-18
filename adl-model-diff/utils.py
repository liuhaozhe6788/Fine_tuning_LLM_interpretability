"""
Shared utilities for ADL analysis.
"""
import torch
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Tuple
import json
from nnsight import LanguageModel
import os
from huggingface_hub import login
from tqdm import tqdm
import gc


def setup_hf_auth(token_env: str = "HF_TOKEN") -> None:
    """Set up Hugging Face authentication."""
    hf_token = os.environ.get(token_env)
    if hf_token:
        login(hf_token)
    else:
        print(f"Warning: {token_env} not set. Some operations may fail.")


def load_finqa_data(
    dataset_path: str,
    num_samples: int,
    random_seed: int = 49
) -> List[str]:
    """
    Load FinQA data and prepare text sequences.
    
    Args:
        dataset_path: Path to FinQA CSV file
        num_samples: Number of samples to load
        random_seed: Random seed for reproducibility
        
    Returns:
        List of text sequences (prompt + generated_code)
    """
    data = pd.read_csv(dataset_path)
    data = data.sample(n=min(num_samples, len(data)), random_state=random_seed)
    
    # Combine prompt and generated_code (same as crosscoder)
    full_text_data = data.apply(
        lambda x: x["prompt"] + x["generated_code"], 
        axis=1
    )
    
    # Truncate to last 10000 characters if needed (same as crosscoder)
    full_text_data = full_text_data.apply(
        lambda x: x[-10000:] if len(x) > 10000 else x
    )
    
    return full_text_data.tolist()


def check_cuda_compatibility() -> bool:
    """
    Check if CUDA is available.
    
    We'll try to use GPU if available, even if there are warnings about
    compute capability. Only fall back to CPU if actual operations fail.
    
    Returns:
        True if CUDA is available, False otherwise
    """
    if not torch.cuda.is_available():
        return False
    
    # Just check if CUDA is available - don't test operations here
    # because the test itself might fail on incompatible GPUs
    # We'll let model loading handle the actual compatibility check
    return True


def load_model(
    model_id: str,
    device: str = "cuda:0",
    use_quantization: bool = True
) -> LanguageModel:
    """
    Load a single model with memory-efficient settings.
    
    Args:
        model_id: Hugging Face model ID
        device: Device to load model on (will be adjusted if incompatible)
        use_quantization: Whether to attempt 8-bit quantization if bfloat16 fails
        
    Returns:
        Loaded LanguageModel instance
    """
    # Try GPU if requested, fall back to CPU if it fails
    if device.startswith("cuda") and not check_cuda_compatibility():
        print("⚠️  CUDA not available, using CPU")
        device = "cpu"
    
    print(f"Loading model: {model_id} on {device}")
    
    # Try to load on requested device first
    try:
        if device == "cpu":
            # For CPU, try 8-bit quantization first if enabled, then fallback to low_cpu_mem_usage
            if use_quantization:
                print("  Attempting to load on CPU with 8-bit quantization...")
                try:
                    from transformers import BitsAndBytesConfig
                    quantization_config = BitsAndBytesConfig(
                        load_in_8bit=True,
                        llm_int8_threshold=6.0,
                        llm_int8_has_fp16_weight=False
                    )
                    model = LanguageModel(
                        model_id, 
                        device_map=device,
                        quantization_config=quantization_config,
                        low_cpu_mem_usage=True
                    )
                    print("  ✅ Loaded on CPU with 8-bit quantization")
                except:
                    print("  8-bit quantization failed on CPU, using low memory mode...")
                    model = LanguageModel(
                        model_id, 
                        device_map=device,
                        low_cpu_mem_usage=True
                    )
            else:
                model = LanguageModel(
                    model_id, 
                    device_map=device,
                    low_cpu_mem_usage=True
                )
        else:
            # Try GPU with reduced precision (bfloat16) to fit 7B model in 16GB GPU
            # Try 8-bit quantization first (most memory efficient, ~4GB vs ~14GB)
            if use_quantization:
                print("  Attempting to load with 8-bit quantization for GPU...")
                try:
                    from transformers import BitsAndBytesConfig
                    quantization_config = BitsAndBytesConfig(
                        load_in_8bit=True,
                        llm_int8_threshold=6.0,
                        llm_int8_has_fp16_weight=False
                    )
                    model = LanguageModel(
                        model_id, 
                        device_map=device,
                        quantization_config=quantization_config
                    )
                    print("  ✅ Loaded with 8-bit quantization")
                except (ImportError, AttributeError, TypeError) as e:
                    # Fallback to bfloat16
                    print(f"  8-bit quantization failed ({e}), trying bfloat16...")
                    try:
                        model = LanguageModel(
                            model_id, 
                            device_map=device,
                            torch_dtype=torch.bfloat16
                        )
                        print("  ✅ Loaded with bfloat16 precision")
                    except (TypeError, AttributeError) as e2:
                        # Final fallback
                        print(f"  ⚠️  bfloat16 not supported, loading in full precision")
                        model = LanguageModel(model_id, device_map=device)
            else:
                # Quantization disabled, try bfloat16 then full precision
                print("  Attempting to load with bfloat16 precision for GPU...")
                try:
                    model = LanguageModel(
                        model_id, 
                        device_map=device,
                        torch_dtype=torch.bfloat16
                    )
                    print("  ✅ Loaded with bfloat16 precision")
                except (TypeError, AttributeError) as e:
                    print(f"  ⚠️  bfloat16 not supported, loading in full precision")
                    model = LanguageModel(model_id, device_map=device)
    except RuntimeError as e:
        error_str = str(e).lower()
        if device.startswith("cuda"):
            if "out of memory" in error_str or "cuda oom" in error_str:
                print(f"⚠️  GPU out of memory even with quantization ({e}), falling back to CPU")
                device = "cpu"
                # Try 8-bit quantization on CPU if available
                if use_quantization:
                    try:
                        from transformers import BitsAndBytesConfig
                        quantization_config = BitsAndBytesConfig(
                            load_in_8bit=True,
                            llm_int8_threshold=6.0,
                            llm_int8_has_fp16_weight=False
                        )
                        model = LanguageModel(
                            model_id, 
                            device_map=device,
                            quantization_config=quantization_config,
                            low_cpu_mem_usage=True
                        )
                        print("  ✅ Loaded on CPU with 8-bit quantization")
                    except:
                        model = LanguageModel(
                            model_id, 
                            device_map=device,
                            low_cpu_mem_usage=True
                        )
                else:
                    model = LanguageModel(
                        model_id, 
                        device_map=device,
                        low_cpu_mem_usage=True
                    )
            elif "no kernel image" in error_str or "cuda capability" in error_str or "cuda error" in error_str:
                print(f"⚠️  GPU operation failed ({e}), falling back to CPU")
                device = "cpu"
                model = LanguageModel(
                    model_id, 
                    device_map=device,
                    low_cpu_mem_usage=True
                )
            else:
                raise
        else:
            raise
    except Exception as e:
        # If quantization fails for other reasons, try without it
        if device.startswith("cuda") and "quantization" in str(e).lower():
            print(f"⚠️  Quantization failed ({e}), trying without quantization...")
            try:
                model = LanguageModel(model_id, device_map=device)
            except RuntimeError as e2:
                error_str = str(e2).lower()
                if "out of memory" in error_str:
                    print(f"⚠️  GPU out of memory without quantization, falling back to CPU")
                    device = "cpu"
                    model = LanguageModel(
                        model_id, 
                        device_map=device,
                        low_cpu_mem_usage=True
                    )
                else:
                    raise
        else:
            raise
    
    return model


def load_models(
    base_model_id: str,
    ft_model_id: str,
    device: str = "cuda:0"
) -> Tuple[LanguageModel, LanguageModel]:
    """
    Load base and fine-tuned models.
    
    Automatically falls back to CPU if GPU is incompatible.
    
    Args:
        base_model_id: Hugging Face model ID for base model
        ft_model_id: Hugging Face model ID for fine-tuned model
        device: Device to load models on (will be adjusted if incompatible)
        
    Returns:
        Tuple of (base_model, ft_model)
    """
    # Check GPU compatibility
    if device.startswith("cuda") and not check_cuda_compatibility():
        print("⚠️  Falling back to CPU due to GPU incompatibility")
        device = "cpu"
    
    print(f"Loading base model: {base_model_id} on {device}")
    try:
        # Use low_cpu_mem_usage for CPU mode to reduce memory footprint
        if device == "cpu":
            base_model = LanguageModel(
                base_model_id, 
                device_map=device,
                low_cpu_mem_usage=True
            )
        else:
            base_model = LanguageModel(base_model_id, device_map=device)
    except RuntimeError as e:
        if "no kernel image" in str(e) or "CUDA capability" in str(e):
            print(f"⚠️  GPU error during base model loading, falling back to CPU: {e}")
            device = "cpu"
            base_model = LanguageModel(
                base_model_id, 
                device_map=device,
                low_cpu_mem_usage=True
            )
        else:
            raise
    
    print(f"Loading fine-tuned model: {ft_model_id} on {device}")
    try:
        # Use low_cpu_mem_usage for CPU mode to reduce memory footprint
        if device == "cpu":
            ft_model = LanguageModel(
                ft_model_id, 
                device_map=device,
                low_cpu_mem_usage=True
            )
        else:
            ft_model = LanguageModel(ft_model_id, device_map=device)
    except RuntimeError as e:
        if "no kernel image" in str(e) or "CUDA capability" in str(e):
            print(f"⚠️  GPU error during fine-tuned model loading, falling back to CPU: {e}")
            device = "cpu"
            ft_model = LanguageModel(
                ft_model_id, 
                device_map=device,
                low_cpu_mem_usage=True
            )
        else:
            raise
    
    return base_model, ft_model


def extract_activations(
    model: LanguageModel,
    texts: List[str],
    layer: int,
    num_tokens: int,
    batch_size: int = 8
) -> torch.Tensor:
    """
    Extract activations from a model for given texts.
    
    Args:
        model: LanguageModel instance
        texts: List of text strings
        layer: Layer index to extract from
        num_tokens: Number of tokens to extract per text
        batch_size: Batch size for processing
        
    Returns:
        Tensor of shape [num_texts, num_tokens, hidden_dim]
    """
    # Clear GPU cache before starting
    clear_gpu_cache()
    
    # Check GPU memory if using CUDA
    if torch.cuda.is_available():
        try:
            memory_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
            memory_reserved = torch.cuda.memory_reserved() / 1024**3  # GB
            print(f"  GPU memory before extraction: {memory_allocated:.2f}GB allocated, {memory_reserved:.2f}GB reserved")
        except:
            pass
    
    tokenizer = model.tokenizer
    activations = []
    
    # Process texts one at a time (like crosscoder does)
    # This is simpler and more memory efficient
    # nnsight's trace() expects strings, not tensors
    gpu_failed = False
    
    # Process one sample at a time with aggressive memory clearing
    # This is slower but much more memory efficient
    for i, text in enumerate(tqdm(texts, desc="Extracting activations")):
            # Truncate text to approximate num_tokens (rough estimate: 4 chars per token)
            # We'll extract first num_tokens from the actual activations
            max_chars = num_tokens * 4
            text_truncated = text[:max_chars] if len(text) > max_chars else text
            
            if len(text_truncated) == 0:
                continue
            
            # Extract activations using nnsight (same as crosscoder)
            # Pass string directly - nnsight will tokenize it
            try:
                # Clear cache before each trace to free up memory
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                with model.trace(text_truncated):
                    # Extract from MLP output at specified layer (same as crosscoder)
                    act = model.model.layers[layer].mlp.output.save()
            except RuntimeError as e:
                error_str = str(e).lower()
                if ("out of memory" in error_str or "cuda oom" in error_str) and not gpu_failed:
                    print(f"⚠️  GPU out of memory during activation extraction: {e}")
                    print("⚠️  Model needs to be reloaded on CPU. This will happen automatically.")
                    gpu_failed = True
                    # Clear cache and signal reload needed
                    clear_gpu_cache()
                    raise RuntimeError("GPU_OOM: Model needs to be reloaded on CPU")
                elif ("no kernel image" in error_str or "cuda error" in error_str or "cuda capability" in error_str) and not gpu_failed:
                    print(f"⚠️  GPU operation failed during activation extraction: {e}")
                    print("⚠️  Model needs to be reloaded on CPU. This will happen automatically.")
                    gpu_failed = True
                    clear_gpu_cache()
                    raise RuntimeError("GPU_INCOMPATIBLE: Model needs to be reloaded on CPU")
                else:
                    raise
            
            # Remove batch dimension: [1, seq_len, hidden_dim] -> [seq_len, hidden_dim]
            act = act.squeeze(0)  # [seq_len, hidden_dim]
            
            # Take first num_tokens (or pad if shorter)
            current_len = act.shape[0]
            if current_len < num_tokens:
                # Pad with zeros if sequence is shorter
                padding = torch.zeros(num_tokens - current_len, act.shape[1], dtype=act.dtype, device=act.device)
                act = torch.cat([act, padding], dim=0)
            else:
                # Truncate if longer
                act = act[:num_tokens, :]
            
            activations.append(act.detach().cpu())
            
            # Aggressive memory clearing after each sample
            del act
            clear_gpu_cache()
            gc.collect()
            
            # Every 10 samples, do a more aggressive cleanup
            if (i + 1) % 10 == 0:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                gc.collect()
    
    # Stack all activations
    return torch.stack(activations, dim=0)  # [num_texts, num_tokens, hidden_dim]


def compute_activation_differences(
    base_acts: torch.Tensor,
    ft_acts: torch.Tensor
) -> torch.Tensor:
    """
    Compute activation differences (fine-tuned - base).
    
    Args:
        base_acts: Base model activations [num_samples, num_tokens, hidden_dim]
        ft_acts: Fine-tuned model activations [num_samples, num_tokens, hidden_dim]
        
    Returns:
        Activation differences [num_samples, num_tokens, hidden_dim]
    """
    assert base_acts.shape == ft_acts.shape, "Activation shapes must match"
    return ft_acts - base_acts


def save_json(data: Dict[Any, Any], path: Path) -> None:
    """Save data as JSON file."""
    with open(path, 'w') as f:
        json.dump(data, f, indent=2, default=str)


def save_csv(data: pd.DataFrame, path: Path) -> None:
    """Save DataFrame as CSV file."""
    data.to_csv(path, index=False)


def clear_gpu_cache():
    """Clear GPU cache."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

