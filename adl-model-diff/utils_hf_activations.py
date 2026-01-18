"""
Utility functions for loading pre-extracted activations from HuggingFace.
This allows us to reuse activations from crosscoder experiments, saving GPU memory.
"""
import torch
from datasets import load_dataset
from typing import Tuple
from huggingface_hub import login
import os
from pathlib import Path
import json
import time

# Try to import DownloadConfig (may not be available in all versions)
try:
    from datasets.utils.download_manager import DownloadConfig
    HAS_DOWNLOAD_CONFIG = True
except ImportError:
    HAS_DOWNLOAD_CONFIG = False


def load_hf_activations(
    base_dataset_id: str = "liuhaozhe6788/acts-finqa-base",
    ft_dataset_id: str = "liuhaozhe6788/acts-finqa-lora",
    hf_token: str = None,
    num_samples: int = None,
    max_retries: int = 3,
    timeout: int = 60
) -> Tuple[torch.Tensor, torch.Tensor, dict]:
    """
    Load pre-extracted activations from HuggingFace datasets.
    
    These activations are from layer 16, same seed (49), same dataset as crosscoder.
    They are scaled, so we return scaling factors to unscale if needed.
    
    Args:
        base_dataset_id: HF dataset ID for base model activations
        ft_dataset_id: HF dataset ID for fine-tuned model activations
        hf_token: HuggingFace token (or uses HF_TOKEN env var)
        num_samples: Number of samples to load (None = all)
        max_retries: Maximum number of retry attempts for downloads
        timeout: Timeout in seconds for HTTP requests
        
    Returns:
        Tuple of (base_acts, ft_acts, scaling_factors) where:
        - base_acts: [num_samples, seq_len, hidden_dim] tensor
        - ft_acts: [num_samples, seq_len, hidden_dim] tensor
        - scaling_factors: dict with scaling factors (for unscaling if needed)
    """
    if hf_token is None:
        hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        login(hf_token)
    
    # Try to create download config (some versions don't support all parameters)
    download_config = None
    if HAS_DOWNLOAD_CONFIG:
        try:
            download_config = DownloadConfig(max_retries=max_retries)
        except (TypeError, AttributeError):
            # Some versions don't support max_retries in DownloadConfig
            download_config = None
    
    print(f"Loading base model activations from {base_dataset_id}...")
    for attempt in range(max_retries):
        try:
            if download_config is not None:
                base_dataset = load_dataset(
                    base_dataset_id, 
                    split="train",
                    download_config=download_config
                )
            else:
                base_dataset = load_dataset(
                    base_dataset_id, 
                    split="train"
                )
            break
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = (attempt + 1) * 10
                print(f"  ⚠️  Attempt {attempt + 1} failed: {type(e).__name__}: {e}")
                print(f"  Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                print(f"  ❌ Failed after {max_retries} attempts")
                raise
    
    if num_samples is not None:
        base_dataset = base_dataset.select(range(min(num_samples, len(base_dataset))))
    
    print(f"Loading fine-tuned model activations from {ft_dataset_id}...")
    for attempt in range(max_retries):
        try:
            if download_config is not None:
                ft_dataset = load_dataset(
                    ft_dataset_id, 
                    split="train",
                    download_config=download_config
                )
            else:
                ft_dataset = load_dataset(
                    ft_dataset_id, 
                    split="train"
                )
            break
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = (attempt + 1) * 10
                print(f"  ⚠️  Attempt {attempt + 1} failed: {type(e).__name__}: {e}")
                print(f"  Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                print(f"  ❌ Failed after {max_retries} attempts")
                raise
    
    if num_samples is not None:
        ft_dataset = ft_dataset.select(range(min(num_samples, len(ft_dataset))))
    
    # Extract activations from datasets
    # The datasets have a column like "base_model_acts" or "ft_model_acts"
    base_key = "base_model_acts" if "base_model_acts" in base_dataset.column_names else base_dataset.column_names[0]
    ft_key = "ft_model_acts" if "ft_model_acts" in ft_dataset.column_names else ft_dataset.column_names[0]
    
    print(f"  Base activations key: {base_key}")
    print(f"  FT activations key: {ft_key}")
    
    # Convert to tensors
    base_acts_list = [torch.tensor(item) for item in base_dataset[base_key]]
    ft_acts_list = [torch.tensor(item) for item in ft_dataset[ft_key]]
    
    # Stack into single tensor
    base_acts = torch.stack(base_acts_list)  # [num_samples, seq_len, hidden_dim]
    ft_acts = torch.stack(ft_acts_list)  # [num_samples, seq_len, hidden_dim]
    
    print(f"  Base activations shape: {base_acts.shape}")
    print(f"  FT activations shape: {ft_acts.shape}")
    
    # Load scaling factors if available
    scaling_factors = {}
    scaling_file = Path("../crosscoder-model-diff/scaling_factors.json")
    if scaling_file.exists():
        with open(scaling_file, 'r') as f:
            scaling_factors = json.load(f)
        print(f"  Loaded scaling factors: {scaling_factors}")
    else:
        print("  ⚠️  No scaling factors file found. Activations may be scaled.")
    
    # Note: The activations in HF are already scaled. For ADL, we typically want unscaled.
    # But since we're computing differences, scaling cancels out, so it's fine.
    # If you need unscaled, divide by the scaling factors.
    
    return base_acts, ft_acts, scaling_factors


def unscale_activations(acts: torch.Tensor, scaling_factor: float) -> torch.Tensor:
    """Unscale activations by dividing by scaling factor."""
    return acts / scaling_factor

