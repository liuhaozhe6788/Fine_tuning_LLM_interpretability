"""
Configuration for ADL analysis.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import List
import os


@dataclass
class ADLConfig:
    """Configuration for ADL analysis."""
    
    # Models
    base_model_id: str = "mistralai/Mistral-7B-Instruct-v0.3"
    ft_model_id: str = "liuhaozhe6788/mistralai_Mistral-7B-Instruct-v0.3-FinQA-lora"
    
    # Dataset
    dataset_path: str = "../data/clean_with_code/FinQA/finqa_train_generated_filtered.csv"
    num_samples: int = 1024
    random_seed: int = 49  # Match crosscoder
    # Note: KL divergence uses custom queries from JSON files, not sampled FinQA data, so no seed alignment needed
    
    # Analysis parameters
    layer: int = 16  # Middle layer (0.5 depth for 32-layer model) - matches crosscoder
    positions: List[int] = None  # Will default to [0, 1, 2, 3, 4, 5]
    sequence_length: int = 64  # Total tokens to extract (reduced from 128 to save memory)
    # Note: KL experiments may use max_tokens_per_sample: 1024 (from diffing-toolkit config)
    
    # GPU optimization
    use_quantization: bool = True  # Use 8-bit quantization for GPU to fit 7B model in 16GB
    
    # Logit lens
    logit_lens_k: int = 100  # Top-k tokens to save
    
    # Patchscope
    patchscope_enabled: bool = True
    patchscope_tokens_k: int = 20
    
    # Token relevance
    token_relevance_enabled: bool = True
    token_relevance_k: int = 20  # Top-k tokens to analyze
    
    # Causal effect
    causal_effect_enabled: bool = True
    causal_effect_max_samples: int = 1000
    causal_effect_batch_size: int = 2  # KL uses batch_size=1-4, ADL uses 2 for memory efficiency
    
    # Output - use scratch if available, otherwise current dir
    results_dir: Path = None  # Will be set in __post_init__
    device: str = "cuda:0"
    
    # Hugging Face
    hf_token_env: str = "HF_TOKEN"
    hf_cache_dir: Path = None  # Will be set in __post_init__
    
    # Option to use pre-extracted activations from HuggingFace (from crosscoder experiments)
    # NOTE: Currently disabled - HF dataset stores [samples, hidden_dim] but ADL needs [samples, seq_len, hidden_dim]
    use_hf_activations: bool = False  # If True, load from HF instead of extracting
    hf_base_acts_dataset: str = "liuhaozhe6788/acts-finqa-base"
    hf_ft_acts_dataset: str = "liuhaozhe6788/acts-finqa-lora"
    
    def __post_init__(self):
        """Set defaults after initialization."""
        if self.positions is None:
            self.positions = [0, 1, 2, 3, 4, 5]
        
        # Determine scratch space
        scratch_base = self._get_scratch_base()
        
        # Set HF cache to scratch
        if self.hf_cache_dir is None:
            self.hf_cache_dir = scratch_base / "hf-cache"
            self.hf_cache_dir.mkdir(parents=True, exist_ok=True)
            # Set environment variables for HF cache
            os.environ["HF_HOME"] = str(self.hf_cache_dir)
            os.environ["TRANSFORMERS_CACHE"] = str(self.hf_cache_dir)
        
        # Set results directory to scratch (dynamic based on num_samples)
        if self.results_dir is None:
            model_name = self.base_model_id.split('/')[-1]
            self.results_dir = scratch_base / "adl-results" / f"{model_name}_{self.num_samples}_samples"
        
        # Create results directory
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (self.results_dir / "logit_lens").mkdir(exist_ok=True)
        (self.results_dir / "patchscope").mkdir(exist_ok=True)
        (self.results_dir / "token_relevance").mkdir(exist_ok=True)
        (self.results_dir / "causal_effect").mkdir(exist_ok=True)
        (self.results_dir / "steering").mkdir(exist_ok=True)
        (self.results_dir / "summaries").mkdir(exist_ok=True)
    
    @staticmethod
    def _get_scratch_base() -> Path:
        """Get scratch space base directory, or fallback to current directory."""
        # Try /work/scratch/{user} first (ETH cluster standard)
        user = os.environ.get("USER", "user")
        work_scratch_path = Path("/work/scratch") / user
        if work_scratch_path.exists() and work_scratch_path.is_dir():
            return work_scratch_path
        
        # Try /scratch/{user}
        scratch_path = Path("/scratch") / user
        if scratch_path.exists() and scratch_path.is_dir():
            return scratch_path
        
        # Try /work/{user}
        work_path = Path("/work") / user
        if work_path.exists() and work_path.is_dir():
            return work_path
        
        # Fallback to current directory
        return Path.cwd()

