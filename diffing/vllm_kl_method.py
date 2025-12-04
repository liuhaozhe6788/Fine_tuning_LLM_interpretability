import torch
import numpy as np
import os, gc
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from vllm.outputs import RequestOutput
from vllm.distributed.parallel_state import destroy_model_parallel
from typing import List, Tuple, Dict
from huggingface_hub import snapshot_download

# --- 1. Configuration based on inference.py ---
BASE_MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"
ADAPTER_NAME = "liuhaozhe6788/mistralai_Mistral-7B-Instruct-v0.3-peftq_proj_k_proj_v_proj_o_proj-bs1-ne1"
LORA_RANK = 1 # Assuming rank is 1 from the adapter name
# Download LoRA weights once to get the local path
print(f"Downloading LoRA weights for adapter: {ADAPTER_NAME}")
try:
    LORA_PATH = snapshot_download(repo_id=ADAPTER_NAME)
    print(f"LoRA weights downloaded to: {LORA_PATH}")
except Exception as e:
    print(f"Could not download LoRA weights: {e}")
    print("Using a placeholder path. You MUST ensure weights are local.")
    LORA_PATH = "/tmp/placeholder/path"

LORA_REQUEST = LoRARequest("FinQA_adapter", LORA_RANK, LORA_PATH)

# --- 2. Simulated Input Data (Prompts) ---
# In a real setup (like kl.py), these would come from the SampleCache/Dataset
SIMULATED_PROMPTS = [
    "The capital of France is",
    "What is the average of 10, 20, and 30?",
    "Asset and liability management activities the primary objective of asset and liability management is to provide",
]

# --- 3. Core VLLM KL Calculation Logic ---

def get_logprobs(
    llm: LLM, 
    prompt: str, 
    lora_request: LoRARequest = None
) -> RequestOutput:
    """Helper to get log probabilities from vLLM for the prompt tokens."""
    # max_tokens=0 ensures that vLLM only performs prompt scoring, no generation.
    # prompt_logprobs=1 requests the log probability for the next token prediction
    # at every position in the prompt.
    sampling_params = SamplingParams(
        temperature=0.0, # Ensures deterministic sampling if needed, but irrelevant for max_tokens=0
        top_p=1.0, 
        max_tokens=0, 
        prompt_logprobs=1 
    )
    
    outputs = llm.generate(
        [prompt], 
        sampling_params=sampling_params,
        lora_request=lora_request
    )
    return outputs[0]

def compute_kl_divergence_approximation(
    llm: LLM, 
    prompts: List[str], 
    lora_request: LoRARequest
) -> Dict[str, List[Dict]]:
    """
    Computes KL divergence approximation (logP_ft - logP_base) for a batch of prompts.
    """
    results = []
    
    for i, prompt in enumerate(prompts):
        print(f"\n--- Processing Prompt {i+1} ---")
        
        # 1. Run inference for Finetuned model (Base + LoRA)
        ft_output = get_logprobs(llm, prompt, lora_request=lora_request)
        
        # 2. Run inference for Base model (LoRA is None)
        # NOTE: If using the same LLM instance, vLLM automatically handles 
        # dropping the LoRA weights when lora_request is None.
        base_output = get_logprobs(llm, prompt, lora_request=None)

        # Log probabilities for prompt tokens (excluding BOS token, index 0)
        # Each entry is a dict: {token_id: logprob}
        ft_log_probs_dict_list = ft_output.prompt_logprobs[1:] 
        base_log_probs_dict_list = base_output.prompt_logprobs[1:]

        # Extract the log probability for the actual token ID in the sequence
        # The key is the token ID for the token at that position
        ft_log_probs = [list(d.values())[0] for d in ft_log_probs_dict_list]
        base_log_probs = [list(d.values())[0] for d in base_log_probs_dict_list]
        tokens = [llm.get_tokenizer().decode(list(d.keys())[0]) for d in ft_log_probs_dict_list]
        
        # Ensure lengths match
        if len(ft_log_probs) != len(base_log_probs):
            print("Warning: Log probability lists have different lengths. Skipping calculation.")
            continue

        # Calculate KL approximation: log(P_ft) - log(P_base)
        # This is equivalent to the penalty term in RLHF
        per_token_kl_approx = np.array(ft_log_probs) - np.array(base_log_probs)
        
        mean_per_sample_kl_approx = np.mean(per_token_kl_approx)
        
        print(f"Tokens: {tokens}")
        print(f"FT Log Probs: {np.round(ft_log_probs, 4)}")
        print(f"Base Log Probs: {np.round(base_log_probs, 4)}")
        print(f"Per-Token KL Approx: {np.round(per_token_kl_approx, 4)}")
        print(f"Mean Per-Sample KL Approx: {np.round(mean_per_sample_kl_approx, 4)}")
        
        results.append({
            "prompt": prompt,
            "tokens": tokens,
            "per_token_kl_approx": per_token_kl_approx.tolist(),
            "mean_per_sample_kl_approx": mean_per_sample_kl_approx
        })
        
    return {"kl_results": results}

# --- 4. Main Execution Block ---

def run_kl_divergence():
    # 1. Initialize vLLM (must enable lora)
    print("\nInitializing vLLM model...")
    try:
        # We use a single LLM instance for both base and finetuned inference
        llm = LLM(
            model=BASE_MODEL_NAME, 
            enable_lora=True, 
            max_lora_rank=512, # Match the max rank to support the adapter
            # Other parameters like tensor_parallel_size may be needed
        )
    except Exception as e:
        print(f"Failed to initialize vLLM: {e}")
        print("Please ensure you have vLLM installed and CUDA/GPU available.")
        return

    # 2. Compute KL divergence
    kl_results = compute_kl_divergence_approximation(
        llm, 
        SIMULATED_PROMPTS, 
        LORA_REQUEST
    )

    # 3. Cleanup
    destroy_model_parallel()
    del llm
    gc.collect()
    torch.cuda.empty_cache()
    
    print("\nKL Divergence Approximation Completed.")
    # In a real setting, you would save kl_results to disk (e.g., JSON file)

if __name__ == "__main__":
    run_kl_divergence()