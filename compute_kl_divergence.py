"""
Compute KL divergence between a fine-tuned model and the base model.
This script reuses the KL computation logic from utils.py.
Supports multi-GPU setups to avoid OOM errors.
"""

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from typing import List, Optional
import numpy as np


# ============== Utility functions from utils.py ==============

def selective_log_softmax(logits, index):
    """
    A memory-efficient implementation of the common `log_softmax -> gather` operation.
    """
    if logits.dtype in [torch.float32, torch.float64]:
        selected_logits = torch.gather(logits, dim=-1, index=index.unsqueeze(-1)).squeeze(-1)
        logsumexp_values = torch.stack([torch.logsumexp(lg, dim=-1) for lg in logits])
        per_token_logps = selected_logits - logsumexp_values
    else:
        per_token_logps = []
        for row_logits, row_labels in zip(logits, index):
            row_logps = F.log_softmax(row_logits, dim=-1)
            row_per_token_logps = row_logps.gather(dim=-1, index=row_labels.unsqueeze(-1)).squeeze(-1)
            per_token_logps.append(row_per_token_logps)
        per_token_logps = torch.stack(per_token_logps)
    return per_token_logps


def compute_kl(new_logprobs, ref_logprobs, logits_p=None, logits_q=None):
    """
    Compute KL divergence.
    
    Two modes:
    - MC estimator (default): KL ≈ log p(x) - log q(x) under samples from p
    - Exact (stepwise/RB): KL = sum_x p(x) * (log p(x) - log q(x))
    """
    if logits_p is not None:
        # Exact KL computation (stepwise method)
        logp = torch.log_softmax(logits_p, dim=-1)
        logq = torch.log_softmax(logits_q, dim=-1)
        return torch.sum(torch.exp(logp) * (logp - logq), dim=-1)
    # MC estimator
    return new_logprobs - ref_logprobs


def first_true_indices(tensor):
    """Find the first True index in each row."""
    # If no True value, return the length of the tensor
    indices = torch.where(tensor, 
                          torch.arange(tensor.shape[1], device=tensor.device).expand_as(tensor),
                          tensor.shape[1] * torch.ones_like(tensor))
    return indices.min(dim=1).values


# ============== Main KL computation function ==============

def compute_kl_between_models(
    policy_model,
    ref_model,
    tokenizer,
    prompts: List[str],
    policy_device: str = "cuda:0",
    ref_device: str = "cuda:1",
    max_new_tokens: int = 128,
    temperature: float = 1.0,
    batch_size: int = 4,
):
    """
    Compute KL divergence between policy (fine-tuned) and reference (base) model.
    
    Args:
        policy_model: The fine-tuned model
        ref_model: The reference/base model
        tokenizer: The tokenizer
        prompts: List of prompts to generate from
        policy_device: Device for policy model (e.g., "cuda:0")
        ref_device: Device for reference model (e.g., "cuda:1")
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        batch_size: Batch size for processing
    
    Returns:
        dict with 'kl_mc' (MC estimator), 'kl_exact' (exact KL), and details
    """
    policy_model.eval()
    ref_model.eval()
    
    all_kl_mc = []
    all_kl_exact = []
    all_queries = []
    all_responses = []
    
    INVALID_LOGPROB = 1.0
    
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i + batch_size]
        
        # Tokenize prompts
        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        
        # Move inputs to policy device for generation
        inputs_policy = {k: v.to(policy_device) for k, v in inputs.items()}
        
        query = inputs_policy["input_ids"]
        context_length = query.shape[1]
        
        with torch.no_grad():
            # Generate from policy model
            generation_output = policy_model.generate(
                **inputs_policy,
                max_new_tokens=max_new_tokens,
                temperature=temperature + 1e-7,
                do_sample=True,
                top_p=1.0,
                pad_token_id=tokenizer.pad_token_id,
                return_dict_in_generate=True,
                output_scores=True,
            )
            
            query_response = generation_output.sequences
            response = query_response[:, context_length:]
            
            # Get policy logits (on policy_device)
            policy_attention_mask = (query_response != tokenizer.pad_token_id).long()
            policy_output = policy_model(query_response, attention_mask=policy_attention_mask)
            policy_logits = policy_output.logits[:, context_length - 1: -1]
            policy_logits = policy_logits / (temperature + 1e-7)
            policy_logprob = selective_log_softmax(policy_logits, response)
            
            # Get reference logits (move data to ref_device)
            query_response_ref = query_response.to(ref_device)
            ref_attention_mask = (query_response_ref != tokenizer.pad_token_id).long()
            ref_output = ref_model(query_response_ref, attention_mask=ref_attention_mask)
            ref_logits = ref_output.logits[:, context_length - 1: -1]
            ref_logits = ref_logits / (temperature + 1e-7)
            
            # Move ref_logits back to policy_device for KL computation
            ref_logits = ref_logits.to(policy_device)
            response_for_ref = response  # already on policy_device
            ref_logprob = selective_log_softmax(ref_logits, response_for_ref)
            
            # Compute sequence lengths (mask out padding)
            sequence_lengths = first_true_indices(response == tokenizer.pad_token_id) - 1
            response_idxs = torch.arange(response.shape[1], device=policy_device).repeat(response.shape[0], 1)
            padding_mask = response_idxs > sequence_lengths.unsqueeze(1)
            
            # Mask invalid positions
            policy_logprob = torch.masked_fill(policy_logprob, padding_mask, INVALID_LOGPROB)
            ref_logprob = torch.masked_fill(ref_logprob, padding_mask, INVALID_LOGPROB)
            
            # Compute KL (MC estimator)
            kl_mc = compute_kl(policy_logprob, ref_logprob)
            kl_mc = torch.masked_fill(kl_mc, padding_mask, 0.0).sum(1)
            
            # Compute exact KL (stepwise/RB method)
            kl_exact = compute_kl(policy_logprob, ref_logprob, logits_p=policy_logits, logits_q=ref_logits)
            kl_exact = torch.masked_fill(kl_exact, padding_mask, 0.0).sum(1)
            
            all_kl_mc.extend(kl_mc.cpu().numpy())
            all_kl_exact.extend(kl_exact.cpu().numpy())
            all_queries.extend(tokenizer.batch_decode(query, skip_special_tokens=True))
            all_responses.extend(tokenizer.batch_decode(response, skip_special_tokens=True))
            
            # Clear GPU cache periodically
            if i % (batch_size * 4) == 0:
                torch.cuda.empty_cache()
    
    return {
        "kl_mc_mean": np.mean(all_kl_mc),
        "kl_mc_std": np.std(all_kl_mc),
        "kl_exact_mean": np.mean(all_kl_exact),
        "kl_exact_std": np.std(all_kl_exact),
        "kl_mc_values": all_kl_mc,
        "kl_exact_values": all_kl_exact,
        "queries": all_queries,
        "responses": all_responses,
    }


# ============== Example usage with your Mistral models ==============

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter_name", type=str, 
                        default="liuhaozhe6788/mistralai_Mistral-7B-Instruct-v0.3-peftq_proj_k_proj_v_proj_o_proj-bs1-ne1")
    parser.add_argument("--model_name", type=str, default="mistralai/Mistral-7B-Instruct-v0.3")
    parser.add_argument("--policy_device", type=str, default="cuda:0", help="GPU for policy/fine-tuned model")
    parser.add_argument("--ref_device", type=str, default="cuda:1", help="GPU for reference/base model")
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_prompts", type=int, default=10, help="Number of prompts to use")
    args = parser.parse_args()
    
    # Check available GPUs
    num_gpus = torch.cuda.device_count()
    print(f"Available GPUs: {num_gpus}")
    for i in range(num_gpus):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    
    if num_gpus < 2:
        print("\nWARNING: Less than 2 GPUs available. Using single GPU mode.")
        print("This may cause OOM errors with large models.")
        args.ref_device = args.policy_device
    
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"\nLoading base/reference model on {args.ref_device}...")
    ref_model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16,
        device_map=args.ref_device,
    )
    
    print(f"\nLoading fine-tuned model (base + LoRA adapter) on {args.policy_device}...")
    policy_model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16,
        device_map=args.policy_device,
    )
    policy_model = PeftModel.from_pretrained(policy_model, args.adapter_name)
    
    # Print memory usage
    print("\nGPU Memory Usage after loading models:")
    for i in range(num_gpus):
        allocated = torch.cuda.memory_allocated(i) / 1024**3
        reserved = torch.cuda.memory_reserved(i) / 1024**3
        print(f"  GPU {i}: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved")
    
    # Sample prompts - you can replace these with your actual prompts
    sample_prompts = [
        "<s>[INST] What is machine learning? [/INST]",
        "<s>[INST] Explain the concept of neural networks. [/INST]",
        "<s>[INST] How does gradient descent work? [/INST]",
        "<s>[INST] What are transformers in deep learning? [/INST]",
        "<s>[INST] Describe reinforcement learning. [/INST]",
        "<s>[INST] What is the difference between supervised and unsupervised learning? [/INST]",
        "<s>[INST] How do convolutional neural networks work? [/INST]",
        "<s>[INST] What is attention mechanism? [/INST]",
        "<s>[INST] Explain backpropagation. [/INST]",
        "<s>[INST] What is transfer learning? [/INST]",
    ][:args.num_prompts]
    
    print(f"\nComputing KL divergence on {len(sample_prompts)} prompts...")
    print(f"  Policy model: {args.policy_device}")
    print(f"  Reference model: {args.ref_device}")
    
    results = compute_kl_between_models(
        policy_model=policy_model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        prompts=sample_prompts,
        policy_device=args.policy_device,
        ref_device=args.ref_device,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
    )
    
    print("\n" + "="*60)
    print("KL DIVERGENCE RESULTS (policy || reference)")
    print("="*60)
    print(f"MC Estimator KL:     {results['kl_mc_mean']:.4f} ± {results['kl_mc_std']:.4f}")
    print(f"Exact (Stepwise) KL: {results['kl_exact_mean']:.4f} ± {results['kl_exact_std']:.4f}")
    print("="*60)
    
    print("\nPer-sample KL values:")
    for i, (q, r, kl_mc, kl_ex) in enumerate(zip(
        results['queries'][:5], 
        results['responses'][:5],
        results['kl_mc_values'][:5],
        results['kl_exact_values'][:5]
    )):
        print(f"\n--- Sample {i+1} ---")
        print(f"Query: {q[:100]}...")
        print(f"Response: {r[:100]}...")
        print(f"KL (MC): {kl_mc:.4f}, KL (Exact): {kl_ex:.4f}")