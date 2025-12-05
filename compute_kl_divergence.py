"""
Compute KL divergence between a fine-tuned model and the base model.
This script reuses the KL computation logic from utils.py.
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
    device: str = "cuda",
    max_new_tokens: int = 128,
    temperature: float = 1.0,
    stepwise: bool = False,  # If True, use exact KL; if False, use MC estimator
    batch_size: int = 4,
):
    """
    Compute KL divergence between policy (fine-tuned) and reference (base) model.
    
    Args:
        policy_model: The fine-tuned model
        ref_model: The reference/base model
        tokenizer: The tokenizer
        prompts: List of prompts to generate from
        device: Device to use
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        stepwise: If True, compute exact KL; if False, use MC estimator
        batch_size: Batch size for processing
    
    Returns:
        dict with 'kl_mc' (MC estimator), 'kl_exact' (exact KL if stepwise=True), and details
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
        ).to(device)
        
        query = inputs["input_ids"]
        context_length = query.shape[1]
        
        with torch.no_grad():
            # Generate from policy model
            generation_output = policy_model.generate(
                **inputs,
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
            
            # Get policy logits
            policy_output = policy_model(query_response, attention_mask=(query_response != tokenizer.pad_token_id).long())
            policy_logits = policy_output.logits[:, context_length - 1: -1]
            policy_logits = policy_logits / (temperature + 1e-7)
            policy_logprob = selective_log_softmax(policy_logits, response)
            
            # Get reference logits
            ref_output = ref_model(query_response, attention_mask=(query_response != tokenizer.pad_token_id).long())
            ref_logits = ref_output.logits[:, context_length - 1: -1]
            ref_logits = ref_logits / (temperature + 1e-7)
            ref_logprob = selective_log_softmax(ref_logits, response)
            
            # Compute sequence lengths (mask out padding)
            sequence_lengths = first_true_indices(response == tokenizer.pad_token_id) - 1
            response_idxs = torch.arange(response.shape[1], device=device).repeat(response.shape[0], 1)
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
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_prompts", type=int, default=10, help="Number of prompts to use")
    args = parser.parse_args()
    
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print("Loading base/reference model...")
    ref_model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16,
        device_map=args.device,
    )
    
    print("Loading fine-tuned model (base + LoRA adapter)...")
    policy_model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16,
        device_map=args.device,
    )
    policy_model = PeftModel.from_pretrained(policy_model, args.adapter_name)
    
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
    results = compute_kl_between_models(
        policy_model=policy_model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        prompts=sample_prompts,
        device=args.device,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
    )
    
    print("\n" + "="*60)
    print("KL DIVERGENCE RESULTS (policy || reference)")
    print("="*60)
    print(f"MC Estimator KL:    {results['kl_mc_mean']:.4f} ± {results['kl_mc_std']:.4f}")
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