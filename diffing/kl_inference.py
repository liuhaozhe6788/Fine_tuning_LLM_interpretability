import torch
import torch.nn.functional as F
from vllm import LLM
from vllm.lora.request import LoRARequest
from huggingface_hub import snapshot_download


# ------------------------------
# KL Divergence Helper
# ------------------------------

def compute_kl(base_logits, ft_logits):
    """
    Compute per-token KL divergence KL(finetuned || base)
    base_logits, ft_logits: [1, seq_len, vocab_size]
    """
    base_log_probs = torch.nn.functional.log_softmax(base_logits, dim=-1)
    ft_log_probs   = torch.nn.functional.log_softmax(ft_logits, dim=-1)

    ft_probs = torch.exp(ft_log_probs)  # probability distribution of finetuned model

    kl = torch.sum(ft_probs * (ft_log_probs - base_log_probs), dim=-1)
    mean_kl = kl.mean()

    return kl[0].tolist(), float(mean_kl)


# ------------------------------
# Main vLLM KL Divergence Runner
# ------------------------------
'''
def run_kl_divergence(prompt, base_model_name, adapter_repo):
    """
    Runs KL divergence between:
      - base model logits
      - LoRA-finetuned model logits

    Returns dictionary with tokens, per-token KL scores, and mean KL.
    """

    print("Loading base model…")
    base_llm = LLM(model=base_model_name)

    tokenizer = base_llm.get_tokenizer()
    encoded = tokenizer(prompt, return_tensors="pt", padding=False)

    input_ids = encoded["input_ids"]
    attn = torch.ones_like(input_ids)

    # ---- Base model logits ----
    base_out = base_llm.compute_logits(
        input_ids=input_ids,
        attention_mask=attn
    )
    base_logits = base_out.logits


    # ---- LoRA finetuned model logits ----
    print("Loading LoRA adapter…")
    ft_llm = LLM(model=base_model_name, enable_lora=True)
    adapter_path = snapshot_download(repo_id=adapter_repo)

    ft_out = ft_llm.compute_logits(
        input_ids=input_ids,
        attention_mask=attn,
        lora_request=LoRARequest("KL-LoRA", 1, adapter_path)
    )
    ft_logits = ft_out.logits


    # ---- Compute KL divergence ----
    per_token_kl, mean_kl = compute_kl(base_logits, ft_logits)

    tokens = [tokenizer.decode([tid]) for tid in input_ids[0]]

    return {
        "tokens": tokens,
        "per_token_kl": per_token_kl,
        "mean_kl": mean_kl,
    }

#in inference.py
from kl_inference import run_kl_divergence

print("\nRunning KL divergence analysis...\n")

kl_result = run_kl_divergence(
    prompt=passage_query,   # the prompt you already use
    base_model_name=model_name,
    adapter_repo=adapter_name  # your LoRA adapter
)

print("\n----- KL DIVERGENCE RESULTS -----")

for tok, kl in zip(kl_result["tokens"], kl_result["per_token_kl"]):
    print(f"{tok!r}: KL={kl:.6f}")

print("\nMean KL:", kl_result["mean_kl"])
'''