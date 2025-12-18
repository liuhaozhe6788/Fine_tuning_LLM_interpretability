"""
Compute KL divergence between a fine-tuned model and the base model.
Supports multi-GPU setups, custom queries from JSON, and PDF context injection.
"""

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from typing import List, Optional, Dict, Any
import numpy as np
import json
import os
from datetime import datetime
import re
from collections import defaultdict

# ============== PDF Text Extraction ==============

def extract_text_from_pdf(pdf_path: str, max_pages: Optional[int] = None) -> str:
    """Extract text from a PDF file."""
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(pdf_path)
        text_parts = []
        pages_to_read = min(len(doc), max_pages) if max_pages else len(doc)
        for page_num in range(pages_to_read):
            page = doc[page_num]
            text_parts.append(page.get_text())
        doc.close()
        return "\n".join(text_parts)
    except ImportError:
        try:
            import pdfplumber
            text_parts = []
            with pdfplumber.open(pdf_path) as pdf:
                pages_to_read = min(len(pdf.pages), max_pages) if max_pages else len(pdf.pages)
                for i in range(pages_to_read):
                    page_text = pdf.pages[i].extract_text()
                    if page_text:
                        text_parts.append(page_text)
            return "\n".join(text_parts)
        except ImportError:
            raise ImportError(
                "No PDF library found. Install with:\n"
                "  pip install pymupdf --break-system-packages"
            )


def load_document_context(doc_config: Dict[str, Any], base_dir: str = ".") -> str:
    """Load document context from various sources."""
    doc_type = doc_config.get("type", "text")
    
    if doc_type == "text":
        return doc_config.get("text", "")
    elif doc_type == "pdf":
        pdf_path = doc_config.get("path", "")
        if not os.path.isabs(pdf_path):
            pdf_path = os.path.join(base_dir, pdf_path)
        max_pages = doc_config.get("max_pages", None)
        return extract_text_from_pdf(pdf_path, max_pages)
    elif doc_type == "txt":
        txt_path = doc_config.get("path", "")
        if not os.path.isabs(txt_path):
            txt_path = os.path.join(base_dir, txt_path)
        with open(txt_path, "r", encoding="utf-8") as f:
            return f.read()
    else:
        raise ValueError(f"Unknown document type: {doc_type}")


# ============== Query Loading ==============

def load_queries_from_json(json_path: str) -> List[Dict[str, Any]]:
    """
    Load queries from a JSON file.
    
    Expected format:
    {
        "queries": [
            {
                "question": "What is...",
                "expected_answer": "The answer is...",  // optional
                "documents": [  // optional
                    {"type": "pdf", "path": "path/to/doc.pdf", "max_pages": 10},
                    {"type": "text", "text": "Some context..."},
                    {"type": "txt", "path": "path/to/doc.txt"}
                ]
            }
        ]
    }
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    if isinstance(data, list):
        return data
    elif isinstance(data, dict) and "queries" in data:
        return data["queries"]
    else:
        raise ValueError("JSON must contain a list or a dict with 'queries' key")


def format_prompt_with_context(
    question: str,
    documents: Optional[List[Dict[str, Any]]] = None,
    base_dir: str = ".",
    prompt_template: str = "mistral",
    max_context_chars: int = 8000,
) -> str:
    """Format a prompt with optional document context."""
    context_parts = []
    
    if documents:
        for i, doc_config in enumerate(documents):
            try:
                doc_text = load_document_context(doc_config, base_dir)
                if len(doc_text) > max_context_chars // len(documents):
                    doc_text = doc_text[:max_context_chars // len(documents)] + "\n[... truncated ...]"
                context_parts.append(f"Document {i+1}:\n{doc_text}")
            except Exception as e:
                print(f"Warning: Could not load document {i+1}: {e}")
    
    if context_parts:
        context = "\n\n".join(context_parts)
        full_question = f"Based on the following documents, answer the question.\n\n{context}\n\nQuestion: {question}"
    else:
        full_question = question
    
    if prompt_template == "mistral":
        return f"<s>[INST] {full_question} [/INST]"
    elif prompt_template == "llama":
        return f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{full_question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    elif prompt_template == "raw":
        return full_question
    else:
        return f"<s>[INST] {full_question} [/INST]"


# ============== Utility functions ==============

def selective_log_softmax(logits, index):
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
    if logits_p is not None:
        logp = torch.log_softmax(logits_p, dim=-1)
        logq = torch.log_softmax(logits_q, dim=-1)
        return torch.sum(torch.exp(logp) * (logp - logq), dim=-1)
    return new_logprobs - ref_logprobs


def first_true_indices(tensor):
    indices = torch.where(tensor, 
                          torch.arange(tensor.shape[1], device=tensor.device).expand_as(tensor),
                          tensor.shape[1] * torch.ones_like(tensor))
    return indices.min(dim=1).values

VALUE_INVESTING_TERMS = {             

    # Valuation
    "pe", "p/e", "earnings", "valuation", "multiple",
    "market", "cap", "capitalization", "marketcap","price-to",

    # Cash flow
    "cash", "flow", "fcf", "free", "operating","free-cash","cash-flow",

    # Profitability
    "net", "income", "profit", "margin",

    # Capital efficiency
    "roic", "return", "invested", "capital", "return-on",

    # Balance sheet
    "debt", "equity", "liabilities", "assets",

    # Share structure
    "shares", "outstanding", "buyback", "repurchase",

    # Time horizon
    "five-year", "year", "years", "average", "growth", "long-term"
}

def bucket_token(tok: str) -> str:
    t = tok.lower().replace("▁", "").replace("Ġ", "")
    

    # ---- SENTINELS / CONTROL ----
    if "###end" in t or "end python" in t:
        return "sentinel"

    # ---- EXECUTION TRACE ----
    if re.search(r"(add|subtract|multiply|divide)\(#?\d", t):
        return "execution_trace"

    # ---- CALCULATION TEMPLATE ----
    if re.match(r"step\d+|ans|average", t):
        return "calculation_template"
    if t in ["+", "-", "*", "/", "="]:
        return "calculation_template"

    # ---- PROGRAMMING SYNTAX ----
    if any(x in t for x in ["const_", "#", "(", ")", "=", ":"]):
        return "code_syntax"

    # ---- VALUE INVESTING ----
    if t in VALUE_INVESTING_TERMS:
        return "value_investing"

    # ---- NUMBERS ----
    if re.search(r"\d", t):
        return "number"

    # ---- INSTRUCTION SCAFFOLD ----
    if t in ["answer", "step", "steps", "strictly", "follow"]:
        return "instruction_scaffold"

    # ---- DISCOURSE / PROSE ----
    if t in ["because", "however", "therefore", "collectively"]:
        return "discourse"

    if t.strip() == "":
        return "whitespace"

    return "lexical"

def aggregate_kl_by_bucket(token_kl_records):
    agg = defaultdict(lambda: {
        "total_kl": 0.0,
        "count": 0,
        "mean_kl": 0.0,
    })

    for rec in token_kl_records:
        b = rec["bucket"]
        agg[b]["total_kl"] += rec["kl"]
        agg[b]["count"] += 1

    for b in agg:
        agg[b]["mean_kl"] = agg[b]["total_kl"] / max(1, agg[b]["count"])

    return agg

# ============== Main KL computation ==============

def post_process_answer(ans: str, end_marker: str = "###End Python") -> str:
    if end_marker in ans:
        return ans.split(end_marker)[0] + "\n" + end_marker
    return ans


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
    generate_ref_responses: bool = True,  # New parameter
):
    """
    Compute KL divergence and generate responses from both models.
    
    Args:
        generate_ref_responses: If True, also generate responses from reference model
    """
    policy_model.eval()
    ref_model.eval()
    
    all_kl_mc = []
    all_kl_exact = []
    all_queries = []
    all_policy_responses = []
    all_ref_responses = []
    all_exact_per_token = []
    all_mc_per_token = []

    token_kl_records = []
    
    INVALID_LOGPROB = 1.0
    
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i + batch_size]
        print(f"  Processing batch {i//batch_size + 1}/{(len(prompts) + batch_size - 1)//batch_size}...")
        
        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=4096
        )
        
        inputs_policy = {k: v.to(policy_device) for k, v in inputs.items()}
        query = inputs_policy["input_ids"]
        context_length = query.shape[1]
        
        with torch.no_grad():
            # ===== Generate from POLICY model =====
            policy_generation = policy_model.generate(
                **inputs_policy,
                max_new_tokens=max_new_tokens,
                temperature=temperature + 1e-7,
                do_sample=True,
                top_p=1.0,
                pad_token_id=tokenizer.pad_token_id,
                return_dict_in_generate=True,
                output_scores=True,
            )
            
            query_response = policy_generation.sequences
            policy_response = query_response[:, context_length:]
            
            # Decode policy responses
            policy_response_texts = tokenizer.batch_decode(policy_response, skip_special_tokens=True)
            policy_response_texts = [
                post_process_answer(ans) for ans in policy_response_texts
            ]
            all_policy_responses.extend(policy_response_texts)
            
            # ===== Generate from REFERENCE model =====
            if generate_ref_responses:
                inputs_ref = {k: v.to(ref_device) for k, v in inputs.items()}
                ref_generation = ref_model.generate(
                    **inputs_ref,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature + 1e-7,
                    do_sample=True,
                    top_p=1.0,
                    pad_token_id=tokenizer.pad_token_id,
                    return_dict_in_generate=True,
                    output_scores=True,
                )
                ref_response = ref_generation.sequences[:, context_length:]
                ref_response_texts = tokenizer.batch_decode(ref_response, skip_special_tokens=True)
                all_ref_responses.extend(ref_response_texts)
            else:
                all_ref_responses.extend(["[Not generated]"] * len(batch_prompts))
            
            # ===== Compute KL divergence (using policy model's generations) =====
            policy_attention_mask = (query_response != tokenizer.pad_token_id).long()
            policy_output = policy_model(query_response, attention_mask=policy_attention_mask)
            policy_logits = policy_output.logits[:, context_length - 1: -1]
            policy_logits = policy_logits / (temperature + 1e-7)
            policy_logprob = selective_log_softmax(policy_logits, policy_response)
            
            query_response_ref = query_response.to(ref_device)
            ref_attention_mask = (query_response_ref != tokenizer.pad_token_id).long()
            ref_output = ref_model(query_response_ref, attention_mask=ref_attention_mask)
            ref_logits = ref_output.logits[:, context_length - 1: -1]
            ref_logits = ref_logits / (temperature + 1e-7)
            
            ref_logits = ref_logits.to(policy_device)
            ref_logprob = selective_log_softmax(ref_logits, policy_response)
            
            sequence_lengths = first_true_indices(policy_response == tokenizer.pad_token_id) - 1
            response_idxs = torch.arange(policy_response.shape[1], device=policy_device).repeat(policy_response.shape[0], 1)
            padding_mask = response_idxs > sequence_lengths.unsqueeze(1)
            
            policy_logprob = torch.masked_fill(policy_logprob, padding_mask, INVALID_LOGPROB)
            ref_logprob = torch.masked_fill(ref_logprob, padding_mask, INVALID_LOGPROB)
            
            kl_mc = compute_kl(policy_logprob, ref_logprob)
            kl_mc = torch.masked_fill(kl_mc, padding_mask, 0.0).sum(1)
            
            kl_exact_tokenwise = compute_kl(
                policy_logprob, ref_logprob,
                logits_p=policy_logits,
                logits_q=ref_logits
            )

            kl_exact_tokenwise = torch.masked_fill(
                kl_exact_tokenwise, padding_mask, 0.0
            )

            kl_exact = kl_exact_tokenwise.sum(1)


            # ---- Token-level KL aggregation ----

            for b in range(policy_response.shape[0]):
                token_ids = policy_response[b]
                token_kls = kl_exact_tokenwise[b]
                
                pad_mask  = padding_mask[b]

                tokens = tokenizer.convert_ids_to_tokens(token_ids.tolist())

                for tok, kl_val, is_pad in zip(tokens, token_kls.tolist(), pad_mask.tolist()):
                    if is_pad:
                        continue

                    bucket = bucket_token(tok)

                    token_kl_records.append({
                        "token": tok,
                        "bucket": bucket,
                        "kl": float(kl_val),
                    })

            valid_tokens = (~padding_mask).sum(dim=1).clamp_min(1)

            kl_mc_per_token = kl_mc / valid_tokens
            kl_exact_per_token = kl_exact / valid_tokens
            
            all_kl_mc.extend(kl_mc.cpu().numpy())
            all_kl_exact.extend(kl_exact.cpu().numpy())
            all_queries.extend(tokenizer.batch_decode(query, skip_special_tokens=True))
            all_mc_per_token.extend(kl_mc_per_token.cpu().numpy())
            all_exact_per_token.extend(kl_exact_per_token.cpu().numpy())
            
            if i % (batch_size * 4) == 0:
                torch.cuda.empty_cache()
    
    return {
        "kl_mc_mean": np.mean(all_kl_mc),
        "kl_mc_std": np.std(all_kl_mc),
        "kl_exact_mean": np.mean(all_kl_exact),
        "kl_exact_std": np.std(all_kl_exact),
        "kl_mc_per_token_mean": np.mean(all_mc_per_token),
        "kl_mc_per_token_std": np.std(all_mc_per_token),
        "kl_exact_per_token_mean": np.mean(all_exact_per_token),
        "kl_exact_per_token_std": np.std(all_exact_per_token),
        "kl_mc_values": all_kl_mc,
        "kl_exact_values": all_kl_exact,
        "queries": all_queries,
        "policy_responses": all_policy_responses,
        "ref_responses": all_ref_responses,
        # Keep backward compatibility
        "responses": all_policy_responses,
        "bucket_kl_summary": aggregate_kl_by_bucket(token_kl_records)

    }


def save_results(results: dict, config: dict, output_dir: str = "results"):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    adapter_short = config.get("adapter_name", "unknown").split("/")[-1][:30]
    run_dir = os.path.join(output_dir, f"kl_divergence_{adapter_short}_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    
    summary = {
        "kl_mc_mean": float(results["kl_mc_mean"]),
        "kl_mc_std": float(results["kl_mc_std"]),
        "kl_exact_mean": float(results["kl_exact_mean"]),
        "kl_exact_std": float(results["kl_exact_std"]),
        "kl_mc_per_token_mean": float(results["kl_mc_per_token_mean"]),
        "kl_mc_per_token_std": float(results["kl_mc_per_token_std"]),
        "kl_exact_per_token_mean": float(results["kl_exact_per_token_mean"]),
        "kl_exact_per_token_std": float(results["kl_exact_per_token_std"]),
        "bucket_kl_summary": results["bucket_kl_summary"],
        "num_samples": len(results["kl_mc_values"]),
        "timestamp": timestamp,
    }
    
    with open(os.path.join(run_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    
    # Save config without prompts (they can be huge with PDF content)
    config_to_save = {k: v for k, v in config.items() if k != "prompts_used"}
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(config_to_save, f, indent=2)
    
    samples = []
    for i in range(len(results["kl_mc_values"])):
        samples.append({
            "index": i,
            "query": results["queries"][i][:2000],  # Truncate for readability
            "policy_response": results["policy_responses"][i],
            "ref_response": results["ref_responses"][i],
            "kl_mc": float(results["kl_mc_values"][i]),
            "kl_exact": float(results["kl_exact_values"][i]),
        })
    
    with open(os.path.join(run_dir, "samples.json"), "w") as f:
        json.dump(samples, f, indent=2)
    
    np.save(os.path.join(run_dir, "kl_mc_values.npy"), np.array(results["kl_mc_values"]))
    np.save(os.path.join(run_dir, "kl_exact_values.npy"), np.array(results["kl_exact_values"]))
    
    report_lines = [
        "=" * 70,
        "KL DIVERGENCE EXPERIMENT REPORT",
        "=" * 70,
        "",
        "CONFIGURATION:",
        "-" * 40,
        f"  Base Model:     {config.get('model_name', 'N/A')}",
        f"  Adapter:        {config.get('adapter_name', 'N/A')}",
        f"  Query Source:   {config.get('query_source', 'N/A')}",
        f"  Num Prompts:    {config.get('num_prompts', 'N/A')}",
        f"  Max New Tokens: {config.get('max_new_tokens', 'N/A')}",
        f"  Temperature:    {config.get('temperature', 'N/A')}",
        "",
        "RESULTS:",
        "-" * 40,
        f"  MC Estimator KL:     {results['kl_mc_mean']:.4f} ± {results['kl_mc_std']:.4f}",
        f"  Exact (Stepwise) KL: {results['kl_exact_mean']:.4f} ± {results['kl_exact_std']:.4f}",
        "",
        "PER-SAMPLE KL VALUES:",
        "-" * 40,
    ]
    
    for i, (q, policy_r, ref_r, kl_mc, kl_ex) in enumerate(zip(
        config.get("original_questions", results["queries"]),
        results["policy_responses"],
        results["ref_responses"],
        results["kl_mc_values"],
        results["kl_exact_values"]
    )):
        report_lines.extend([
            f"\n--- Sample {i+1} ---",
            f"Question: {str(q)[:300]}{'...' if len(str(q)) > 300 else ''}",
            f"",
            f"POLICY (Fine-tuned) Response:",
            f"{policy_r[:500]}{'...' if len(policy_r) > 500 else ''}",
            f"",
            f"REFERENCE (Base) Response:",
            f"{ref_r[:500]}{'...' if len(ref_r) > 500 else ''}",
            f"",
            f"KL (MC): {kl_mc:.4f}, KL (Exact): {kl_ex:.4f}",
        ])
    
    with open(os.path.join(run_dir, "report.txt"), "w") as f:
        f.write("\n".join(report_lines))
    
    print(f"\nResults saved to: {run_dir}")
    return run_dir


# ============== Main ==============

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter_name", type=str, 
                        default="liuhaozhe6788/mistralai_Mistral-7B-Instruct-v0.3-peftq_proj_k_proj_v_proj_o_proj-bs1-ne1")
    parser.add_argument("--model_name", type=str, default="mistralai/Mistral-7B-Instruct-v0.3")
    parser.add_argument("--policy_device", type=str, default="cuda:0")
    parser.add_argument("--ref_device", type=str, default="cuda:1")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_prompts", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--queries_json", type=str, default=None,
                        help="Path to JSON file with custom queries")
    parser.add_argument("--prompt_template", type=str, default="mistral",
                        choices=["mistral", "llama", "raw"])
    parser.add_argument("--max_context_chars", type=int, default=8000)
    parser.add_argument("--generate_ref_responses", action="store_true", default=True,
                        help="Generate responses from reference model too (default: True)")
    parser.add_argument("--no_ref_responses", action="store_true",
                        help="Skip generating reference model responses (faster)")
    
    args = parser.parse_args()
    
    num_gpus = torch.cuda.device_count()
    print(f"Available GPUs: {num_gpus}")
    for i in range(num_gpus):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    
    if num_gpus < 2:
        print("\nWARNING: Less than 2 GPUs available.")
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
    
    print(f"\nLoading fine-tuned model on {args.policy_device}...")
    policy_model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16,
        device_map=args.policy_device,
    )
    policy_model = PeftModel.from_pretrained(policy_model, args.adapter_name)
    
    # Load queries
    query_source = "default"
    original_questions = []
    
    if args.queries_json:
        print(f"\nLoading queries from: {args.queries_json}")
        query_data = load_queries_from_json(args.queries_json)
        base_dir = os.path.dirname(os.path.abspath(args.queries_json))
        
        prompts = []
        for q in query_data:
            question = q.get("question", q.get("instruction", ""))
            documents = q.get("documents", None)
            
            formatted_prompt = format_prompt_with_context(
                question=question,
                documents=documents,
                base_dir=base_dir,
                prompt_template=args.prompt_template,
                max_context_chars=args.max_context_chars,
            )
            prompts.append(formatted_prompt)
            original_questions.append(question)
        
        query_source = args.queries_json
        print(f"  Loaded {len(prompts)} queries")
    else:
        prompts = [
            "<s>[INST] What is machine learning? [/INST]",
            "<s>[INST] Explain neural networks. [/INST]",
            "<s>[INST] How does gradient descent work? [/INST]",
        ]
        original_questions = prompts
    
    if args.num_prompts is not None:
        prompts = prompts[:args.num_prompts]
        original_questions = original_questions[:args.num_prompts]
    
    print(f"\nComputing KL divergence on {len(prompts)} prompts...")
    generate_ref = not args.no_ref_responses
    print(f"  Generate reference responses: {generate_ref}")
    
    results = compute_kl_between_models(
        policy_model=policy_model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        prompts=prompts,
        policy_device=args.policy_device,
        ref_device=args.ref_device,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
        temperature=args.temperature,
        generate_ref_responses=generate_ref,
    )
    
    print("\n" + "="*60)
    print("KL DIVERGENCE RESULTS")
    print("="*60)
    print(f"MC Estimator KL:     {results['kl_mc_mean']:.4f} ± {results['kl_mc_std']:.4f}")
    print(f"Exact (Stepwise) KL: {results['kl_exact_mean']:.4f} ± {results['kl_exact_std']:.4f}")
    print("="*60)
    
    config = {
        "model_name": args.model_name,
        "adapter_name": args.adapter_name,
        "policy_device": args.policy_device,
        "ref_device": args.ref_device,
        "max_new_tokens": args.max_new_tokens,
        "batch_size": args.batch_size,
        "num_prompts": len(prompts),
        "temperature": args.temperature,
        "query_source": query_source,
        "prompt_template": args.prompt_template,
        "max_context_chars": args.max_context_chars,
        "original_questions": original_questions,
        "num_gpus": num_gpus,
    }
    
    save_results(results, config, output_dir=args.output_dir)