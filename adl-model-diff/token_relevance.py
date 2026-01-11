"""
Token Relevance Analysis for ADL.

Simplified version that:
1. Loads tokens from logit lens results
2. Computes frequent tokens from the fine-tuning dataset
3. Analyzes which tokens are relevant to the fine-tuning domain
"""
import torch
from pathlib import Path
from typing import List, Dict, Any, Tuple
from collections import Counter
import json
import pandas as pd
from tqdm import tqdm

from utils import (
    load_finqa_data,
    save_json,
    save_csv,
)
from config import ADLConfig


COMMON_WORDS = {
    "the", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by",
    "from", "up", "about", "into", "through", "during", "before", "after",
    "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
    "do", "does", "did", "will", "would", "could", "should", "may", "might",
    "must", "can", "this", "that", "these", "those", "a", "an", "ing",
}


def _is_generic_token(token: str) -> bool:
    """Check if a token is generic/common."""
    # Remove tokenizer artifacts
    clean_token = token.replace("▁", "").replace("Ġ", "").replace("Ċ", "").strip()
    
    if len(clean_token) <= 1:
        return True
    
    # Pure punctuation
    import re
    if re.match(r"^[^\w\s]+$", clean_token):
        return True
    
    # Common suffixes/prefixes
    if clean_token.lower() in {"'s", "'t", "'re", "'ve", "'ll", "'d", "'m", "ing", "ion", "ly"}:
        return True
    
    # Whitespace only
    if re.match(r"^[\s\n\r\t]+$", clean_token):
        return True
    
    return clean_token.lower() in COMMON_WORDS


def compute_frequent_tokens(
    texts: List[str],
    tokenizer,
    num_tokens: int = 100,
    min_count: int = 5
) -> List[str]:
    """
    Compute frequent non-generic tokens from the dataset.
    
    Args:
        texts: List of text strings
        tokenizer: Tokenizer to use
        num_tokens: Number of top frequent tokens to return
        min_count: Minimum occurrence count
        
    Returns:
        List of frequent tokens (sorted by frequency)
    """
    all_tokens = []
    for text in tqdm(texts, desc="Tokenizing for frequent tokens"):
        tokens = tokenizer.tokenize(text)
        all_tokens.extend(tokens)
    
    # Count tokens
    counts = Counter(all_tokens)
    
    # Filter: non-generic and above min_count
    domain_tokens = [
        (tok, cnt) for tok, cnt in counts.items()
        if (not _is_generic_token(tok)) and cnt >= min_count
    ]
    
    # Sort by frequency
    domain_tokens.sort(key=lambda x: x[1], reverse=True)
    
    # Return top tokens
    frequent = [tok for tok, _ in domain_tokens[:num_tokens]]
    return frequent


def analyze_token_relevance_for_position(
    config: ADLConfig,
    position: int,
    logit_lens_results: Dict[str, Any],
    frequent_tokens: List[str],
    output_dir: Path
) -> Dict[str, Any]:
    """
    Analyze token relevance for a specific position.
    
    Args:
        config: ADLConfig instance
        position: Token position
        logit_lens_results: Results from logit lens (for this position)
        frequent_tokens: List of frequent tokens from fine-tuning dataset
        output_dir: Directory to save results
        
    Returns:
        Dictionary with relevance analysis
    """
    results = {}
    
    # Analyze each variant (diff, base, ft)
    for variant in ["diff", "base", "ft"]:
        if variant not in logit_lens_results:
            continue
        
        variant_data = logit_lens_results[variant]
        tokens = variant_data.get("tokens", [])
        probabilities = variant_data.get("probabilities", [])
        
        # Compute relevance metrics
        # 1. Fraction of tokens that appear in frequent tokens
        in_frequent = sum(1 for t in tokens if t in frequent_tokens)
        frequent_fraction = in_frequent / len(tokens) if tokens else 0.0
        
        # 2. Weighted fraction (weighted by probability)
        total_weight = sum(probabilities)
        relevant_weight = sum(
            prob for tok, prob in zip(tokens, probabilities)
            if tok in frequent_tokens
        )
        weighted_fraction = relevant_weight / total_weight if total_weight > 0 else 0.0
        
        # 3. Filter out generic tokens and recompute
        non_generic_tokens = [t for t in tokens if not _is_generic_token(t)]
        non_generic_probs = [
            prob for tok, prob in zip(tokens, probabilities)
            if not _is_generic_token(tok)
        ]
        
        non_generic_in_frequent = sum(
            1 for t in non_generic_tokens if t in frequent_tokens
        )
        non_generic_fraction = (
            non_generic_in_frequent / len(non_generic_tokens)
            if non_generic_tokens else 0.0
        )
        
        results[variant] = {
            "total_tokens": len(tokens),
            "non_generic_tokens": len(non_generic_tokens),
            "in_frequent_tokens": in_frequent,
            "frequent_fraction": frequent_fraction,
            "weighted_fraction": weighted_fraction,
            "non_generic_fraction": non_generic_fraction,
            "top_tokens": tokens[:config.token_relevance_k],
            "top_tokens_in_frequent": [
                t for t in tokens[:config.token_relevance_k]
                if t in frequent_tokens
            ],
        }
    
    # Save results
    output_file = output_dir / f"token_relevance_layer_{config.layer}_pos_{position}.json"
    save_json(results, output_file)
    
    return results


def run_token_relevance_analysis(config: ADLConfig) -> None:
    """
    Main function to run token relevance analysis.
    """
    print("\n" + "="*50)
    print("Starting Token Relevance Analysis")
    print("="*50 + "\n")
    
    # Load logit lens results
    logit_lens_dir = config.results_dir / "logit_lens"
    summary_file = logit_lens_dir / "logit_lens_summary.json"
    
    if not summary_file.exists():
        print(f"⚠️  Logit lens results not found at {summary_file}")
        print("   Please run logit lens analysis first.")
        return
    
    print(f"Loading logit lens results from {summary_file}...")
    with open(summary_file, 'r') as f:
        logit_lens_results = json.load(f)
    print(f"✅ Loaded results for {len(logit_lens_results)} positions")
    
    # Load fine-tuning dataset to compute frequent tokens
    print(f"\nLoading fine-tuning dataset to compute frequent tokens...")
    texts = load_finqa_data(
        config.dataset_path,
        num_samples=min(10000, len(pd.read_csv(config.dataset_path))),  # Use more samples for frequent tokens
        random_seed=config.random_seed
    )
    print(f"✅ Loaded {len(texts)} samples for frequent token analysis")
    
    # Get tokenizer (load a model temporarily just for tokenizer)
    from utils import load_model
    ft_model = load_model(config.ft_model_id, device="cpu", use_quantization=False)  # Use CPU just for tokenizer
    tokenizer = ft_model.tokenizer
    
    # Compute frequent tokens
    print("\nComputing frequent tokens from fine-tuning dataset...")
    frequent_tokens = compute_frequent_tokens(
        texts,
        tokenizer,
        num_tokens=100,
        min_count=5
    )
    print(f"✅ Found {len(frequent_tokens)} frequent tokens")
    print(f"   Examples: {frequent_tokens[:10]}")
    
    # Delete model (we only needed tokenizer)
    del ft_model
    import gc
    gc.collect()
    
    # Analyze each position
    print(f"\nAnalyzing token relevance for positions {config.positions}...")
    output_dir = config.results_dir / "token_relevance"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_results = {}
    for pos in tqdm(config.positions, desc="Processing positions"):
        pos_str = str(pos)
        if pos_str not in logit_lens_results:
            print(f"  ⚠️  Position {pos} not found in logit lens results, skipping")
            continue
        
        results = analyze_token_relevance_for_position(
            config,
            pos,
            logit_lens_results[pos_str],
            frequent_tokens,
            output_dir
        )
        all_results[pos] = results
    
    # Save overall summary
    summary_file = config.results_dir / "summaries" / "token_relevance_summary.json"
    save_json(all_results, summary_file)
    
    # Save frequent tokens for reference
    frequent_tokens_file = output_dir / "frequent_tokens.json"
    save_json({"frequent_tokens": frequent_tokens}, frequent_tokens_file)
    
    print(f"\n✅ Token relevance analysis complete!")
    print(f"   Results saved to: {output_dir}")
    print(f"   Summary saved to: {summary_file}")
    
    # Print summary statistics
    print("\n" + "="*50)
    print("Token Relevance Summary")
    print("="*50)
    for pos, results in all_results.items():
        print(f"\nPosition {pos}:")
        for variant in ["diff", "base", "ft"]:
            if variant in results:
                data = results[variant]
                print(f"  {variant}:")
                print(f"    Frequent fraction: {data['frequent_fraction']:.3f}")
                print(f"    Weighted fraction: {data['weighted_fraction']:.3f}")
                print(f"    Non-generic fraction: {data['non_generic_fraction']:.3f}")
                print(f"    Top tokens in frequent: {len(data['top_tokens_in_frequent'])}/{config.token_relevance_k}")


if __name__ == "__main__":
    # Example usage
    config = ADLConfig(
        num_samples=10,
        positions=[0, 1, 2],
    )
    run_token_relevance_analysis(config)

