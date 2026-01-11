# Activation Difference Lens (ADL) Analysis

This directory contains the implementation of Activation Difference Lens analysis for comparing the base and fine-tuned Mistral-7B-Instruct models on FinQA.

## Overview

ADL analyzes differences between base and fine-tuned models by:
1. **Logit Lens**: Projecting activation differences through the output head to see what tokens they predict
2. **Patchscope**: Projecting activation differences through the model to understand their semantic meaning
3. **Token Relevance**: Identifying which tokens are most relevant to the differences
4. **Causal Effect**: Measuring the causal impact of activation differences

## Models

- **Base Model**: `mistralai/Mistral-7B-Instruct-v0.3`
- **Fine-tuned Model**: `liuhaozhe6788/mistralai_Mistral-7B-Instruct-v0.3-FinQA-lora`

## Dataset

- **Source**: FinQA training data (`../data/clean_with_code/FinQA/finqa_train_generated_filtered.csv`)
- **Format**: `prompt + generated_code`
- **Samples**: 1024 (configurable)
- **Random seed**: 49 (for reproducibility and consistency with crosscoder)

## Configuration

- **Layer**: 16 (middle layer, 0.5 depth for 32-layer model) - matches crosscoder
- **Positions**: First 6 tokens [0, 1, 2, 3, 4, 5]
- **Sequence length**: 128 tokens (analyzing first 6 positions)
- **Random seed**: 49 (matches crosscoder; KL experiments may use different seed)
- **Dataset**: FinQA train split (`finqa_train_generated_filtered.csv`)

## Consistency with Other Experiments

This implementation is designed to be consistent with:
- **Crosscoder experiments**: Same layer (16), seed (49), dataset, and data format
- **KL divergence experiments**: 
  - Same models (base and fine-tuned)
  - Same dataset path available
  - **Note**: KL divergence uses custom queries from JSON files rather than sampling from FinQA, so no seed alignment needed for data sampling
  - KL uses `batch_size: 1-4`, `temperature: 1.0`, `max_new_tokens: 128-256`
  - ADL uses `batch_size: 2` for causal effect, focuses on activation analysis rather than generation

**Key differences:**
- **ADL**: Samples from FinQA train data with seed 49, analyzes activations at layer 16
- **KL**: Uses custom queries/prompts, computes KL divergence on generated outputs
- **Crosscoder**: Uses same FinQA data with seed 49, trains autoencoders on activations

## Usage

### Local Development

```bash
# Install dependencies
pip install -r ../requirements.txt

# Run analysis
python adl_analysis.py
```

### Cluster Execution

**Full Analysis (all components):**
```bash
sbatch run_full_adl_job.sh
```
- Includes: logit_lens, patchscope, token_relevance, causal_effect
- **Estimated time**: 10-20+ hours (causal effect is very slow)

**Fast Analysis (skip causal effect):**
```bash
sbatch run_fast_adl_job.sh
```
- Includes: logit_lens, patchscope, token_relevance
- **Estimated time**: 1-2 hours
- **Recommended for initial runs**

**Minimal Analysis (logit lens only):**
```bash
sbatch run_minimal_adl_job.sh
```
- Includes: logit_lens, token_relevance
- **Estimated time**: < 10 minutes
- **Fastest option for quick results**

### Component Speed Analysis

| Component | Forward Passes | Estimated Time | Can Skip? |
|-----------|---------------|----------------|-----------|
| **Logit Lens** | ~18 (small projections) | < 1 min | No (core component) |
| **Token Relevance** | 0 (analyzes results) | < 1 min | No (fast, useful) |
| **Patchscope** | ~240 (full model) | 30-60 min | Yes (optional) |
| **Causal Effect** | ~12,000 (full model) | 10-20+ hours | **Yes (very slow)** |

**Recommendation**: Start with `run_fast_adl_job.sh` to get results quickly, then run causal effect separately if needed.

## Output Structure

```
results/
└── Mistral-7B-Instruct-v0.3_1024_samples/
    ├── logit_lens/
    │   ├── logit_lens_summary.json
    │   └── logit_lens_per_position.csv
    ├── patchscope/
    │   ├── patchscope_summary.json
    │   └── patchscope_per_position.csv
    ├── token_relevance/
    │   ├── token_relevance_scores.json
    │   └── token_relevance_ranked.csv
    ├── causal_effect/
    │   ├── causal_effects.json
    │   └── causal_effects_summary.csv
    └── summaries/
        ├── overall_summary.json
        └── key_findings.md
```

## Files

- `adl_analysis.py` - Main analysis script
- `logit_lens.py` - Logit lens implementation
- `patchscope.py` - Patchscope implementation
- `token_relevance.py` - Token relevance analysis
- `causal_effect.py` - Causal effect measurement
- `utils.py` - Shared utilities
- `config.py` - Configuration management
- `run_adl.sh` - SLURM job script

## Consistency with Other Methods

This implementation is designed to be consistent with:
- **Crosscoder**: Same dataset, layer 16, same random seed
- **KL Divergence**: Same prompts/data format

Results can be compared across all three methods for comprehensive model analysis.

