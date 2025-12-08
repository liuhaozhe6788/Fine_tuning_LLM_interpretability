# CrossCoder Training Plan

## Overview
Train a CrossCoder autoencoder to learn shared latent representations between a base model and a fine-tuned model. The CrossCoder compresses activations from both models into a shared dictionary space, enabling mechanistic interpretability analysis of model differences.

## Prerequisites

### 1. Activation Collection
- ✅ Collect activations from both base model and fine-tuned model
- ✅ Activations should be from the same layer (e.g., layer 16 MLP output)
- ✅ Activations should be aligned (same prompts, same token positions)
- ✅ Store activations as HuggingFace datasets or have them ready for streaming
- ✅ Normalize activations using scaling factors (estimated norm scaling)

### 2. Data Requirements
- Base model activations dataset:  `[num_tokens, d_model]`
- Fine-tuned model activations dataset: `[num_tokens, d_model]`
- Both datasets should have the same number of tokens
- Datasets should be shuffled and aligned

## Training Loop

### 1. Initialization
- Load base model and fine-tuned model with nnsight
- Load activation datasets (or set up Buffer for on-the-fly collection)
- Initialize CrossCoder with config
- Initialize Adam optimizer with specified learning rate and betas
- Initialize LambdaLR scheduler
- Initialize wandb for logging

### 2. Training Steps
For each training step:
1. **Sample batch**: Get aligned activations from both models
   - Shape: `[batch_size, 2, d_model]` (2 = base model, fine-tuned model)
2. **Forward pass**: 
   - Encode: `[batch, 2, d_model]` → `[batch, d_hidden]`
   - Decode: `[batch, d_hidden]` → `[batch, 2, d_model]`
3. **Compute losses**:
   - L2 reconstruction loss: MSE between input and reconstruction
   - L1 sparsity loss: Weighted sum of activations (sparsity penalty)
   - L0 loss: Count of active features (for monitoring)
   - Explained variance: Per model and overall
4. **Backward pass**:
   - Total loss = L2_loss + l1_coeff * L1_loss
   - Gradient clipping (max_norm=1.0)
   - Optimizer step
   - Scheduler step
5. **Logging**: Log metrics to wandb every `log_every` steps
6. **Checkpointing**: Save model, optimizer, scheduler every `save_every` steps

### 3. Metrics to Track
- `loss`: Total training loss
- `l2_loss`: Reconstruction loss
- `l1_loss`: Sparsity loss
- `l0_loss`: Number of active features (sparsity)
- `l1_coeff`: Current L1 coefficient (for monitoring schedule)
- `lr`: Current learning rate
- `explained_variance`: Overall explained variance
- `explained_variance_A`: Explained variance for base model
- `explained_variance_B`: Explained variance for fine-tuned model

## Implementation Options

### Pre-collected Activations (Current Implementation)
- Load activations from HuggingFace datasets
- Simpler, faster training (no model forward passes during training)
- Requires pre-collection step
- Good for large-scale training

### Crosscoder model selection:
- vanilla crosscoder
- crosscoder with batch topk activation masking
- add auxillary loss
## Validation/Evaluation

### During Training
- Monitor explained variance (should increase)
- Monitor L0 loss (should decrease, indicating sparsity)
- Monitor reconstruction loss (should decrease)

