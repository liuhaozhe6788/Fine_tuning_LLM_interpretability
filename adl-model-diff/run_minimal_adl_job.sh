#!/bin/bash
#SBATCH --job-name=adl_minimal
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --account=deep_learning
#SBATCH --partition=jobs
#SBATCH --mem=48G
#SBATCH --time=100:00:00

# Load modules system and CUDA 12.8 (for PyTorch compatibility)
. /etc/profile.d/modules.sh
module add cuda/12.8

# Activate venv
source ~/venvs/diffing-env/bin/activate

# Setup scratch space
source ~/Fine_tuning_LLM_interpretability/adl-model-diff/setup_scratch.sh

# Set HF token (should be set in environment or passed via --hf-token)
# export HF_TOKEN="your_token_here"

# Set PyTorch memory allocator to reduce fragmentation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Navigate to directory
cd ~/Fine_tuning_LLM_interpretability/adl-model-diff

# Run MINIMAL ADL analysis (only logit lens + token relevance)
# This is the fastest option - skips patchscope and causal effect
set -e  # Exit on error

echo "Starting MINIMAL ADL analysis at $(date)"
echo "Python path: $(which python)"
echo "Working directory: $(pwd)"
echo "Available memory: $(free -h | grep Mem | awk '{print $2}')"
echo "PyTorch CUDA alloc config: $PYTORCH_CUDA_ALLOC_CONF"
echo ""
echo "⚠️  MINIMAL MODE: Only running logit_lens + token_relevance"
echo "   Skipping: patchscope (240 forward passes), causal_effect (12,000 forward passes)"
echo ""

# Use CPU for very small runs (more reliable than GPU OOM)
# HF activations disabled due to shape mismatch
# Very small sample size (20) - 7B model needs ~14GB just for weights, leaving little room
python adl_analysis.py \
    --num_samples 20 \
    --layer 16 \
    --positions 0 1 2 3 4 5 \
    --device cpu \
    --components logit_lens token_relevance

EXIT_CODE=$?
echo "ADL analysis finished at $(date) with exit code: $EXIT_CODE"

if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Minimal ADL analysis complete!"
    echo ""
    echo "To run additional components later:"
    echo "  Patchscope: python adl_analysis.py --components patchscope"
    echo "  Causal Effect: python adl_analysis.py --components causal_effect"
else
    echo "❌ ADL analysis failed with exit code: $EXIT_CODE"
    exit $EXIT_CODE
fi

