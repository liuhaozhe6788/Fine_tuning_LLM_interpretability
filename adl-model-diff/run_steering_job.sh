#!/bin/bash
#SBATCH --job-name=adl_steering
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

# Run steering analysis
# Steering uses mean differences from logit_lens, so logit_lens must be run first
set -e  # Exit on error

echo "Starting STEERING analysis at $(date)"
echo "Python path: $(which python)"
echo "Working directory: $(pwd)"
echo "Available memory: $(free -h | grep Mem | awk '{print $2}')"
echo "PyTorch CUDA alloc config: $PYTORCH_CUDA_ALLOC_CONF"
echo ""
echo "⚠️  STEERING MODE: Uses mean differences from logit_lens results"
echo "   Make sure logit_lens has been run first!"
echo "   Will test multiple steering strengths per position"
echo ""

# Use CPU (more reliable) - steering generates text which is memory intensive
# Uses 50 samples (same as other experiments) but only 20 prompts for generation
python adl_analysis.py \
    --num_samples 50 \
    --layer 16 \
    --positions 0 1 2 3 4 5 \
    --device cpu \
    --components steering

EXIT_CODE=$?
echo "Steering analysis finished at $(date) with exit code: $EXIT_CODE"

if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Steering analysis complete!"
else
    echo "❌ Steering analysis failed with exit code: $EXIT_CODE"
    exit $EXIT_CODE
fi

