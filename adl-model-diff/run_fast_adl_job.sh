#!/bin/bash
#SBATCH --job-name=adl_fast
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

# Run fast ADL analysis (skips slow causal effect)
# Add error handling and verbose output
set -e  # Exit on error

echo "Starting FAST ADL analysis at $(date)"
echo "Python path: $(which python)"
echo "Working directory: $(pwd)"
echo "Available memory: $(free -h | grep Mem | awk '{print $2}')"
echo "PyTorch CUDA alloc config: $PYTORCH_CUDA_ALLOC_CONF"
echo ""
echo "⚠️  FAST MODE: Skipping Causal Effect (slowest component)"
echo "   Running: logit_lens, patchscope, token_relevance"
echo ""

# Use CPU (more reliable) - can try GPU if you want, but CPU worked for 20 samples
# Start with 50 samples - can increase if it works
# Includes patchscope which adds ~240 forward passes per sample
python adl_analysis.py \
    --num_samples 50 \
    --layer 16 \
    --positions 0 1 2 3 4 5 \
    --device cpu \
    --components logit_lens patchscope token_relevance

EXIT_CODE=$?
echo "ADL analysis finished at $(date) with exit code: $EXIT_CODE"

if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ ADL analysis complete!"
    echo ""
    echo "Note: Causal Effect was skipped. Run it separately if needed:"
    echo "  python adl_analysis.py --components causal_effect"
else
    echo "❌ ADL analysis failed with exit code: $EXIT_CODE"
    exit $EXIT_CODE
fi

