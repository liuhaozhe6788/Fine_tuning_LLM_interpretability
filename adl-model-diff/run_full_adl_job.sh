#!/bin/bash
#SBATCH --job-name=adl_full
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

# Set PyTorch memory allocator to reduce fragmentation (as suggested in OOM error)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Navigate to directory
cd ~/Fine_tuning_LLM_interpretability/adl-model-diff

# Run full ADL analysis with all components
# Add error handling and verbose output
set -e  # Exit on error

echo "Starting FULL ADL analysis at $(date)"
echo "Python path: $(which python)"
echo "Working directory: $(pwd)"
echo "Available memory: $(free -h | grep Mem | awk '{print $2}')"
echo "PyTorch CUDA alloc config: $PYTORCH_CUDA_ALLOC_CONF"
echo ""
echo "⚠️  FULL MODE: Running all components including Causal Effect"
echo "   Components: logit_lens, patchscope, token_relevance, causal_effect"
echo "   Causal effect is very slow (~12,000 forward passes per sample)"
echo "   Estimated time: 10-20+ hours for 50 samples"
echo ""
echo "Configuration:"
echo "  - Samples: 50"
echo "  - Device: CPU (8-bit quantization enabled)"
echo "  - Sequence length: 64 tokens (reduced for memory efficiency)"
echo "  - Processing: One sample at a time with aggressive memory clearing"
echo ""
python adl_analysis.py \
    --num_samples 50 \
    --layer 16 \
    --positions 0 1 2 3 4 5 \
    --device cpu \
    --components logit_lens patchscope token_relevance causal_effect

EXIT_CODE=$?
echo "ADL analysis finished at $(date) with exit code: $EXIT_CODE"

if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ ADL analysis complete!"
else
    echo "❌ ADL analysis failed with exit code: $EXIT_CODE"
    exit $EXIT_CODE
fi

