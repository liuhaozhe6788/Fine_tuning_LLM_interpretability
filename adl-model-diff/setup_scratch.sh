#!/bin/bash
# Setup script to configure scratch space for ADL analysis
# Run this before running any ADL scripts: source setup_scratch.sh

# Check if scratch exists (ETH cluster uses /work/scratch/{user})
if [ -d "/work/scratch/$USER" ]; then
    SCRATCH_BASE="/work/scratch/$USER"
    echo "✅ Using scratch space: $SCRATCH_BASE (100GB limit)"
elif [ -d "/scratch/$USER" ]; then
    SCRATCH_BASE="/scratch/$USER"
    echo "✅ Using scratch space: $SCRATCH_BASE"
elif [ -d "/work/$USER" ]; then
    SCRATCH_BASE="/work/$USER"
    echo "✅ Using work space: $SCRATCH_BASE"
else
    SCRATCH_BASE="$HOME"
    echo "⚠️  Scratch/work not available, using home: $SCRATCH_BASE"
fi

# Create directories
mkdir -p "$SCRATCH_BASE/hf-cache"
mkdir -p "$SCRATCH_BASE/adl-results"
mkdir -p "$SCRATCH_BASE/adl-data"

# Set Hugging Face cache to scratch
export HF_HOME="$SCRATCH_BASE/hf-cache"
export TRANSFORMERS_CACHE="$SCRATCH_BASE/hf-cache"
export HF_HOME="$SCRATCH_BASE/hf-cache"

echo "Environment variables set:"
echo "  HF_HOME=$HF_HOME"
echo "  TRANSFORMERS_CACHE=$TRANSFORMERS_CACHE"
echo ""
echo "Directories created:"
echo "  HF Cache: $SCRATCH_BASE/hf-cache"
echo "  Results: $SCRATCH_BASE/adl-results"
echo "  Data: $SCRATCH_BASE/adl-data"
echo ""
echo "✅ Setup complete! Models and data will be stored in scratch space."

