"""
Main ADL analysis script.

Runs all ADL components:
1. Logit Lens
2. Patchscope (if enabled)
3. Token Relevance (if enabled)
4. Causal Effect (if enabled)
"""
import argparse
from pathlib import Path
from config import ADLConfig
from logit_lens import run_logit_lens_analysis
from patchscope import run_patchscope_analysis
from token_relevance import run_token_relevance_analysis
from causal_effect import run_causal_effect_analysis
from steering import run_steering_analysis


def main():
    parser = argparse.ArgumentParser(description="Run ADL analysis")
    parser.add_argument(
        "--num_samples",
        type=int,
        default=1024,
        help="Number of samples to analyze (default: 1024)"
    )
    parser.add_argument(
        "--components",
        nargs="+",
        default=["logit_lens"],
        choices=["logit_lens", "patchscope", "token_relevance", "causal_effect", "steering"],
        help="Which components to run (default: logit_lens)"
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=16,
        help="Layer to analyze (default: 16)"
    )
    parser.add_argument(
        "--positions",
        nargs="+",
        type=int,
        default=[0, 1, 2, 3, 4, 5],
        help="List of token positions to analyze (default: 0 1 2 3 4 5)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device to use (e.g., 'cuda:0' or 'cpu')"
    )
    parser.add_argument(
        "--disable_quantization",
        action="store_true",
        help="Disable 8-bit quantization for GPU models"
    )
    
    args = parser.parse_args()
    
    # Determine device
    import torch
    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("⚠️  CUDA not available, falling back to CPU.")
        device = "cpu"
    
    # Create config
    config = ADLConfig(
        num_samples=args.num_samples,
        layer=args.layer,
        positions=args.positions,
        device=device,
        use_quantization=not args.disable_quantization,
    )
    
    print("=" * 60)
    print("ADL Analysis Configuration")
    print("=" * 60)
    print(f"Models:")
    print(f"  Base: {config.base_model_id}")
    print(f"  Fine-tuned: {config.ft_model_id}")
    print(f"Dataset: {config.dataset_path}")
    print(f"Samples: {config.num_samples}")
    print(f"Layer: {config.layer}")
    print(f"Positions: {config.positions}")
    print(f"Device: {config.device}")
    print(f"Quantization Enabled: {config.use_quantization}")
    print(f"Results dir: {config.results_dir}")
    print(f"Components: {args.components}")
    print("=" * 60)
    print()
    
    # Run components
    if "logit_lens" in args.components:
        print("\n" + "=" * 60)
        print("Running Logit Lens Analysis")
        print("=" * 60)
        run_logit_lens_analysis(config)
    
    if "patchscope" in args.components:
        print("\n" + "=" * 60)
        print("Running Patchscope Analysis")
        print("=" * 60)
        run_patchscope_analysis(config)
    
    if "token_relevance" in args.components:
        print("\n" + "=" * 60)
        print("Running Token Relevance Analysis")
        print("=" * 60)
        run_token_relevance_analysis(config)
    
    if "causal_effect" in args.components:
        print("\n" + "=" * 60)
        print("Running Causal Effect Analysis")
        print("=" * 60)
        run_causal_effect_analysis(config)
    
    if "steering" in args.components:
        print("\n" + "=" * 60)
        print("Running Steering Analysis")
        print("=" * 60)
        run_steering_analysis(config)
    
    print("\n" + "=" * 60)
    print("✅ ADL Analysis Complete!")
    print("=" * 60)
    print(f"Results saved to: {config.results_dir}")


if __name__ == "__main__":
    import sys
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error in ADL analysis: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)

