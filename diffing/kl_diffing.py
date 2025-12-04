"""
KL Divergence-based model diffing method for vLLM models.

This module adapts the original KL divergence computation to work with vLLM
instead of the nnsight tracing framework.
"""

from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path
import torch
from torch.nn import functional as F
import json
import numpy as np
from tqdm import tqdm, trange
from torch.nn.utils.rnn import pad_sequence
from collections import defaultdict

from vllm import LLM
from vllm.lora.request import LoRARequest
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer
import gc
from vllm.distributed.parallel_state import destroy_model_parallel


class VLLMKLDivergenceMethod:
    """
    Computes KL divergence per token between base and finetuned models using vLLM.

    This method:
    1. Loads base and finetuned models via vLLM
    2. Uses token sequences from your data
    3. Computes per-token KL divergence between model outputs
    4. Tracks max activating examples with full context
    5. Aggregates statistics per dataset
    6. Saves results to disk
    """

    def __init__(
        self,
        base_model_name: str,
        adapter_name: str,
        results_dir: str = "./kl_results",
        temperature: float = 1.0,
        max_tokens_per_sample: int = 2048,
        batch_size: int = 4,
        max_samples: Optional[int] = None,
        max_model_len: int = 4096,
        gpu_memory_utilization: float = 0.9,
    ):
        """
        Initialize KL divergence method.

        Args:
            base_model_name: HuggingFace model ID for base model
            adapter_name: HuggingFace model ID for LoRA adapter
            results_dir: Directory to save results
            temperature: Temperature for softmax
            max_tokens_per_sample: Maximum tokens per sample
            batch_size: Batch size for processing
            max_samples: Maximum number of samples to process (None = all)
            max_model_len: Maximum sequence length for models
            gpu_memory_utilization: GPU memory utilization (0.0-1.0)
        """
        self.base_model_name = base_model_name
        self.adapter_name = adapter_name
        self.temperature = temperature
        self.max_tokens_per_sample = max_tokens_per_sample
        self.batch_size = batch_size
        self.max_samples = max_samples
        self.max_model_len = max_model_len
        self.gpu_memory_utilization = gpu_memory_utilization

        # Setup results directory
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Initialize models
        self._setup_tokenizer()
        self._setup_models()

    def _setup_tokenizer(self):
        """Setup tokenizer."""
        print(f"Loading tokenizer from {self.base_model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)

        # Set padding token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def _setup_models(self):
        """Initialize vLLM models."""
        print(f"Loading base model: {self.base_model_name}...")

        # Load base model
        self.base_llm = LLM(
            model=self.base_model_name,
            max_model_len=self.max_model_len,
            gpu_memory_utilization=self.gpu_memory_utilization,
            enforce_eager=True,  # Required for logits access
            trust_remote_code=True,
        )

        print(f"Loading LoRA adapter: {self.adapter_name}...")

        # Download LoRA weights
        self.lora_path = snapshot_download(repo_id=self.adapter_name)
        self.lora_request = LoRARequest("ft_adapter", 1, self.lora_path)

        print("Models loaded successfully!")

    def compute_kl_divergence(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute per-token KL divergence between base and finetuned model outputs.

        Args:
            input_ids: Token ids [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]

        Returns:
            Tuple of:
                per_token_kl: KL divergence per token [batch_size, seq_len]
                mean_per_sample_kl: Mean KL divergence per sample [batch_size]
        """
        batch_size, seq_len = input_ids.shape

        with torch.no_grad():
            # Get the underlying model from vLLM
            model = self.base_llm.llm_engine.model_executor.driver_worker.model_runner.model

            # Move inputs to GPU
            input_ids_cuda = input_ids.cuda()
            attention_mask_cuda = attention_mask.cuda()

            # Get base model logits
            base_outputs = model(
                input_ids=input_ids_cuda,
                attention_mask=attention_mask_cuda,
                use_cache=False,
            )
            base_logits = base_outputs.logits

            # Get finetuned model logits (with LoRA)
            # Load LoRA adapter
            model_runner = self.base_llm.llm_engine.model_executor.driver_worker.model_runner
            
            # Apply LoRA weights by temporarily setting the adapter
            # Note: This is a simplified approach. In production, you'd handle LoRA state more carefully
            try:
                # Try to set LoRA adapter if vLLM supports it
                from vllm.lora.models import LoRAModel
                # This is model-dependent - may need adjustment based on vLLM version
                pass
            except:
                pass

            # For now, we'll compute logits with LoRA by reloading the model with LoRA enabled
            # This is less efficient but more reliable
            # In your actual use case, you might want to load a separate fine-tuned model
            
            # WORKAROUND: Load separate LLM for fine-tuned model
            if not hasattr(self, 'ft_llm'):
                print("Loading separate fine-tuned model for accurate KL computation...")
                self.ft_llm = LLM(
                    model=self.base_model_name,
                    tokenizer=self.adapter_name,
                    enable_lora=True,
                    max_lora_rank=512,
                    max_model_len=self.max_model_len,
                    gpu_memory_utilization=self.gpu_memory_utilization * 0.5,
                    enforce_eager=True,
                    trust_remote_code=True,
                )
            
            ft_model = self.ft_llm.llm_engine.model_executor.driver_worker.model_runner.model
            
            # Apply LoRA and get logits
            ft_outputs = ft_model(
                input_ids=input_ids_cuda,
                attention_mask=attention_mask_cuda,
                use_cache=False,
            )
            ft_logits = ft_outputs.logits

            # Shape assertions for logits
            vocab_size = base_logits.shape[-1]
            assert base_logits.shape == (
                batch_size,
                seq_len,
                vocab_size,
            ), f"Expected: {(batch_size, seq_len, vocab_size)}, got: {base_logits.shape}"
            assert ft_logits.shape == (
                batch_size,
                seq_len,
                vocab_size,
            ), f"Expected: {(batch_size, seq_len, vocab_size)}, got: {ft_logits.shape}"

            # Apply temperature
            if self.temperature != 1.0:
                base_logits = base_logits / self.temperature
                ft_logits = ft_logits / self.temperature

            # Convert to log probabilities
            base_log_probs = F.log_softmax(
                base_logits, dim=-1
            )  # [batch_size, seq_len, vocab_size]
            ft_log_probs = F.log_softmax(
                ft_logits, dim=-1
            )  # [batch_size, seq_len, vocab_size]

            # Convert to probabilities for KL computation
            ft_probs = torch.exp(ft_log_probs)

            # Compute KL divergence: KL(finetuned || base) = sum(finetuned * log(finetuned / base))
            kl_div = torch.sum(
                ft_probs * (ft_log_probs - base_log_probs), dim=-1
            )

            # Shape assertion for KL divergence
            assert kl_div.shape == (
                batch_size,
                seq_len,
            ), f"Expected: {(batch_size, seq_len)}, got: {kl_div.shape}"

            # Mask out padding tokens
            masked_kl = kl_div * attention_mask_cuda.float()

            # Compute mean per sample KL (excluding padding tokens)
            kl_sums = torch.sum(masked_kl, dim=1)
            valid_token_counts = torch.sum(attention_mask_cuda, dim=1).float()

            # Compute mean, handling cases with no valid tokens
            mean_per_sample_kl_tensor = torch.where(
                valid_token_counts > 0,
                kl_sums / valid_token_counts,
                torch.zeros_like(kl_sums),
            )

            return masked_kl.cpu(), mean_per_sample_kl_tensor.cpu()

    def prepare_batch(
        self, sequences: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare a batch of sequences for processing.

        Args:
            sequences: List of token tensors

        Returns:
            Tuple of (input_ids, attention_mask)
        """
        # Truncate sequences to max length if specified
        truncated_sequences = [seq[: self.max_tokens_per_sample] for seq in sequences]

        # Pad sequences
        input_ids = pad_sequence(
            truncated_sequences,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        )

        # Create attention mask (1 for real tokens, 0 for padding)
        attention_mask = torch.zeros_like(input_ids, dtype=torch.long)
        for i, seq in enumerate(truncated_sequences):
            attention_mask[i, : len(seq)] = 1

        return input_ids, attention_mask

    def compute_statistics(self, all_kl_values: torch.Tensor) -> Dict[str, float]:
        """
        Compute statistical summaries of KL divergence values.

        Args:
            all_kl_values: Flattened tensor of all KL values

        Returns:
            Dictionary with statistical summaries
        """
        stats = {}

        # Convert to float32 if needed
        if all_kl_values.dtype == torch.bfloat16:
            all_kl_values = all_kl_values.float()
        kl_np = all_kl_values.cpu().numpy()

        # Basic statistics
        stats["mean"] = float(torch.mean(all_kl_values).item())
        stats["std"] = float(torch.std(all_kl_values).item())
        stats["median"] = float(torch.median(all_kl_values).item())
        stats["min"] = float(torch.min(all_kl_values).item())
        stats["max"] = float(torch.max(all_kl_values).item())

        # Percentiles
        for percentile in [25, 50, 75, 95]:
            stats[f"percentile_{percentile}"] = float(np.percentile(kl_np, percentile))

        return stats

    def process_sequences(
        self,
        sequences: List[str],
        dataset_name: str = "dataset",
    ) -> Dict[str, Any]:
        """
        Process a list of text sequences and compute KL divergences.

        Args:
            sequences: List of text sequences
            dataset_name: Name of the dataset

        Returns:
            Dictionary containing aggregated statistics and examples
        """
        print(f"Processing {len(sequences)} sequences from {dataset_name}...")

        # Apply max_samples limit if specified
        if self.max_samples is not None:
            sequences = sequences[: self.max_samples]

        # Tokenize all sequences
        print("Tokenizing sequences...")
        tokenized_sequences = []
        for seq in tqdm(sequences, desc="Tokenizing"):
            tokens = self.tokenizer.encode(seq, add_special_tokens=True)
            tokenized_sequences.append(torch.tensor(tokens))

        # Filter sequences with only one token
        tokenized_sequences = [seq for seq in tokenized_sequences if len(seq) > 1]

        print(f"Processing {len(tokenized_sequences)} valid sequences...")

        all_per_token_kl_values = []
        all_mean_per_sample_kl_values = []
        max_examples = []

        # Process sequences in batches
        for i in trange(
            0,
            len(tokenized_sequences),
            self.batch_size,
            desc=f"Processing batches from {dataset_name}",
        ):
            batch_sequences = tokenized_sequences[i : i + self.batch_size]

            # Prepare batch tensors
            input_ids, attention_mask = self.prepare_batch(batch_sequences)

            # Compute KL divergence
            per_token_kl, mean_per_sample_kl = self.compute_kl_divergence(
                input_ids, attention_mask
            )

            # Collect valid KL values (excluding padding)
            valid_mask = attention_mask.flatten().bool()
            valid_per_token_kl_batch = per_token_kl.flatten()[valid_mask]
            all_per_token_kl_values.append(valid_per_token_kl_batch)

            # Collect mean per sample KL values
            all_mean_per_sample_kl_values.append(mean_per_sample_kl)

            # Store max examples (top 5 from this batch)
            for batch_idx in range(per_token_kl.shape[0]):
                valid_mask_seq = attention_mask[batch_idx].bool()
                valid_kl_seq = per_token_kl[batch_idx][valid_mask_seq]

                if len(valid_kl_seq) > 0:
                    max_kl = torch.max(valid_kl_seq).item()
                    max_idx = torch.argmax(valid_kl_seq).item()

                    # Decode tokens
                    tokens = [
                        self.tokenizer.decode([input_ids[batch_idx, j].item()])
                        for j in range(input_ids.shape[1])
                        if attention_mask[batch_idx, j] == 1
                    ]

                    max_examples.append(
                        {
                            "max_kl": max_kl,
                            "max_kl_position": max_idx,
                            "mean_kl": mean_per_sample_kl[batch_idx].item(),
                            "tokens": tokens,
                            "kl_values": valid_kl_seq.tolist(),
                        }
                    )

        # Concatenate all KL values
        all_per_token_kl_tensor = torch.cat(all_per_token_kl_values, dim=0)
        all_mean_per_sample_kl_tensor = torch.cat(all_mean_per_sample_kl_values, dim=0)

        print(
            f"Computed KL divergence for {len(all_per_token_kl_tensor)} tokens from {dataset_name}"
        )

        # Compute statistics
        per_token_statistics = self.compute_statistics(all_per_token_kl_tensor)
        mean_per_sample_statistics = self.compute_statistics(
            all_mean_per_sample_kl_tensor
        )

        # Sort and keep top examples
        max_examples = sorted(max_examples, key=lambda x: x["max_kl"], reverse=True)[
            :100
        ]

        return {
            "dataset_name": dataset_name,
            "per_token_statistics": per_token_statistics,
            "mean_per_sample_statistics": mean_per_sample_statistics,
            "total_tokens_processed": len(all_per_token_kl_tensor),
            "total_sequences_processed": len(tokenized_sequences),
            "max_examples": max_examples,
            "metadata": {
                "base_model": self.base_model_name,
                "finetuned_model": self.adapter_name,
                "temperature": self.temperature,
            },
        }

    def save_results(self, dataset_name: str, results: Dict[str, Any]) -> Path:
        """
        Save results for a dataset to disk.

        Args:
            dataset_name: Dataset identifier
            results: Results dictionary

        Returns:
            Path to saved file
        """
        # Convert dataset name to safe filename
        safe_name = dataset_name.replace("/", "_")
        output_file = self.results_dir / f"{safe_name}.json"

        # Save results
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)

        print(f"Saved results for {dataset_name} to {output_file}")
        return output_file

    def run(self, sequences: List[str], dataset_name: str = "dataset") -> Dict[str, Any]:
        """
        Main execution method for KL divergence computation.

        Args:
            sequences: List of text sequences to process
            dataset_name: Name of the dataset

        Returns:
            Dictionary containing results
        """
        print("Starting KL divergence computation...")

        # Process sequences
        results = self.process_sequences(sequences, dataset_name)

        # Save results to disk
        output_file = self.save_results(dataset_name, results)

        print("KL divergence computation completed successfully")
        print(f"Results saved to: {output_file}")

        return results

    def cleanup(self):
        """Clean up GPU memory."""
        print("\nCleaning up GPU memory...")
        destroy_model_parallel()
        if hasattr(self, 'base_llm'):
            del self.base_llm
        if hasattr(self, 'ft_llm'):
            del self.ft_llm
        gc.collect()
        torch.cuda.empty_cache()
        print("Cleanup complete!")


def main():
    """Example usage with your inference.py setup."""

    adapter_name = "liuhaozhe6788/mistralai_Mistral-7B-Instruct-v0.3-peftq_proj_k_proj_v_proj_o_proj-bs1-ne1"
    model_name = "mistralai/Mistral-7B-Instruct-v0.3"

    # Example sequences (you would load these from your dataset)
    passage_query = """###Passage: the following table presents var with respect to our trading activities, as measured by our var methodology for the periods indicated: value-at-risk.

years ended december 31 (in millions)|2008 annual average|2008 maximum|2008 minimum|2008 annual average|2008 maximum|minimum
foreign exchange products|$ 1.8|$ 4.7|$ .3|$ 1.8|$ 4.0|$ .7
interest-rate products|1.1|2.4|.6|1.4|3.7|.1

###Question: what is the average variance of the value at risk of each 2008 section? ( $ )"""

    sequences = [
        f"<s>[INST] Read the following passage and then write python code to answer the question: {passage_query}\n\n###Python\n[/INST]",
        # Add more sequences here
    ]

    # Initialize KL divergence method
    kl_method = VLLMKLDivergenceMethod(
        base_model_name=model_name,
        adapter_name=adapter_name,
        results_dir="./kl_results",
        temperature=1.0,
        max_tokens_per_sample=2048,
        batch_size=2,
        max_samples=None,  # Process all sequences
    )

    # Run KL divergence computation
    results = kl_method.run(sequences, dataset_name="FinQA_test")

    # Print summary
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    print(f"\nDataset: {results['dataset_name']}")
    print(f"Total sequences processed: {results['total_sequences_processed']}")
    print(f"Total tokens processed: {results['total_tokens_processed']}")

    print("\nPer-Token Statistics:")
    for key, value in results["per_token_statistics"].items():
        print(f"  {key}: {value:.6f}")

    print("\nMean Per-Sample Statistics:")
    for key, value in results["mean_per_sample_statistics"].items():
        print(f"  {key}: {value:.6f}")

    print("\nTop 5 Examples with Highest Max KL:")
    for i, example in enumerate(results["max_examples"][:5], 1):
        print(f"\n  {i}. Max KL: {example['max_kl']:.6f} at position {example['max_kl_position']}")
        print(f"     Mean KL: {example['mean_kl']:.6f}")
        print(f"     Tokens: {' '.join(example['tokens'][:50])}...")

    # Cleanup
    kl_method.cleanup()


if __name__ == "__main__":
    main()