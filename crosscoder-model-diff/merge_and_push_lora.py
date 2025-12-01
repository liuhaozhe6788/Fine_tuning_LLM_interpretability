"""
Script to merge LoRA adapter with base Mistral model and push to HuggingFace Hub
"""
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os
import torch
# Configuration
BASE_MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"
LORA_ADAPTER_ID = "liuhaozhe6788/mistralai_Mistral-7B-Instruct-v0.3-peftq_proj_k_proj_v_proj_o_proj-bs1-ne1"
# Change this to your desired repo name
OUTPUT_MODEL_ID = "liuhaozhe6788/mistralai_Mistral-7B-Instruct-v0.3-FinQA-lora"

# Login to HuggingFace
hf_token = os.environ.get("HF_TOKEN")

device = 'cuda:0'

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID,
    torch_dtype=torch.bfloat16
)
tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL_ID,
    padding_side="left"
)

tokenizer.padding_side = "left"

# Load LoRA adapter
lora_model = PeftModel.from_pretrained(
    base_model,
    LORA_ADAPTER_ID,
    device_map="auto",
    torch_dtype=torch.bfloat16
)

# Merge the adapter weights into the base model
merged_model = lora_model.merge_and_unload()

# Push model and tokenizer
print(f"\nPushing model to {OUTPUT_MODEL_ID}...")
merged_model.push_to_hub(
    OUTPUT_MODEL_ID,
    private=False,
    max_shard_size="5GB",
    safe_serialization=True,  # Use safetensors format
)
tokenizer.push_to_hub(
    OUTPUT_MODEL_ID,
)
print(f"✓ Successfully pushed merged model to HuggingFace Hub")
