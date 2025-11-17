import os
import gc
import sys

from datasets.info import Path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import hashlib
import subprocess as sp
from typing import Optional, List, Union, Dict, Tuple
from tqdm import tqdm

import numpy as np
import pandas as pd
from datasets import load_dataset, Dataset
from peft import AutoPeftModelForCausalLM, prepare_model_for_kbit_training, LoraConfig, get_peft_model, PeftConfig
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments

from config.generation_config import default_config

learning_rate_map_dict = {
    "unsloth/mistral-7b-v0.2-bnb-4bit": 2.5e-5,
    "unsloth/mistral-7b-instruct-v0.2-bnb-4bit": 2.5e-5,
    "mistralai/Mistral-7B-Instruct-v0.3": 2.5e-5,
    "microsoft/Orca-2-7b": 5e-5,
    "microsoft/Orca-2-13b": 5e-5,
    "microsoft/Phi-3-mini-128k-instruct": 1e-4,
    "microsoft/Phi-3-medium-128k-instruct": 1e-4,
}
def load_model_and_tokenizer(
    model_id: str,
    load_in_4bit: bool = False,
    train_mode: bool = True,
    peft_config: Optional[PeftConfig] = None,
    dtype: Optional[str] = "auto",
    device: str = "auto",
    attn_implementation: str = "sdpa",
    padding_side: str = "right",
):
    """
    Load the model and tokenizer from huggingface.
    Args:
        model_id: str
        load_in_4bit: bool -  whether to use 4bit quantization to reduce memory usage.
            # 4bit pre quantized models we support for 4x faster downloading + no OOMs.
            fourbit_models = [
                "unsloth/mistral-7b-bnb-4bit",
                "unsloth/mistral-7b-v0.2-bnb-4bit", # New Mistral 32K base model
                "unsloth/mistral-7b-instruct-v0.2-bnb-4bit",
                "unsloth/llama-2-7b-bnb-4bit",
                "unsloth/llama-2-13b-bnb-4bit",
                "unsloth/codellama-34b-bnb-4bit",
                "unsloth/tinyllama-bnb-4bit",
                "unsloth/gemma-7b-bnb-4bit", # New Google 6 trillion tokens model 2.5x faster!
                "unsloth/gemma-2b-bnb-4bit",
            ] # More models at https://huggingface.co/unsloth
        train_mode: bool - whether to train the model.
        peft_config: Optional[PeftConfig] - the peft configuration to use.
        dtype: torch.dtype - default to None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+.
        device: str - default to auto.
        attn_implementation: str - default to sdpa.
        padding_side: str - default to right.
    """
    if load_in_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    else:
        bnb_config = None
    if peft_config is not None:
        try:
            model = AutoPeftModelForCausalLM.from_pretrained(
                model_id,
                is_trainable=train_mode,
                config=peft_config,
                quantization_config=bnb_config,
                device_map=device,
                torch_dtype=dtype,
                attn_implementation=attn_implementation,
            )
            tokenizer = prepare_tokenizer(model)
        except ValueError:
            print("Failed to load model with AutoPeftModelForCausalLM, now attempting with AutoModelForCausalLM.")
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                quantization_config=bnb_config,
                device_map=device,
                torch_dtype=dtype,
                attn_implementation=attn_implementation,
            )
            tokenizer = prepare_tokenizer(model, padding_side=padding_side)
            if train_mode:
                # If we are not training the model, we do not want to load it in peft mode
                model = prepare_peft_model(model, peft_config=peft_config)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map=device,
            torch_dtype=dtype,
            attn_implementation=attn_implementation,
        )
        tokenizer = prepare_tokenizer(model, padding_side=padding_side)
    print(f"Loaded model on device {model.device} with dtype {model.dtype}.")

    torch.cuda.empty_cache()
    gc.collect()

    return model, tokenizer


def prepare_tokenizer(model, set_pad_token=True, padding_side="right"):
    tokenizer = AutoTokenizer.from_pretrained(model.config._name_or_path)

    tokenizer.padding_side = padding_side 
    if set_pad_token and tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def prepare_peft_model(
    model, peft_config, target_modules: List[str] = ["q_proj", "k_proj", "v_proj", "o_proj"], **lora_config_kwargs
):
    """
    Args:
        target modules - subset of ["q_proj", "k_proj", "v_proj", "o_proj", "fc1", "fc2", "gate_proj", "up_proj", "down_proj"]
    """
    model.gradient_checkpointing_disable()
    model = prepare_model_for_kbit_training(model)  # model becomes float32 instead of bfloat16
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    return model

def format_prompts(
    examples: Union[Dataset, dict],
    prompt_template: str,
) -> List[str]:
    """
    Construct a prompt for each example in examples using the prompt_template.

    Args:
        examples - a dataset containing columns ["context", "query", "weight_context", "answer"],
        prompt_template - the prompt template for which to fill out with examples

    Returns:
        a list of prompts that combines the instruction, formatted input, and expected answer for each example.
    """
    return [
        construct_query(
            prompt_template=prompt_template,
            val_prompt=prompt,
            val_code=generated_code
        )
        for (prompt, generated_code) in zip(
            examples["prompt"], examples["generated_code"]
        )
    ]

def construct_query(
    prompt_template: str,
    val_prompt: str,
    val_code: str,
) -> str:
    instruction = val_prompt.split("###Passage")[0]
    user_prompt = val_prompt.removeprefix(instruction)
    return prompt_template.format(instruction, user_prompt, val_code) + default_config.code_end_marker

def construct_paths(
    DATASET_NAME: str,
    SEED: int,
    MODEL_ID: str,
    PEFT: bool,
    LORA_MODULES: List[str],
    LOAD_IN_4BIT: bool,
    BATCH_SIZE: int,
    EVAL_BATCH_SIZE: int,
    GRAD_ACCUM: int,
    NO_TRAIN: bool,
) -> Tuple[Path, Path]:
    data_dir = os.path.join(
        "data",
        "clean_with_code",
        DATASET_NAME
    )

    model_id = MODEL_ID.replace("/", "_")
    model_id += "-4bit" if LOAD_IN_4BIT else ""
    if not NO_TRAIN:
        model_id += f"-peft{'_'.join(LORA_MODULES)}" if PEFT else ""
        model_id += f"-bs{BATCH_SIZE}"
        model_id += (
            f"-ga{GRAD_ACCUM}" if GRAD_ACCUM != 1 else ""
        )
    else:
        model_id += "-no-train"
    model_parent_dir = os.path.join(data_dir, "models", model_id)
    model_dir = os.path.join(model_parent_dir, "model")   

    os.makedirs(model_parent_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    return data_dir, model_dir