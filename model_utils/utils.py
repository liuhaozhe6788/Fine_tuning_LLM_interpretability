import code
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
                dtype=dtype,
                attn_implementation=attn_implementation,
            )
            tokenizer = prepare_tokenizer(model)
        except ValueError:
            print("Failed to load model with AutoPeftModelForCausalLM, now attempting with AutoModelForCausalLM.")
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                quantization_config=bnb_config,
                device_map=device,
                dtype=dtype,
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
            dtype=dtype,
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

def format_ft_prompts(
    examples: Union[Dataset, dict],
    eos_token: str,
    prompt_template: str,
) -> List[str]:
    """
    Construct a prompt for each example in examples using the prompt_template.

    Args:
        examples - a dataset containing columns ["context", "query", "weight_context", "answer"],
        eos_token - the end of sentence token
        prompt_template - the prompt template for which to fill out with examples

    Returns:
        a list of prompts that combines the instruction, formatted input, and expected answer for each example.
    """
    return [
        construct_ft_query(
            prompt_template=prompt_template,
            val_prompt=prompt,
            val_code=generated_code,
            eos_token=eos_token
        )
        for (prompt, generated_code) in zip(
            examples["prompt"], examples["generated_code"]
        )
    ]

def construct_ft_query(
    prompt_template: str,
    val_prompt: str,
    val_code: str,
    eos_token: str,
) -> str:
    instruction = val_prompt.split("###Passage")[0]
    user_prompt = val_prompt.removeprefix(instruction)
    return prompt_template.format(instruction, user_prompt, val_code) + default_config.code_end_marker + eos_token

def construct_ft_eval_query(
    prompt_template: str,
    val_prompt: str
) -> str:
    instruction = val_prompt.split("###Passage")[0]
    user_prompt = val_prompt.removeprefix(instruction)
    return prompt_template.format(instruction, user_prompt, "")

def construct_paths_and_model_id(
    DATASET_NAME: str,  
    SEED: int,
    MODEL_ID: str,
    PEFT: bool,
    LORA_MODULES: List[str],
    LOAD_IN_4BIT: bool,
    BATCH_SIZE: int,
    NUM_EPOCHS: int,
    GRAD_ACCUM: int,
    NO_TRAIN: bool,
) -> Tuple[Path, Path, str]:
    data_dir = os.path.join(
        "data",
        "clean_with_code",
        DATASET_NAME
    )

    model_id = MODEL_ID.replace("/", "_")
    model_id += "-4bit" if LOAD_IN_4BIT else ""
    model_id += f"-peft{'_'.join(LORA_MODULES)}" if PEFT else ""
    model_id += f"-bs{BATCH_SIZE}"
    model_id += f"-ne{NUM_EPOCHS}"
    model_id += (
        f"-ga{GRAD_ACCUM}" if GRAD_ACCUM != 1 else ""
    )

    model_dir = os.path.join("data", "models", DATASET_NAME, model_id)
    os.makedirs(model_dir, exist_ok=True)

    return data_dir, model_dir, model_id

def extract_code_from_response(result: str) -> str:
    code = result.split("[/INST]")[1].strip()
    # Post-process: cut off everything after '###End Python' if it exists
    end_marker = "###End Python"
    if end_marker in code:
        code = code.split(end_marker)[0] + "\n" + end_marker
    return code

def evaluate_model(
    model,
    tokenizer,
    dataset: Dataset,
    batch_size: int,
    is_response_correct_func,
    prompt_template: str
):
    """
    Given a dataset with columns ["prompt", "expected_answer"], generate Python code and get the actual answer, then evaluate model accuracy against the expected answer.
    1. Generate Python code from prompt
    2. Execute the Python code and get the actual answer
    3. Compare the actual answer with the expected answer using the is_response_correct_func
    4. Return the accuracy
    """
    # Free gpu memory
    gc.collect()
    torch.cuda.empty_cache()

    device = model.device
    tokenizer.padding_side = "left"
    encoded_dataset = dataset.map(
        lambda examples: tokenizer([construct_ft_eval_query(prompt_template, prompt) for prompt in examples["prompt"]], padding=True, return_tensors="pt"),
        batched=True,
        batch_size=batch_size,
    ).select_columns(["input_ids", "attention_mask", "expected_answer"])
    print(encoded_dataset[0])
    encoded_dataset.set_format(
        type="torch", columns=["input_ids", "attention_mask", "expected_answer"], device=device
    )  # required for loading correctly into dataloader
    dataloader = torch.utils.data.DataLoader(encoded_dataset, batch_size=batch_size)
    predictions, labels, is_correct_all = [], [], []
    num_correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader)):
            init_seq_len = batch["input_ids"].shape[1]
            outputs = model.generate(
                **batch,
                do_sample=False,
                temperature=0.0,
                top_p=0.9,
                max_new_tokens=1000,
            )
            responses_only = outputs[:, init_seq_len:]
            decoded_responses = tokenizer.batch_decode(responses_only)
            decoded_responses = [extract_code_from_response(response) for response in decoded_responses]
            is_correct = [
                is_response_correct_func(response, label) for response, label in zip(decoded_responses, batch["expected_answer"])
            ]

            num_correct += sum(is_correct)
            total += len(batch["expected_answer"])
            predictions += decoded_responses
            is_correct_all += is_correct
            labels += batch["expected_answer"]

            print(f"Average accuracy at batch {i}: {num_correct/total} ({num_correct}/{total}).")

    dataset = dataset.map(
        lambda examples: {
            "predictions": predictions,
            "is_correct": is_correct_all,
        },
        batched=True,  # need to set this so that it sets the predictions column to be one element per row from the list
        batch_size=len(
            dataset
        ),  # need to set this so that it doesn't have shape mismatch errors in the length of the column.
    )

    return dataset   

def compute_metrics(df: pd.DataFrame) -> Dict[str, float]:
    """
    Compute metrics for the evaluation results.
    """
    return {
        "accuracy": df["is_correct"].mean(),
    }