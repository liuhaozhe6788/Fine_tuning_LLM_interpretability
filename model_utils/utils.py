import code
import os
import gc
import sys
import re

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

from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from huggingface_hub import snapshot_download

from config.generation_config import default_config
from config.prompt_templates import four_shot_prompt_finqa_templates, zero_shot_eval_instruction, few_shot_eval_instruction
from preprocessing.utils import execute_code

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
    padding_side: str = "right"
):
    """
    Load the model and tokenizer from huggingface.
    Args:
        model_id: str
        base_model_id: str
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
        use_vllm_inference: bool - default to False. Whether to use vllm for inference.
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

def load_vllm_model(model_name: str, adapter_name: str=None, enable_lora: bool = True, max_lora_rank: int = 512):
    if enable_lora:
        llm = LLM(model=model_name, enable_lora=enable_lora, max_lora_rank=max_lora_rank)
        llm.set_tokenizer(prepare_vllm_tokenizer(llm.get_tokenizer(), padding_side="left"))
        lora_path = snapshot_download(repo_id=adapter_name)
        return llm, lora_path
    else:
        llm = LLM(model=model_name)
        llm.set_tokenizer(prepare_vllm_tokenizer(llm.get_tokenizer(), padding_side="left"))
        return llm, None

def prepare_tokenizer(model, set_pad_token=True, padding_side="right"):
    tokenizer = AutoTokenizer.from_pretrained(model.config._name_or_path)

    tokenizer.padding_side = padding_side 
    if set_pad_token and tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

def prepare_vllm_tokenizer(tokenizer, set_pad_token=True, padding_side="left"):
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

def construct_zero_shot_eval_query(
    prompt_template: str,
    val_prompt: str
) -> str:
    instruction = val_prompt.split("###Passage")[0]
    user_prompt = val_prompt.removeprefix(instruction)
    user_prompt = user_prompt.split(default_config.code_start_marker)[0]
    user_prompt = user_prompt + "###Instructions: " + zero_shot_eval_instruction + default_config.code_start_marker
    query = prompt_template.format(instruction, user_prompt, "")
    return query

def construct_few_shot_eval_query(
    few_shot_examples: list,
    prompt_template: str,
    val_prompt: str
) -> str:
    instruction = val_prompt.split("###Passage")[0]
    user_prompt = val_prompt.removeprefix(instruction)
    user_prompt = user_prompt.split(default_config.code_start_marker)[0]
    user_prompt = user_prompt + "###Instructions: " + few_shot_eval_instruction + default_config.code_start_marker
    query = prompt_template.format(instruction, user_prompt, "")
    if len(few_shot_examples) > 0:
        # Combine few-shot examples
        few_shot_text = "\n\n".join(few_shot_examples)
        full_query = few_shot_text + "\n\n" + query
        return full_query
    else:
        return query

def construct_paths_and_model_id(
    DATASET_NAME: str,  
    MODEL_ID: str,
    PEFT: bool,
    LORA_MODULES: List[str],
    LOAD_IN_4BIT: bool,
    BATCH_SIZE: int,
    NUM_EPOCHS: int,
    GRAD_ACCUM: int,
    NO_TRAIN: bool,
) -> Tuple[Path, Path, Path, str]:
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

    results_dir = os.path.join("results", DATASET_NAME, model_id)
    os.makedirs(results_dir, exist_ok=True)

    return data_dir, model_dir, results_dir, model_id

def extract_code_from_zero_shot_response(response: str) -> str:
    """
    Extract Python code from markdown code blocks in the response.
    
    Args:
        response: String that may contain markdown code blocks with ```python ... ```
        
    Returns:
        Extracted Python code, or the original response if no code block is found.
    """
    # Pattern to match ```python followed by code and closing ```
    # re.DOTALL makes . match newlines, re.MULTILINE makes ^/$ work per line
    pattern = r'```python\n\s*(.*?)```'
    match = re.search(pattern, response, re.DOTALL)
    
    if match:
        return match.group(1).strip()
    else:
        # If no code block found, return the original response
        return response.strip()

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
    encoded_dataset = encoded_dataset.rename_column("expected_answer", "labels")
    # print(encoded_dataset[0])
    encoded_dataset.set_format(
        type="torch", columns=["input_ids", "attention_mask", "labels"], device=device
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
            actual_answers = [execute_code(response)[0] for response in decoded_responses]
            is_correct = [
                is_response_correct_func(actual_answer, label) for actual_answer, label in zip(actual_answers, batch["labels"])
            ]

            num_correct += sum(is_correct)
            total += len(batch["labels"])
            predictions += actual_answers        
            is_correct_all += is_correct
            labels += batch["labels"]

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

def compare_answers(actual: str | int | float | bool, expected: str) -> bool:
    """
    Compare actual and expected answers, handling numeric, string, and boolean comparisons.
    
    Args:
        actual: Actual answer from code execution
        expected: Expected answer from dataset
        
    Returns:
        True if answers match, False otherwise
    """
    if not (isinstance(actual, str) or isinstance(actual, int) or isinstance(actual, float) or isinstance(actual, bool)):
        return False
    if pd.isna(actual) or actual is None:
        return False
    
    if pd.isna(expected) or expected is None:
        return False
    
    # Convert to strings and strip whitespace
    actual_str = str(actual).strip()
    expected_str = str(expected).strip()
    
    # Try numeric comparison first
    try:
        # Try numeric comparison
        expected_num = float(expected_str)
        result_num = float(actual_str)
        result_num_div_100 = result_num / 100
        result_num_div_1k = result_num / 1000
        result_num_div_1m = result_num / 1000000
        result_num_div_1b = result_num / 1000000000
        is_valid = abs(expected_num - result_num) < 1e-3
        is_valid |= abs(expected_num - result_num_div_100) < 1e-3
        is_valid |= abs(expected_num - result_num_div_1k) < 1e-3
        is_valid |= abs(expected_num - result_num_div_1m) < 1e-3
        is_valid |= abs(expected_num - result_num_div_1b) < 1e-3
        return is_valid
    except (ValueError, TypeError):
        # Fall back to string comparison
        is_valid = actual_str.lower() == expected_str.lower()
        if is_valid:
            return True
        elif isinstance(actual, bool):
            if actual and expected_str == "yes":
                return True
            elif not actual and expected_str == "no":
                return True
            else:
                return False
        else:
            return False

def evaluate_vllm_model(
    llm,
    lora_path: str=None,
    dataset: Dataset=None,
    batch_size: int=8,
    is_response_correct_func=None,
    prompt_template: str="",
    eval_type: str="fine-tuned",
    few_shot_examples: list=four_shot_prompt_finqa_templates
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

    if eval_type == "fine-tuned":
        mapping_func = construct_ft_eval_query
        mapped_dataset = dataset.map(
            lambda examples: {"prompt": [mapping_func(prompt_template, prompt) for prompt in examples["prompt"]]},
            batched=True,
            batch_size=batch_size,
        ).select_columns(["prompt", "expected_answer"])
    elif eval_type == "zero-shot":
        mapping_func = construct_zero_shot_eval_query
        mapped_dataset = dataset.map(
            lambda examples: {"prompt": [mapping_func(prompt_template, prompt) for prompt in examples["prompt"]]},
            batched=True,
            batch_size=batch_size,
        ).select_columns(["prompt", "expected_answer"])
    elif eval_type == "few-shot":
        mapping_func = construct_few_shot_eval_query
        mapped_dataset = dataset.map(
            lambda examples: {"prompt": [mapping_func(few_shot_examples, prompt_template, prompt) for prompt in examples["prompt"]]},
            batched=True,
            batch_size=batch_size,
        ).select_columns(["prompt", "expected_answer"])

    dataloader = torch.utils.data.DataLoader(mapped_dataset, batch_size=batch_size)
    predictions, labels, is_correct_all = [], [], []
    num_correct = 0
    total = 0

    lora_request=LoRARequest("FinQA_adapter", 1, lora_path) if lora_path is not None else None
    for i, batch in enumerate(tqdm(dataloader)):
        outputs = llm.generate(
            batch["prompt"], 
            sampling_params=SamplingParams(temperature=0.0, top_p=0.9, max_tokens=1000),
            lora_request=lora_request
        )
        if eval_type == "zero-shot":
            code_outputs = [extract_code_from_zero_shot_response(output.outputs[0].text) for output in outputs]
        elif eval_type == "fine-tuned" or eval_type == "few-shot":
            code_outputs = [output.outputs[0].text for output in outputs]
        actual_answers = [execute_code(code_output)[0] for code_output in code_outputs]
        is_correct = [
            is_response_correct_func(actual_answer, label) for actual_answer, label in zip(actual_answers, batch["expected_answer"])
        ]

        num_correct += sum(is_correct)
        total += len(batch["expected_answer"])
        predictions += [str(actual_answer) for actual_answer in actual_answers]        
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
