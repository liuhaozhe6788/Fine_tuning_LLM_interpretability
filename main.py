import argparse
import gc
import json
import os
import random

import numpy as np
import torch

from transformers import TrainingArguments
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from peft import LoraConfig

from huggingface_hub import login

from config.prompt_templates import MODEL_ID_TO_TEMPLATES_DICT
from preprocessing.dataset import BaseDataset

from model_utils.utils import load_model_and_tokenizer, learning_rate_map_dict, format_prompts, construct_paths_and_model_id


login(token=os.getenv("HF_TOKEN"))

def get_args():
    parser = argparse.ArgumentParser(description="Arguments for training a model with Financial QA datasets.")
    parser.add_argument("DATASET_NAME", type=str, default="FinQA", help="Name of the dataset class")
    parser.add_argument("MODEL_ID", type=str, default="mistralai/Mistral-7B-Instruct-v0.3", help="Name of the model to use from huggingface")
    parser.add_argument("-S", "--SEED", type=int, default=3, help="Random seed")
    parser.add_argument(
    "-M",
    "--MODEL_ID",
    type=str,
    help="Name of the model to use from huggingface",
    )   
    parser.add_argument("-P", "--PEFT", action="store_true", help="Whether to train with PEFT")
    parser.add_argument(
        "-LM",
        "--LORA_MODULES",
        type=json.loads,
        default=["q_proj", "k_proj", "v_proj", "o_proj"],
        help="Which modules to train with LoRA",
    )
    parser.add_argument("-BS", "--BATCH_SIZE", type=int, default=1, help="Batch size for training (per device)")
    parser.add_argument("-NE", "--NUM_EPOCHS", type=int, default=1, help="Number of epochs to train for")
    parser.add_argument("-EBS", "--EVAL_BATCH_SIZE", type=int, default=8, help="Batch size for evaluation (per device)")
    parser.add_argument("-F", "--LOAD_IN_4BIT", action="store_true", help="Whether to load in 4 bit")
    parser.add_argument("-GA", "--GRAD_ACCUM", type=int, default=1, help="Number of steps for gradient accumulation")
    parser.add_argument("-MSL", "--MAX_SEQ_LENGTH", type=int, default=4096, help="Maximum sequence length for training")
    parser.add_argument(
        "-NT",
        "--NO-TRAIN",
        action="store_true",
        help="Whether to train the model",
    )
    parser.add_argument(
        "-O",
        "--OVERWRITE",
        action="store_true",
        help="Whether to overwrite existing results and retrain model",
    )
    parser.add_argument(
        "--HF_NAME",
        type=str,
        help="Name of the user on Hugging Face Hub",
    )
    return parser.parse_args()

def main():
    args = get_args()
    DATASET_NAME = args.DATASET_NAME
    SEED = args.SEED
    MODEL_ID = args.MODEL_ID
    PEFT = args.PEFT
    LORA_MODULES = args.LORA_MODULES
    LOAD_IN_4BIT = args.LOAD_IN_4BIT
    BATCH_SIZE = args.BATCH_SIZE
    EVAL_BATCH_SIZE = args.EVAL_BATCH_SIZE
    GRAD_ACCUM = args.GRAD_ACCUM
    MAX_SEQ_LENGTH = args.MAX_SEQ_LENGTH
    OVERWRITE = args.OVERWRITE
    NO_TRAIN = args.NO_TRAIN
    NUM_EPOCHS = args.NUM_EPOCHS
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    train_mode = not NO_TRAIN

    data_dir, model_dir, model_id = construct_paths_and_model_id(
        DATASET_NAME=DATASET_NAME, 
        SEED=SEED, 
        MODEL_ID=MODEL_ID,
        PEFT=PEFT,
        LORA_MODULES=LORA_MODULES,
        LOAD_IN_4BIT=LOAD_IN_4BIT,
        BATCH_SIZE=BATCH_SIZE,
        NUM_EPOCHS=NUM_EPOCHS,
        GRAD_ACCUM=GRAD_ACCUM,
        NO_TRAIN=NO_TRAIN
    )

    repo_id = f"{args.HF_NAME}/{model_id}" # repo_id is the full path to the model on Hugging Face Hub

    dataset = BaseDataset(
        train_path= os.path.join(data_dir, "finqa_train_generated_filtered.csv"),
        val_path= os.path.join(data_dir, "finqa_dev_generated_filtered.csv"),
        test_path= os.path.join(data_dir, "finqa_test_generated_filtered.csv"),
        seed=SEED
    )

    prompt_template, response_template = MODEL_ID_TO_TEMPLATES_DICT[MODEL_ID]
    peft_config = (LoraConfig(
        r=512,
        lora_alpha=1024,
        target_modules=LORA_MODULES,
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM"
    ) if PEFT else None)

    # Load the model
    if not OVERWRITE:
        if (
        os.path.isfile(os.path.join(model_dir, "config.json"))
        or os.path.isfile(os.path.join(model_dir, "adapter_config.json"))
        ):
            # Model has already been trained
            print(f"Model already saved at {model_dir}, attempting to load.")
            model, tokenizer = load_model_and_tokenizer(
                model_id=model_dir,
                load_in_4bit=LOAD_IN_4BIT,
                peft_config=peft_config,
                train_mode=train_mode,
                attn_implementation="sdpa",
            )
            print(f"Loaded fine-tuned model from {model_dir}")
        else:
            print(f"Model not found at {model_dir}, attempting to load from Hugging Face Hub.")
            try: 
                model, tokenizer = load_model_and_tokenizer(
                model_id=repo_id,
                load_in_4bit=LOAD_IN_4BIT,
                peft_config=peft_config,
                train_mode=train_mode,
                attn_implementation="sdpa",
                )
                print(f"Loaded fine-tuned model from {repo_id}")
            except Exception as e:
                print(f"Error loading fine-tuned model from Hugging Face Hub: {e}")
    else:
        print(f"Loading model {MODEL_ID} from huggingface.")
        # Cannot load model with PeftConfig if in training mode
        model, tokenizer = load_model_and_tokenizer(
            model_id=MODEL_ID,
            load_in_4bit=LOAD_IN_4BIT,
            peft_config=peft_config,
            train_mode=train_mode,          
            attn_implementation="sdpa"
        )
        # SFT Train
        response_template_ids = tokenizer.encode(response_template, add_special_tokens=False)
        model.gradient_checkpointing_enable()  # PEFT + checkpointing
        model.enable_input_require_grads()     # ensure input grads
        model.config.use_cache = False         # disable cache
        collator = DataCollatorForCompletionOnlyLM(response_template_ids, tokenizer=tokenizer)
        trainer = SFTTrainer(
            model=model,
            data_collator=collator,
            formatting_func=lambda x: format_prompts(
                x,
                eos_token=tokenizer.eos_token,
                prompt_template=prompt_template,
            ),
            train_dataset=dataset.train_data,
            max_seq_length=MAX_SEQ_LENGTH,
            dataset_num_proc=2,
            packing=False,  # Can make training 5x faster for short sequences.
            processing_class=tokenizer,
            args=TrainingArguments(
                output_dir=model_dir,
                gradient_checkpointing=True,
                per_device_train_batch_size=BATCH_SIZE,
                gradient_accumulation_steps=GRAD_ACCUM,
                warmup_steps=5,
                num_train_epochs=1,
                save_strategy="no",
                learning_rate=learning_rate_map_dict[MODEL_ID],
                fp16=not torch.cuda.is_bf16_supported(),
                bf16=torch.cuda.is_bf16_supported(),
                logging_steps=1,
                optim="adamw_8bit",
                weight_decay=0.01,
                lr_scheduler_type="linear",
                seed=SEED,
            ),
        )

        gc.collect()
        for i in range(torch.cuda.device_count()):
            torch.cuda.set_device(i)
            torch.cuda.empty_cache()

        print("Preparing to train model.")
        trainer_stats = trainer.train()
        print("Trainer stats:", trainer_stats)

        trainer.push_to_hub(repo_id)
        print(f"Model pushed to Hugging Face Hub")    




if __name__ == "__main__":
    main()