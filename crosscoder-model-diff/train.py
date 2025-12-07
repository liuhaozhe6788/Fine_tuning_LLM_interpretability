from utils import *
from nnsight import LanguageModel
from trainer import Trainer
from datasets import load_dataset
device = 'cuda:0'

# Change this to your desired dataset name
base_model_acts = load_dataset("liuhaozhe6788/acts-finqa-base", split="train")
# Change this to your desired dataset name
ft_model_acts = load_dataset("liuhaozhe6788/acts-finqa-lora", split="train")  

default_cfg = {
    "seed": 49,
    "batch_size": 4096,
    "lr": 5e-5,
    "batch_topk": None, # None for full batch, int for topk
    "l1_coeff": 2,
    "beta1": 0.9,
    "beta2": 0.999,
    "d_in": 4096,
    "dict_size": 2**14,
    "enc_dtype": "fp32",
    "model_name": "Mistral-7B-Instruct-v0.3",
    "device": device,
    "log_every": 10,
    "save_every": 100,
    "dec_init_norm": 0.08,
    "wandb_project": "crosscoder",
    "wandb_entity": "liuhaozhe2000",
}

batch_topk_cfg = {
    "seed": 49,
    "batch_size": 4096,
    "lr": 5e-5,
    "batch_topk": 100, # None for full batch, int for topk
    "l1_coeff": 2,
    "beta1": 0.9,
    "beta2": 0.999,
    "d_in": 4096,
    "dict_size": 2**14,
    "enc_dtype": "fp32",
    "model_name": "Mistral-7B-Instruct-v0.3",
    "device": device,
    "log_every": 10,
    "save_every": 100,
    "dec_init_norm": 0.08,
    "wandb_project": "crosscoder",
    "wandb_entity": "liuhaozhe2000",
}

cfg = arg_parse_update_cfg(batch_topk_cfg)

trainer = Trainer(cfg, base_model_acts, ft_model_acts)

trainer.train()