from utils import *
from nnsight import LanguageModel
from trainer import Trainer

device_LM = 'cuda:0'
device_crosscoder = 'cuda:1'

base_model = LanguageModel('mistralai/Mistral-7B-Instruct-v0.3', device_map=device_LM)
chat_model = LanguageModel('liuhaozhe6788/mistralai_Mistral-7B-Instruct-v0.3-FinQA-lora', device_map=device_LM)


input_prompts = ["The Eiffel Tower is in the city of Paris."] * 1024


default_cfg = {
    "seed": 49,
    "batch_size": 4096,
    "buffer_mult": 128,
    "lr": 5e-5,
    "num_tokens": 400_000_000,
    "l1_coeff": 2,
    "beta1": 0.9,
    "beta2": 0.999,
    "d_in": base_model.config.hidden_size,
    "dict_size": 2**14,
    "seq_len": 1024,
    "enc_dtype": "fp32",
    "model_name": "Mistral-7B-Instruct-v0.3",
    "site": "resid_pre",
    "device_LM": device_LM,
    "device_crosscoder": device_crosscoder,
    "model_batch_size": 4,
    "log_every": 100,
    "save_every": 30000,
    "dec_init_norm": 0.08,
}
cfg = arg_parse_update_cfg(default_cfg)

trainer = Trainer(cfg, base_model, chat_model, input_prompts)
trainer.train()