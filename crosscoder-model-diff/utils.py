# %%
import os
from IPython import get_ipython

ipython = get_ipython()
# Code to automatically update the HookedTransformer code as its edited without restarting the kernel
if ipython is not None:
    ipython.magic("load_ext autoreload")
    ipython.magic("autoreload 2")

import plotly.io as pio
pio.renderers.default = "jupyterlab"

# Import stuff
import einops
import json
import argparse

from datasets import load_dataset
from pathlib import Path
import plotly.express as px
from torch.distributions.categorical import Categorical
from tqdm import tqdm
import torch


# crosscoder stuff

def arg_parse_update_cfg(default_cfg):
    """
    Helper function to take in a dictionary of arguments, convert these to command line arguments, look at what was passed in, and return an updated dictionary.

    If in Ipython, just returns with no changes
    """
    if get_ipython() is not None:
        # Is in IPython
        print("In IPython - skipped argparse")
        return default_cfg
    cfg = dict(default_cfg)
    parser = argparse.ArgumentParser()
    for key, value in default_cfg.items():
        if type(value) == bool:
            # argparse for Booleans is broken rip. Now you put in a flag to change the default --{flag} to set True, --{flag} to set False
            if value:
                parser.add_argument(f"--{key}", action="store_false")
            else:
                parser.add_argument(f"--{key}", action="store_true")

        else:
            parser.add_argument(f"--{key}", type=type(value), default=value)
    args = parser.parse_args()
    parsed_args = vars(args)
    cfg.update(parsed_args)
    print("Updated config")
    print(json.dumps(cfg, indent=2))
    return cfg    

def load_pile_lmsys_mixed_tokens():
    print("Loading data from HF")
    data = load_dataset(
        "ckkissane/pile-lmsys-mix-1m-tokenized-gemma-2", 
        split="train", 
        cache_dir="/workspace/cache/"
    )
    data.save_to_disk("/workspace/data/pile-lmsys-mix-1m-tokenized-gemma-2.hf")
    data.set_format(type="torch", columns=["input_ids"])
    all_tokens = data["input_ids"]
    torch.save(all_tokens, "/workspace/data/pile-lmsys-mix-1m-tokenized-gemma-2.pt")
    print(f"Saved tokens to disk")
    return all_tokens
