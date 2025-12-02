from nnsight import LanguageModel
import torch
import tqdm
import pandas as pd
import matplotlib.pyplot as plt
from datasets import Dataset, load_dataset
from huggingface_hub import login
import os
import einops
import numpy as np
# from train import cfg

HF_TOKEN = os.environ.get("HF_TOKEN")
login(HF_TOKEN)

DATASET_ID_NAME = "liuhaozhe6788/acts-finqa-lora"

@torch.no_grad()
def collect_acts(model, prompts):
    acts = []
    for i in tqdm.tqdm(range(len(prompts)), desc="Collecting acts"):
        prompt = prompts[i]
        with model.trace(prompt) as tracer:
            act = model.model.layers[16].mlp.output.save()
        act = act.squeeze(0)

        acts.append(act.bfloat16().detach().cpu())
    acts = torch.vstack(acts)
    return acts

def prepare_text_data():
    data = pd.read_csv("../data/clean_with_code/FinQA/finqa_train_generated_filtered.csv")
    full_text_data = data.apply(lambda x: x["prompt"] + x["generated_code"], axis=1)

    # seq_lens = full_text_data.apply(lambda x: len(x))
    # plt.hist(seq_lens, bins=100)
    # plt.savefig("seq_lens_histogram.png")
    # set the max seq len to last 10000 characters
    full_text_data = full_text_data.apply(lambda x: x[-10000:] if len(x) > 10000 else x)

    all_text_data = full_text_data.tolist()
    return all_text_data

def collect_base_model_acts_and_save_to_hf(all_text_data):
    base_model = LanguageModel('mistralai/Mistral-7B-Instruct-v0.3', device_map='cuda:0')
    base_model_acts = collect_acts(base_model, all_text_data)
    dataset = Dataset.from_dict({"base_model_acts": base_model_acts})
    dataset.push_to_hub(DATASET_ID_NAME, private=False)
    print("Saved base model acts to Hugging Face dataset")

def collect_ft_model_acts_and_save_to_hf(all_text_data):
    ft_model = LanguageModel('liuhaozhe6788/mistralai_Mistral-7B-Instruct-v0.3-FinQA-lora', device_map='cuda:0')
    ft_model_acts = collect_acts(ft_model, all_text_data)
    dataset = Dataset.from_dict({"ft_model_acts": ft_model_acts})
    dataset.push_to_hub(DATASET_ID_NAME, private=False)
    print("Saved ft model acts to Hugging Face dataset")

def estimate_norm_scaling_factor(acts: Dataset, n_acts_for_norm_estimate: int = 1000000):
    n_acts_for_norm_estimate = min(len(acts), n_acts_for_norm_estimate)
    norms_per_batch = []
    mean_norm = acts[:n_acts_for_norm_estimate].norm(dim=-1).mean().item()
    scaling_factor = np.sqrt(4096) / mean_norm
    return scaling_factor

def main():
    all_text_data = prepare_text_data()
    # collect_base_model_acts_and_save_to_hf(all_text_data)
    collect_ft_model_acts_and_save_to_hf(all_text_data)
    dataset = load_dataset(DATASET_ID_NAME, split="train")

if __name__ == "__main__":
    main()
