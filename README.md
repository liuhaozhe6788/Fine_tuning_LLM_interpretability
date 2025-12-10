## Requirements & Setup
- Python 3.12 on macOS/Linux

```bash
pip install -r requirements.txt
```


## Data Preprocessing 
The final preprocessed dataset for FinQA is saved under data/clean_with_code/FinQA/* filtered.csv

To replicate the data preprocessing and code generation process, you need the OpenAI api access.


### 1. Convert Raw JSON to Clean CSV
The folder that contains raw json files is under the `data/` directory.

#### FinQA
```bash
python -m preprocessing.preprocess_finqa
```
Outputs: `data/clean/FinQA/{train,dev,test}.csv` (columns: `pre_text`, `post_text`, `table`, `question`, `answer`, `program`).

---

### 2. Generate Teacher Code
The `process_*_code_generation.py` scripts read the cleaned CSVs, build few-shot prompts, call the OpenAI API (async by default)e and save with columns: `prompt`, `generated_code`, `actual_answer`, and `expected_answer`.

#### FinQA
```bash
python -m preprocessing.process_finqa_code_generation
```
Edit the `process_finqa_samples` calls at the bottom of the file to choose which split to process and where to save the CSV (default paths under `data/clean_with_code/FinQA/`).

---

### 3. Filter Rows With Incorrect Executions
After code generation, remove rows where the executed answer (`actual_answer`) differs from the dataset label (`expected_answer`):

```bash
python -m preprocessing.filter_valid_answers
```
---

## Fine-tuning with nohup
Use the filtered clean FinQA train set to train the lora adaptor. 
```bash
nohup python main.py FinQA mistralai/Mistral-7B-Instruct-v0.3 liuhaozhe6788 -P> training.log 2>&1 &
```
The trained LoRA adaptor is stores in [Hugging Face](https://huggingface.co/liuhaozhe6788/mistralai_Mistral-7B-Instruct-v0.3-peftq_proj_k_proj_v_proj_o_proj-bs1-ne1).

---
## Fine-tuned model evaluation with nohup
Fine-tuned model accuracy evaluation.
### 1. Dev set
- fine-tuned
```bash
nohup python main.py FinQA mistralai/Mistral-7B-Instruct-v0.3 liuhaozhe6788 -P -NT -D --VLLM> eval_ft.log 2>&1 &
```

- zero-shot
```bash
nohup python main.py FinQA mistralai/Mistral-7B-Instruct-v0.3 liuhaozhe6788 -P -NT -D --VLLM --EVAL_TYPE zero-shot> eval_zero_shot.log 2>&1 &
```

- few-shot
```bash
nohup python main.py FinQA mistralai/Mistral-7B-Instruct-v0.3 liuhaozhe6788 -P -NT -D --VLLM --EVAL_TYPE few-shot> eval_few_shot.log 2>&1 &
```

### 2. Test set
- fine-tuned
```bash
nohup python main.py FinQA mistralai/Mistral-7B-Instruct-v0.3 liuhaozhe6788 -P -NT -T --VLLM> eval_ft.log 2>&1 &
```

- zero-shot
```bash
nohup python main.py FinQA mistralai/Mistral-7B-Instruct-v0.3 liuhaozhe6788 -P -NT -T --VLLM --EVAL_TYPE zero-shot> eval_zero_shot.log 2>&1 &
```

- few-shot
```bash
nohup python main.py FinQA mistralai/Mistral-7B-Instruct-v0.3 liuhaozhe6788 -P -NT -T --VLLM --EVAL_TYPE few-shot> eval_few_shot.log 2>&1 &
```
The evaluation results are stored under `results/FinQA`.
## Mistral 7b instruct model and fine-tuned variant inference
```bash
python inference.py
```

## Model diffing
### Crosscoder
Crosscoder is a sparse autoencoder architecture that is trained on the activations with the same hook name from the base LLM and the fine-tuned LLM to compare the difference between the models. By analysing the crosscoder, one can gain insight into how the fine-tuning process changes the features that the model learns. 
```bash
cd crosscoder-model-diff/
```
#### 1. Merge the peft model (optional)
Merge the LoRA with the instruct model and push the merged model weights to Hugging Face Hub for nnsight inference. 

This step is optional. The merged model weight is automatically downloaded from Hugging Face, when we run the following scripts.
```bash
python merge_and_push_lora.py
```
#### 2. Collect activations (optional)
Collect the activations at the output of MLP at layer 17 for both the instruct model and the fine-tuned model with 1024 examples sampled from the FinQA test set. 

This step is optional. The merged model weight is automatically downloaded from Hugging Face, when we run the following scripts.
```bash
python collect_activations.py
```

#### 3. Crosscoder training
We train a BatchTopK crosscoder from [Minder, Julian, et al.](https://arxiv.org/pdf/2504.02922) to alleviate training artifacts that falsely increase the relative norm between the fine-tuned model and the instruct model for a instruct model-specific feature. 

The training weights and configuration is automatically downloaded from Hugging Face, when we run the crosscoder analysis notebook. The training process is logged in [wandb](https://wandb.ai/liuhaozhe2000/crosscoder)
```bash
python train.py
```
#### 4. Crosscoder analysis
We visualise the histogram of relative decoder norm strength and the cosine similarity of decoder vector between models, as well as generate the latent dashboard to display the hot tokens for some interesting latents. The results are stored under `crosscoder-model-diff/results/Mistral-7B-Instruct-v0.3_1k_samples_batchtopk` after running the notebook `analysis_and_dashboard.ipynb`.

