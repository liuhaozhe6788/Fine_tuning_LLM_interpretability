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

### Fine-tuning with nohup

```bash
nohup python main.py FinQA mistralai/Mistral-7B-Instruct-v0.3 liuhaozhe6788 -P> training.log 2>&1 &
```

### Fine-tuned model evaluation with nohup
#### dev set
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

#### test set
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

### Mistral 7b instruct model and fine-tuned variant inference
```bash
python inference.py
```