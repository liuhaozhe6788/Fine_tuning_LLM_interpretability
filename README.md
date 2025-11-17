## Data Preprocessing Guide


### 1. Requirements & Setup
- Python 3.10+ on macOS/Linux
- OpenAI API access (the scripts call GPT-5/GPT-4o models)

```bash
pip install -r requirements.txt
```

---

### 2. Convert Raw JSON to Clean CSV
Each preprocessing script can be run as a module from the project root. They call their respective `preprocess_*()` functions with the default base path (`data/`). If your raw dumps live elsewhere, edit the function call at the bottom of the script or import the function and pass an explicit `base_path`.

#### FinQA
```bash
python -m preprocessing.preprocess_finqa
```
Outputs: `data/clean/FinQA/{train,dev,test}.csv` (columns: `pre_text`, `post_text`, `table`, `question`, `answer`, `program`).

---

### 3. Generate Teacher Code
The `process_*_code_generation.py` scripts read the cleaned CSVs, build few-shot prompts, call the OpenAI API (async by default), execute the returned code, and save `prompt`, `generated_code`, `actual_answer`, and `expected_answer`.

#### FinQA
```bash
python -m preprocessing.process_finqa_code_generation
```
Edit the `process_finqa_samples` calls at the bottom of the file to choose which split to process and where to save the CSV (default paths under `data/clean_with_code/FinQA/`).

---

### 4. Filter Rows With Incorrect Executions
After code generation, remove rows where the executed answer (`actual_answer`) differs from the dataset label (`expected_answer`):

```bash
python -m preprocessing.filter_valid_answers
```
---