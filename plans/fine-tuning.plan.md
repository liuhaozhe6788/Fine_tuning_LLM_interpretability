### Fine-Tuning Plan

#### 1. **Prepare Datasets**

- **Datasets:** FinQA, ConvFinQA
- **Cleaning:** Ensure each dataset contains only question-context-code-answer pairs where the generated code produces the correct answer.

- **Split:** Use predefined train/dev/test splits for each dataset.[^1]


#### 2. **Select Student Models**

- **Models:** PHI-3-MINI (3.8B), Mistral 7B, ORCA-2-7B
- **Download:** Obtain the models from Hugging Face or official repositories.[^1]


#### 3. **Tokenization and Formatting**

- **Tokenizer:** Use the tokenizer corresponding to each student model.
- **Prompt Format:** Format each sample as a prompt-completion pair for the tokenizer. For  Mistral:

```
<s>[INST] question text context text [/INST] generated Python code </s>
```
Remember to add <EOS> after each example!

- **Dataset Preparation:** Convert each dataset into a list of prompt-completion pairs.[^1]


#### 4. **Fine-Tuning with SFTTrainer**

- **Library:** Use the SFTTrainer from the Hugging Face TRL library.
- **Configuration:**
    - **Model:** Load the student model.
    - **Tokenizer:** Load the corresponding tokenizer.
    - **Dataset:** Load the prepared dataset.
    - **Training Arguments:** Set learning rate, batch size, number of epochs, etc., as described in the paper.[^1]
- **Training Script Example:**

```python
from trl import SFTTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

model_name = "model_name"
dataset_name = "dataset_name"

model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

dataset = load_dataset("json", data_files=f"{dataset_name}.json")

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    dataset_text_field="text",
    max_seq_length=512,
    args={
        "output_dir": "output_dir",
        "num_train_epochs": 6,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 8,
        "learning_rate": 2e-5,
        "logging_steps": 10,
        "save_steps": 500,
        "evaluation_strategy": "steps",
        "eval_steps": 500,
        "save_total_limit": 2,
    },
)

trainer.train()
```
#### 5. **Save the peft model to Hugging Face Hub**
- After training the model with the SFTTrainer, save the model to Hugging Face Hub. 
- To save the model, we need to configure the HF_API_KEY.

#### 6. **Hyperparameter Tuning:**
 Optimize hyperparameters (num_of_epoch, ) using the dev set. 

