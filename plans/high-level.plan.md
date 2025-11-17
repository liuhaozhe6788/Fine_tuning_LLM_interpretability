Here is a specific, step-by-step plan to replicate the results of the research paper "Fine-tuning Smaller Language Models for Question Answering over Financial Documents" (EMNLP 2024):

### Step-by-Step Replication Plan

#### 1. **Select Datasets**

- Download the following financial QA datasets:
    - FinQA (Chen et al., 2021b)
    - ConvFinQA (Chen et al., 2022)
    - TATQA (Zhu et al., 2021)
- Use the predefined train/dev/test splits as described in Table 1 of the paper.[^1]


#### 2. **Prepare the Teacher Model**

- Use Gemini 2.5 pro as the teacher model for code generation.[^1]
- For each dataset, prepare a 4-shot prompt using exemplars from the paper (see Appendix E for FinQA, ConvFinQA, and TATQA prompts).[^1]


#### 3. **Generate Training Data**

- For each question in the training set, use the teacher model to generate Python code that:
    - Identifies relevant entities.
    - Writes the required formula.
    - Performs the calculation and stores the answer in a variable `ans`.[^1]
- Use the "Program of Thought" (PoT) prompting strategy to ensure consistent code structure.[^1]


#### 4. **Curate and Filter Data**

- Execute the generated code and compare the output with the ground truth answer.
- Remove any samples where the teacher-generated code produces an incorrect answer.[^1]
- Format the remaining samples as prompt-completion pairs for the student model, following the required tokenizer format (e.g., for Mistral, use `INST` tokens).[^1]


#### 5. **Select Student Models**

- Choose open-source models for fine-tuning:
    - PHI-3-MINI (3.8B), PHI-3-MEDIUM (14B)
    - Mistral 7B
    - ORCA-2-7B, ORCA-2-13B[^1]
- Download the models from Hugging Face or the official repositories.[^1]


#### 6. **Fine-tune Student Models**

- Use LoRA (Low-Rank Adaptation) for parameter-efficient fine-tuning via Hugging Face's PEFT library.[^1]
- Optimize hyperparameters (learning rate, LoRA rank, batch size, etc.) as described in Table 6 of the paper.[^1]
- Train on the curated prompt-completion pairs for 6 epochs, using a batch size of 1 and gradient accumulation as needed.[^1]


#### 7. **Evaluate Fine-tuned Models**

- For each test set, prompt the fine-tuned model to generate Python code for the question.
- Execute the generated code and compare the result with the ground truth answer.[^1]
- Use the same evaluation metrics as the paper (accuracy, concept understanding, entity extraction, executable code).[^1]


#### 8. **Analyze Model Capabilities**

- Use GPT-4 to assess concept understanding and entity extraction by comparing the student model's code with the teacher's code, using the prompts in Appendix C.[^1]
- Manually inspect a subset of samples to validate improvements in code structure and reasoning.[^1]


#### 9. **Experiment with Data Size**

- Repeat the fine-tuning process with smaller datasets (e.g., 1500 samples from FinQA, or a mix of FinQA and ConvFinQA samples) to reproduce the data efficiency experiments in Tables 4 and 5.[^1]


#### 10. **Report Results**

- Compare the accuracy of your fine-tuned models with the baselines (zero-shot, few-shot, and GPT-4) as shown in Table 2.[^1]
- Document the improvements in concept understanding, entity extraction, and code generation for each model.[^1]

***

### Key Implementation Details

- **Prompt Format:** Ensure all prompts and completions follow the structure specified in Figure 5 and Appendix B.[^1]
- **Evaluation:** Use the vLLM framework for inference and the same code execution pipeline as described in the paper.[^1]
- **Hardware:** The experiments were run on a machine with 24 cores, 220GB RAM, and an A100 GPU (80GB).[^1]

***

<div align="center">⁂</div>

[^1]: https://aclanthology.org/2024.findings-emnlp.617.pdf

