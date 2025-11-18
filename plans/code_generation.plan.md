Here is a specific plan for generating code from the teacher model:

### Step-by-Step Plan: Generate Code 

#### 1. **Prepare the Prompt Template for Teacher Model**

- Use the 4-shot prompt template provided in Appendix E of the paper for each dataset (FinQA, ConvFinQA, TATQA).[^1]
- Each prompt should include:
    - The context (text+table)
    - The question
    - The answer hint with program
    - For ConvFinQA, we also have the previous questions in conversation history
- Ensure the prompt format matches the structure in Figure 5 of Appendix B.[^1]


#### 2. **Generate Code with the Teacher Model**

- For each question in the training set, dev set, and test set, send the prompt (with the 4-shot examples) to GPT-5.[^1]
- Instruct GPT-5 to generate Python code that:
    - Identifies relevant entities from the financial text.
    - Writes the required formula.
    - Performs the calculation and stores the answer in a variable named `ans`.[^1]

- Run the generated code using a Python interpreter.
- Compare the output of the code with the ground truth answer. Beware of the different expressions in percentage, hundreds, thousands, and millions. 
- If the code produces the correct answer, keep the sample.
- If the code produces an incorrect answer, discard the sample or manually correct the code if possible.[^1]

#### 3. **Curate the Dataset for Fine-tuning**

- Collect all valid prompt-completion instances for which the code produces the correct answer.
- Instruct the student model with the formatted prompt that includes:
    - The context (text+table)
    - The question
    - The python code
    - For ConvFinQA, we also have the previous questions in conversation history
- Format each sample for different student models.
Ensure the formatting matches the tokenizer requirements for the student model (e.g., for Mistral, use `INST` tokens as described in Appendix B).[^1]


