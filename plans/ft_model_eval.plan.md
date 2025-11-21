### fine-tuned student model evaluation

####  1. **Prepare base student model and fine-tuned student model**

- **Base model**: mistralai/Mistral-7B-Instruct-v0.3
- **Fine-tuned model**: mistralai/Mistral-7B-Instruct-v0.3-peftq_proj_k_proj_v_proj_o_proj-bs1 on the train set of FinQA dataset


####  2. **Do model inference in different prompting strategies using VLLM with the test set**


- **Zero-shot Prompting**:  
  - Use the base model (`mistralai/Mistral-7B-Instruct-v0.3`) to generate answers for each problem in the FinQA test set without providing any example solutions in the prompt.
  - Record outputs for evaluation.

- **Few-shot Prompting**:  
  - Construct prompts for the base model that include several (e.g., 3-5) solved example QA/code pairs from the training data followed by the test question.
  - Use consistent formatting for each example and the test question.
  - Generate and record model outputs for each test case.

- **Fine-tuned Model Inference**:
  - Use the fine-tuned model (`liuhaozhe6788/Mistral-7B-Instruct-v0.3-peftq_proj_k_proj_v_proj_o_proj-bs1`) and run inference on each test set instance.
  - Prompts may be the same as in zero-shot, unless the finetuning protocol used different prompt structures.

####  3. **Compare the answer from the generated code with the expected answer and calculate accuracy**
- For each of the strategies above:
  - Compare the generated code and final answer variable with ground truth.
  - Record and compute accuracy (and/or other relevant metrics).