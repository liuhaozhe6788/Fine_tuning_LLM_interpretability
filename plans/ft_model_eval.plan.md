### fine-tuned student model evaluation

####  1. **Prepare base student model and fine-tuned student model**

- **Base model**: mistralai/Mistral-7B-Instruct-v0.3
- **Fine-tuned model**: mistralai/Mistral-7B-Instruct-v0.3-peftq_proj_k_proj_v_proj_o_proj-bs1 on the train set of FinQA dataset


####  2. **Do model inference in different prompting strategies using VLLM with the test set**

- ****

####  3. **Compare the answer from the generated code with the expected answer and calculate accuracy**
