# Collect activation for the mistral instruct model and the fine tuned model
## Load dataset
First use a toy dataset, then change to the FinQA training set, the input data is the combined prompt and generated code.
## Load model
Load the model with nnsight.
## Compute activation
Do forward pass and compute the activation of the layer 16 MLP output for each position in the generated code, and ignore the positions in the prompt.
## Store the activation as dataset
Store the activations as Hugging Face dataset
