# Collect activation for the mistral instruct model and the fine tuned model
## Load dataset
Sample 3000 examples from the FinQA training set, the input data is the combined prompt and generated code.
## Load model
Load the model with nnsight.
## Compute activation
Do forward pass and compute the activation of the layer 16 MLP output for each position in the full sequence. Rearrange the activations in the shape of (num_of_total_acts, act_dim). Normalize the activations by the scaling factor for the respective model.
## Store the activation as dataset
Store the activations as Hugging Face dataset
