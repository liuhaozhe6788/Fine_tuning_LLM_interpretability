from utils import *
from crosscoder import CrossCoder
from nnsight import LanguageModel
torch.set_grad_enabled(False);

device = 'cuda:0'
base_estimated_scaling_factor = 0.2835
chat_estimated_scaling_factor = 0.2533

base_model = LanguageModel('mistralai/Mistral-7B-Instruct-v0.3', device_map=device)
chat_model = LanguageModel('liuhaozhe6788/mistralai_Mistral-7B-Instruct-v0.3-FinQA-lora', device_map=device)

cross_coder = CrossCoder.load_from_hf(device=device)

model_name = "Mistral-7B-Instruct-v0.3"
norms = cross_coder.W_dec.norm(dim=-1)
norms.shape

relative_norms = norms[:, 1] / norms.sum(dim=-1)
relative_norms.shape

fig = px.histogram(
    relative_norms.detach().cpu().numpy(), 
    title=f"{model_name} Base vs FT Model Diff",
    labels={"value": "Relative decoder norm strength"},
    nbins=200,
)

fig.update_layout(showlegend=False)
fig.update_yaxes(title_text="Number of Latents")

# Update x-axis ticks
fig.update_xaxes(
    tickvals=[0, 0.25, 0.5, 0.75, 1.0],
    ticktext=['0', '0.25', '0.5', '0.75', '1.0']
)

fig.show()
results_dir = Path(f"results/{model_name}")
results_dir.mkdir(parents=True, exist_ok=True)
fig.write_image(results_dir / "relative_norms.png")


shared_latent_mask = (relative_norms < 0.7) & (relative_norms > 0.3)
shared_latent_mask.shape

# Cosine similarity of recoder vectors between models

cosine_sims = (cross_coder.W_dec[:, 0, :] * cross_coder.W_dec[:, 1, :]).sum(dim=-1) / (cross_coder.W_dec[:, 0, :].norm(dim=-1) * cross_coder.W_dec[:, 1, :].norm(dim=-1))
cosine_sims.shape

import plotly.express as px
import torch

fig = px.histogram(
    cosine_sims[shared_latent_mask].to(torch.float32).detach().cpu().numpy(), 
    #title="Cosine similarity of decoder vectors between models",
    log_y=True,  # Sets the y-axis to log scale
    range_x=[-1, 1],  # Sets the x-axis range from -1 to 1
    nbins=100,  # Adjust this value to change the number of bins
    labels={"value": "Cosine similarity of decoder vectors between models"}
)

fig.update_layout(showlegend=False)
fig.update_yaxes(title_text="Number of Latents (log scale)")

fig.show()
fig.write_image(results_dir / "cosine_sims.png")