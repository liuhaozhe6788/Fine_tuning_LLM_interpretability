from utils import *
from crosscoder import CrossCoder
import matplotlib.pyplot as plt
import seaborn as sns
torch.set_grad_enabled(False);

device = 'cuda:0'

cross_coder = CrossCoder.load_from_hf(device=device)

model_name = "Mistral-7B-Instruct-v0.3"
norms = cross_coder.W_dec.norm(dim=-1)

relative_norms = norms[:, 1] / norms.sum(dim=-1)

plt.figure(figsize=(10, 6))
sns.histplot(
    relative_norms.detach().cpu().numpy(),
    bins=200,
    kde=False
)
plt.title(f"{model_name} Base vs FT Model Diff")
plt.xlabel("Relative decoder norm strength")
plt.ylabel("Number of Latents")
plt.xticks([0, 0.25, 0.5, 0.75, 1.0], ['0', '0.25', '0.5', '0.75', '1.0'])

results_dir = Path(f"results/{model_name}")
results_dir.mkdir(parents=True, exist_ok=True)
plt.savefig(results_dir / "relative_norms.png", dpi=300, bbox_inches='tight')

##TODO caculate the number of acts with relative norm smaller than 0.25 and larger than 0.75
num_acts_small_norm = (relative_norms < 0.25).sum()
num_acts_large_norm = (relative_norms > 0.75).sum()
print(f"Number of acts with relative norm smaller than 0.25: {num_acts_small_norm}")
print(f"Number of acts with relative norm larger than 0.75: {num_acts_large_norm}")

shared_latent_mask = (relative_norms < 0.7) & (relative_norms > 0.3)

# Cosine similarity of crosscoder vectors between models

cosine_sims = (cross_coder.W_dec[:, 0, :] * cross_coder.W_dec[:, 1, :]).sum(dim=-1) / (cross_coder.W_dec[:, 0, :].norm(dim=-1) * cross_coder.W_dec[:, 1, :].norm(dim=-1))

plt.figure(figsize=(10, 6))
sns.histplot(
    cosine_sims[shared_latent_mask].to(torch.float32).detach().cpu().numpy(),
    bins=100,
    kde=False
)
plt.xlabel("Cosine similarity of decoder vectors between models")
plt.ylabel("Number of Latents (log scale)")
plt.yscale('log')
plt.xlim(-1, 1)

plt.savefig(results_dir / "cosine_sims.png", dpi=300, bbox_inches='tight')