from utils import *
import tqdm
from nnsight import LanguageModel


class Buffer:
    """
    This defines a data buffer, to store a stack of acts across both model that can be used to train the autoencoder. It'll automatically run the model to generate more when it gets halfway empty.
    """

    def __init__(self, cfg, model_A: LanguageModel, model_B: LanguageModel, input_prompts):
        assert model_A.config.hidden_size == model_B.config.hidden_size
        self.cfg = cfg
        self.buffer_size = cfg["batch_size"] * cfg["buffer_mult"]
        self.buffer_batches = self.buffer_size // (cfg["seq_len"] - 1)
        self.buffer_size = self.buffer_batches * (cfg["seq_len"] - 1)
        self.buffer = torch.zeros(
            (self.buffer_size, 2, model_A.config.hidden_size),
            dtype=torch.bfloat16,
            requires_grad=False,
        ).to(cfg["device_crosscoder"])  # hardcoding 2 for model diffing
        self.cfg = cfg
        self.model_A = model_A
        self.model_B = model_B
        self.token_pointer = 0
        self.first = True
        self.normalize = True
        self.input_prompts = input_prompts

        # estimated_norm_scaling_factor_A = self.estimate_norm_scaling_factor(
        #     cfg["model_batch_size"], model_A)
        estimated_norm_scaling_factor_B = self.estimate_norm_scaling_factor(
            cfg["model_batch_size"], model_B)
        print("estimated_norm_scaling_factor_B")
        self.normalisation_factor = torch.tensor(
            [
                # estimated_norm_scaling_factor_A,
                estimated_norm_scaling_factor_B,
            ],
            device=cfg["device_crosscoder"],
            dtype=torch.float32,
        )
        self.refresh()

    @torch.no_grad()
    def estimate_norm_scaling_factor(self, batch_size, model: LanguageModel, n_batches_for_norm_estimate: int = 100):
        n_batches_for_norm_estimate = min(
            len(self.input_prompts) // batch_size, n_batches_for_norm_estimate)
        # stolen from SAELens https://github.com/jbloomAus/SAELens/blob/6d6eaef343fd72add6e26d4c13307643a62c41bf/sae_lens/training/activations_store.py#L370
        norms_per_batch = []
        # for i in range(n_batches_for_norm_estimate):
        prompts = self.input_prompts[0 * batch_size: (0 + 1) * batch_size]
        with model.trace(prompts) as tracer:
            acts = model.model.layers[16].mlp.output.save()
        norms_per_batch.append(acts.norm(dim=-1).mean().item())
        mean_norm = np.mean(norms_per_batch)
        scaling_factor = np.sqrt(model.config.hidden_size) / mean_norm
        return scaling_factor

    @torch.no_grad()
    def refresh(self):
        self.pointer = 0
        print("Refreshing the buffer!")
        with torch.autocast("cuda", torch.bfloat16):
            if self.first:
                num_batches = self.buffer_batches
            else:
                num_batches = self.buffer_batches // 2
            self.first = False
            for _ in tqdm.trange(0, num_batches, self.cfg["model_batch_size"]):
                prompts = self.input_prompts[
                    self.token_pointer: min(
                        self.token_pointer +
                        self.cfg["model_batch_size"], num_batches
                    )
                ]
                with self.model_A.trace(prompts) as tracer:
                    acts_A = self.model_A.model.layers[16].mlp.output.save()

                with self.model_B.trace(prompts) as tracer:
                    acts_B = self.model_B.model.layers[16].mlp.output.save()

                acts = torch.stack(
                    [acts_A, acts_B], dim=0)
                # [2, batch, seq_len, d_model]
                assert acts.shape == (
                    2, prompts.shape[0], prompts.shape[1], self.model_A.config.hidden_size)
                acts = einops.rearrange(
                    acts,
                    "n_layers batch seq_len d_model -> (batch seq_len) n_layers d_model",
                )

                self.buffer[self.pointer: self.pointer + acts.shape[0]] = acts
                self.pointer += acts.shape[0]
                self.token_pointer += self.cfg["model_batch_size"]

        self.pointer = 0
        self.buffer = self.buffer[
            torch.randperm(self.buffer.shape[0]).to(self.cfg["device_crosscoder"])
        ]

    @torch.no_grad()
    def next(self):
        out = self.buffer[self.pointer: self.pointer +
                          self.cfg["batch_size"]].float()
        # out: [batch_size, n_layers, d_model]
        self.pointer += self.cfg["batch_size"]
        if self.pointer > self.buffer.shape[0] // 2 - self.cfg["batch_size"]:
            self.refresh()
        if self.normalize:
            out = out * self.normalisation_factor[None, :, None]
        return out
