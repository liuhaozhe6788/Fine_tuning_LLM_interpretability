from utils import *
from crosscoder import CrossCoder
import tqdm
from nnsight import LanguageModel
from torch.nn.utils import clip_grad_norm_
from datasets import Dataset


class Trainer:
    def __init__(self, cfg, base_model_acts: Dataset, ft_model_acts: Dataset):
        self.cfg = cfg
        self.crosscoder = CrossCoder(cfg)
        self.base_model_acts = base_model_acts
        self.ft_model_acts = ft_model_acts
        self.total_steps = len(base_model_acts) // cfg["batch_size"]
        self.residual_batch_size = len(base_model_acts) % cfg["batch_size"]
        self.optimizer = torch.optim.Adam(
            self.crosscoder.parameters(),
            lr=cfg["lr"],
            betas=(cfg["beta1"], cfg["beta2"]),
        )
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, self.lr_lambda
        )
        self.step_counter = 0

        wandb.init(project=cfg["wandb_project"], entity=cfg["wandb_entity"])

    def lr_lambda(self, step):
        if step < 0.8 * self.total_steps:
            return 1.0
        else:
            return 1.0 - (step - 0.8 * self.total_steps) / (0.2 * self.total_steps)

    def get_l1_coeff(self):
        # Linearly increases from 0 to cfg["l1_coeff"] over the first 0.05 * self.total_steps steps, then keeps it constant
        if self.step_counter < 0.05 * self.total_steps:
            return self.cfg["l1_coeff"] * self.step_counter / (0.05 * self.total_steps)
        else:
            return self.cfg["l1_coeff"]

    def step(self):
        acts_A = self.base_model_acts[self.step_counter:self.step_counter + self.cfg["batch_size"]]
        acts_B = self.ft_model_acts[self.step_counter:self.step_counter + self.cfg["batch_size"]]
        acts_A = torch.tensor(acts_A["base_model_acts"])
        acts_B = torch.tensor(acts_B["ft_model_acts"])
        acts_A = einops.rearrange(acts_A, "batch d_model -> batch 1 d_model")
        acts_B = einops.rearrange(acts_B, "batch d_model -> batch 1 d_model")
        acts = torch.cat([acts_A, acts_B], dim=1)
        acts = acts.to(self.cfg["device"])
        acts = acts.to(self.crosscoder.dtype)
        losses = self.crosscoder.get_losses(acts)
        loss = losses.l2_loss + self.get_l1_coeff() * losses.l1_loss
        loss.backward()
        clip_grad_norm_(self.crosscoder.parameters(), max_norm=1.0)
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad()

        loss_dict = {
            "loss": loss.item(),
            "l2_loss": losses.l2_loss.item(),
            "l1_loss": losses.l1_loss.item(),
            "l0_loss": losses.l0_loss.item(),
            "l1_coeff": self.get_l1_coeff(),
            "lr": self.scheduler.get_last_lr()[0],
            "explained_variance": losses.explained_variance.mean().item(),
            "explained_variance_A": losses.explained_variance_A.mean().item(),
            "explained_variance_B": losses.explained_variance_B.mean().item(),
        }
        self.step_counter += self.cfg["batch_size"]
        return loss_dict

    def step_residual(self):
        acts_A = self.base_model_acts[self.step_counter:self.step_counter + self.residual_batch_size]
        acts_B = self.ft_model_acts[self.step_counter:self.step_counter + self.residual_batch_size]
        acts_A = torch.tensor(acts_A["base_model_acts"])
        acts_B = torch.tensor(acts_B["ft_model_acts"])
        acts_A = einops.rearrange(acts_A, "batch d_model -> batch 1 d_model")
        acts_B = einops.rearrange(acts_B, "batch d_model -> batch 1 d_model")
        acts = torch.cat([acts_A, acts_B], dim=1)
        acts = acts.to(self.cfg["device"])
        acts = acts.to(self.crosscoder.dtype)
        losses = self.crosscoder.get_losses(acts)
        loss = losses.l2_loss + self.get_l1_coeff() * losses.l1_loss
        loss.backward()
        clip_grad_norm_(self.crosscoder.parameters(), max_norm=1.0)
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad()

        loss_dict = {
            "loss": loss.item(),
            "l2_loss": losses.l2_loss.item(),
            "l1_loss": losses.l1_loss.item(),
            "l0_loss": losses.l0_loss.item(),
            "l1_coeff": self.get_l1_coeff(),
            "lr": self.scheduler.get_last_lr()[0],
            "explained_variance": losses.explained_variance.mean().item(),
            "explained_variance_A": losses.explained_variance_A.mean().item(),
            "explained_variance_B": losses.explained_variance_B.mean().item(),
        }
        self.step_counter += self.residual_batch_size
        return loss_dict

    def log(self, loss_dict):
        wandb.log(loss_dict, step=self.step_counter)
        print(loss_dict)

    def save(self):
        self.crosscoder.save(self.optimizer, self.scheduler)

    def train(self):
        self.step_counter = 0
        try:
            for i in tqdm.trange(self.total_steps, desc="Training"):
                loss_dict = self.step()
                if i % self.cfg["log_every"] == 0:
                    self.log(loss_dict)
                if (i + 1) % self.cfg["save_every"] == 0:
                    self.save()
            if self.residual_batch_size > 0:
                loss_dict = self.step_residual()
                self.log(loss_dict)
                self.save()
        finally:
            self.save()