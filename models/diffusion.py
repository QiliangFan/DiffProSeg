from pytorch_lightning import LightningModule
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim import lr_scheduler
from typing import Dict
import json

from metrics import Dice

class DiffusionModel(LightningModule):

    def __init__(self, epsilon_theta: nn.Module, num_steps: int, configuration: Dict):
        super().__init__()

        # configuration
        print("================= Configuration ==================")
        print(json.dumps(configuration, indent=4))
        print("================= Configuration ==================")
        self.config = configuration

        self.num_steps = 100
        betas = torch.linspace(-6, 6, num_steps)
        self.betas = torch.sigmoid(betas) * (0.5e-2 - 1e-5) + 1e-5
        self.alpha = 1 - self.betas
        self.alpha_bar = torch.cumprod(self.alpha, 0)
        self.one_minus_alpha_bar = (1 - self.alpha_bar).sqrt()

        # the neural network (`epsilon_\theta`)
        self.epsilon_theta = epsilon_theta

        # Loss function: compute the difference
        self.delta_loss = nn.SmoothL1Loss()

        # Metrics:
        self.dice = Dice()

        if torch.cuda.is_available():
            self.move_device(torch.device("cuda:0"))

    def configure_optimizers(self):
        # Optimizer
        optim = AdamW(self.epsilon_theta.parameters(), lr=1e-3)
        lr_sche = lr_scheduler.CosineAnnealingLR(optim, T_max=5)
        return {
            "optimizer": optim,
            "lr_scheduler": {
                "scheduler": lr_sche
            }
        } 

    def optimizer_step(
        self, 
        epoch, 
        epoch_idx: int,
        optimizer,
        optimizer_idx = 0,
        optimizer_closure = None,
        on_tpu = False,
        using_native_amp = False,
        using_lbfgs = False
        ):
        warm_up_step = self.config["warm_up"]
        lr = self.config["lr"]
        if self.trainer.global_step < warm_up_step:
            lr_scale = min(1.0, float(self.trainer.global_step + 1) / warm_up_step)
            for pg in optimizer.param_groups:
                pg["lr"] = lr_scale * lr
        optimizer.step(closure=optimizer_closure)

    def forward(self, img: torch.Tensor, label: torch.Tensor):
        """
        Compute the loss
        """
        epsilon = torch.randn_like(img, device=img.device)
        batch_size = img.shape[0]

        # 注意，如果t是(N,), 那么alpha用此下标取出的元素也是(N,)
        # 这样会导致系数无法与样本相乘, 因此需要扩展维度
        # (N,) -> (N, 1, 1, 1, 1)
        t = torch.randint(0, self.num_steps, size=(batch_size,), device=img.device)
        alpha_bar_t = self.alpha_bar[t.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)]

        x_t = torch.sqrt(alpha_bar_t) * label + torch.sqrt(1 - alpha_bar_t) * epsilon

        generated_noise = self.epsilon_theta(x_t, img, t)

        return self.delta_loss(generated_noise, epsilon)

    def training_step(self, batch, batch_idx: int):
        img, label = batch
        loss = self(img, label)
        return loss

    def test_step(self, batch, batch_idx: int):
        img, label = batch
        pred = self.reverse_process(img)
        dice = self.dice(pred, label)
        metrics = {
            "Dice": dice.item()
        }
        self.log_dict(metrics, prog_bar=True, on_step=True)
        return metrics

    @torch.no_grad()
    def diffusion_process(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Can be implemented with the inductive manner (`q_{x_t}(\cdot)`)
        """
        z = torch.rand_like(x, device=x.device)
        alpha_bar_t = self.alpha_bar[t]
        xt = torch.sqrt(alpha_bar_t) * x + torch.sqrt(1 - alpha_bar_t) * z
        return xt

    @torch.no_grad()
    def reverse_process(self, img: torch.Tensor):
        """
        The reverse process is only related to inference stage (withou contribution to training process)
        """
        x = torch.randn_like(img, device=img.device)
        for t in reversed(range(self.num_steps)):
            t = torch.tensor([t])
            x = self.sample(img, x, t)
        return x

    @torch.no_grad()
    def sample(self, img: torch.Tensor, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Single step of total reverse process (with multiple sample steps)
        Sample one image each time.
        """
        assert img.shape[0] == 1, f"In inference stage, the batch size should be 1, but got {img.shape[0]}"
        z = torch.randn_like(img, device=img.device)

        alpha_t = self.alpha[t]
        alpha_bar_t = self.alpha_bar[t]
        beta_t = 1 - alpha_t

        epsilon_output = self.epsilon_theta(x, img, t)
        # 尾部的最后一项还是参考的原始Diffusion Model的公式
        x_t_minus_1 = 1 / torch.sqrt(alpha_t) * (x - beta_t/(torch.sqrt(1 - alpha_bar_t))*epsilon_output) + beta_t.sqrt() * z
        return x_t_minus_1
    
    def move_device(self, device: torch.device):
        self.betas = self.betas.to(device)
        self.alpha = self.alpha.to(device)
        self.alpha_bar = self.alpha_bar.to(device)
        self.one_minus_alpha_bar = self.one_minus_alpha_bar.to(device)