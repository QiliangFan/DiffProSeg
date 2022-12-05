from pytorch_lightning import LightningModule
import torch
from torch import nn

class DiffusionModel(LightningModule):

    def __init__(self, num_steps: int):
        super().__init__()

        self.num_steps = 100
        betas = torch.linspace(-6, 6, num_steps)
        self.betas = torch.sigmoid(betas) * (0.5e-2 - 1e-5) + 1e-5
        self.alpha = 1 - self.betas
        self.alpha_bar = torch.cumprod(self.alpha, 0)
        self.one_minus_alpha_bar = (1 - self.alpha_bar).sqrt()

    def forward(self, x: torch.Tensor):
        pass

    def diffusion_process(self, x: torch.Tensor):
        """
        Can be implemented with the inductive manner
        """
        pass

    def reverse_process(self, x: torch.Tensor):
        """
        The reverse process is only related to inference stage (withou contribution to training process)
        """
        pass

    def sample(self, x0: torch.Tensor, t: torch.Tensor):
        """
        Single step of total reverse process (with multiple sample steps)

        Params:
        x0: (batch, ch=1, z, y, x)
        t: (batch, )
        """
        pass
    