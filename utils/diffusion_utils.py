import torch 
import numpy as np


def get_betas(num_steps: int) -> torch.Tensor:
    betas = torch.linspace(-6, 6, num_steps)
    betas = torch.sigmoid(betas) * (0.5e-2 - 1e-5) + 1e-5
    return betas


