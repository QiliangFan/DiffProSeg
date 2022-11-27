"""
Unlike traditional U-Net (3D), this module is conditioned on time steps of diffusion process.
"""

import torch
from torch import nn

class UNet(nn.Module):

    def __init__(self):
        super().__init__()
        pass

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        """
        x: input data  (batch_size, channel, depth, width, height)
        t: time step  (batch_size,)
        """
        pass