"""
Unlike traditional U-Net (3D), this module is conditioned on time steps of diffusion process.
"""

import torch
from torch import nn

from .conv import ResGNet, downsample_layer, upsample_layer


class UNet(nn.Module):

    def __init__(self, num_steps: int):
        super().__init__()

        cur_channel = 4
        expand = 4
        num_layer = 4
        self.num_layer = num_layer

        self.down_layers = []
        self.up_layers = []

        # the first/last layer
        self.down_layers.append(nn.Sequential(
            ResGNet(1, cur_channel, num_steps),
            ResGNet(cur_channel, cur_channel, num_steps)
        ))
        self.up_layers.append(nn.Sequential(
            ResGNet(cur_channel * 2, cur_channel, num_steps),
            ResGNet(cur_channel, 1, num_steps)
        ))

        # the down/layer
        for i in range(1, num_layer):
            next_channel = cur_channel * expand
            self.down_layers.append(nn.Sequential(
                downsample_layer(cur_channel),
                ResGNet(cur_channel, next_channel, num_steps),
                ResGNet(next_channel, next_channel, num_steps),
            ))

            self.up_layers.insert(0, nn.Sequential(
                ResGNet(next_channel * 2, next_channel, num_steps),
                ResGNet(next_channel, cur_channel, num_steps),
                upsample_layer(cur_channel)
            ))
            cur_channel = next_channel

        self.bottom = nn.Sequential(
            downsample_layer(cur_channel),
            ResGNet(cur_channel, cur_channel, num_steps),
            upsample_layer(cur_channel),
        )


    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        x: input data  (batch_size, channel, depth, width, height)
        t: time step  (batch_size,)
        """
        forward_features = []
        for i in range(self.num_layer):
            x  = self.down_layers[i](x, t)
            forward_features.insert(0, x)
        
        x = self.bottom(x, t)

        for i in range(self.num_layer):
            x = torch.cat([x, forward_features[i]], dim=1)
            x = self.up_layers[t](x, t)
        
        return x

class CondImageUNet(UNet):
    """
    PyTorch version implementation of SegDiff
    """

    def __init__(self, num_steps: int):
        super().__init__(num_steps)
        
        self.F = ResGNet(1, 1, num_steps=num_steps)
        self.G = ResGNet(1, 1, num_steps=num_steps)


    def forward(self, xt: torch.Tensor, image: torch.Tensor, t: torch.Tensor):
        G = self.G(image)
        F = self.F(xt)
        xt = F + G
        xt_minus_1 = super()(xt)
        return xt_minus_1