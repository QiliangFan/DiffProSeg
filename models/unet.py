"""
Unlike traditional U-Net (3D), this module is conditioned on time steps of diffusion process.
"""

import torch
from torch import nn

from .conv import ResGNet, DownSample, UPSample


class DownLayer(nn.Module):

    def  __init__(self, cur_channel: int, next_channel: int, num_steps: int, endpoint = False):
        super().__init__()
        if endpoint:
            # the first layer
            modules = [
                ResGNet(cur_channel, next_channel, num_steps),
                ResGNet(next_channel, next_channel, num_steps)
            ]
        else:
            # the normal down layers
            modules = [
                DownSample(cur_channel),
                ResGNet(cur_channel, next_channel, num_steps),
                ResGNet(next_channel, next_channel, num_steps),
            ]
        self.module = nn.ModuleList(modules)

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        for i in range(len(self.module)):
            x, t = self.module[i](x, t)
        return x, t


class UpLayer(nn.Module):
    
    def __init__(self, cur_channel: int, next_channel: int, num_steps: int, endpoint = False):
        super().__init__()
        if endpoint:
            # the last layer
            modules = [
                ResGNet(cur_channel, next_channel, num_steps),
                ResGNet(next_channel, 1, num_steps)
            ]
        else:
            modules = [
                ResGNet(cur_channel, cur_channel // 2, num_steps),
                ResGNet(cur_channel // 2, next_channel, num_steps),
                UPSample(next_channel)
            ]
        self.module = nn.ModuleList(modules)

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        for i in range(len(self.module)):
            x, t = self.module[i](x, t)
        return x, t

class BottomLayer(nn.Module):

    def __init__(self, feature_dim: int, num_steps: int):
        super().__init__()
        modules = [
            DownSample(feature_dim),
            ResGNet(feature_dim, feature_dim, num_steps),
            UPSample(feature_dim)
        ]
        self.module = nn.ModuleList(modules)

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        for i in range(len(self.module)):
            x, t = self.module[i](x, t)
        return x, t


class UNet(nn.Module):

    def __init__(self, num_steps: int):
        super().__init__()

        cur_channel = 4
        expand = 4
        num_layer = 4
        self.num_layer = num_layer

        down_layers = []
        up_layers = []

        # the first/last layer
        down_layers.append(DownLayer(1, cur_channel, num_steps, endpoint=True))
        up_layers.append(UpLayer(cur_channel * 2, cur_channel, num_steps, endpoint=True))

        # the down/up layer
        for i in range(1, num_layer):
            next_channel = cur_channel * expand
            down_layers.append(DownLayer(cur_channel, next_channel, num_steps))
            up_layers.insert(0, UpLayer(next_channel * 2, cur_channel, num_steps))
            cur_channel = next_channel
        
        self.down_layers = nn.ModuleList(down_layers)
        self.up_layers = nn.ModuleList(up_layers)

        self.bottom = BottomLayer(cur_channel, num_steps)

        self.act = nn.Sigmoid()


    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        x: input data  (batch_size, channel, depth, width, height)
        t: time step  (batch_size,)
        """
        forward_features = []
        for i in range(self.num_layer):
            x, t = self.down_layers[i](x, t)
            forward_features.insert(0, x)
        
        x, t = self.bottom(x, t)

        for i in range(self.num_layer):
            x = torch.cat([x, forward_features[i]], dim=1)
            x, t = self.up_layers[i](x, t)
        x = self.act(x)
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
        G, t = self.G(image, t)
        F, t = self.F(xt, t)
        x = F + G

        # Original UNet
        forward_features = []
        for i in range(self.num_layer):
            x, t = self.down_layers[i](x, t)
            forward_features.insert(0, x)
        
        x, t = self.bottom(x, t)

        for i in range(self.num_layer):
            x = torch.cat([x, forward_features[i]], dim=1)
            x, t = self.up_layers[i](x, t)
        x = self.act(x)
        # x_t_minus_1
        return x