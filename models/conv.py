
import torch
from torch import nn


class Identity(nn.Module):
    
    def __init__(self):
        """
        Identity mapping
        """
        super().__init__()

    def forward(self, x):
        return x


def conv_block(
    in_channel: int,
    out_channel: int,
    kernel_size,
    stride,
    padding,
    inplace=False,
    norm=True,
    act=True
):
    layers = []
    layers.append(
        nn.Conv3d(in_channel, out_channel, kernel_size, stride, padding)
    )
    if norm:
        layers.append(nn.InstanceNorm3d(out_channel, affine=True, momentum=0.4))
    if act:
        layers.append(nn.ELU(inplace=inplace))
    return nn.Sequential(*layers)


class DownSample(nn.Module):

    def __init__(self, in_channel: int):
        super().__init__()
        self.map = nn.Sequential(
        nn.AvgPool3d(kernel_size=2, stride=2),
        nn.InstanceNorm3d(in_channel, affine=True, momentum=1),
        nn.ELU()
    )

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        return self.map(x), t


class UPSample(nn.Module):

    def __init__(self, in_channel: int):
        super().__init__()
        self.map = nn.Sequential(
        nn.Upsample(scale_factor=2),
        nn.InstanceNorm3d(in_channel, affine=True, momentum=1),
        nn.ELU()
    )

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        return self.map(x), t


class ResGNet(nn.Module):

    def __init__(self, in_channel: int, out_channel: int, num_steps: int):
        """
        in_channel: 
        out_channel:
        num_steps: The total amount of diffusion steps.
        """
        super().__init__()

        self.raw_in_channel = in_channel
        self.raw_out_channel = out_channel
        self.kernel_style = [
            (1, 3, 3),
            (1, 1, 3),
            (1, 3, 1),
            (3, 3, 1),
        ]
        self.split_num = len(self.kernel_style)
        if in_channel % self.split_num == 0 and out_channel % self.split_num == 0:
            in_channel //= self.split_num
            out_channel //= self.split_num

        blocks = []
        residuals = []
        padding_style = [
            (k1 // 2, k2 // 2, k3 // 2) for k1, k2, k3 in self.kernel_style
        ]

        for i in range(len(self.kernel_style)):
            cor = len(self.kernel_style) - i - 1
            blocks.append(nn.Sequential(
                conv_block(in_channel, in_channel, kernel_size=self.kernel_style[i], stride=1, padding=padding_style[i]),
                conv_block(in_channel, in_channel, kernel_size=self.kernel_style[cor], stride=1, padding=padding_style[cor]),
                conv_block(in_channel, out_channel, kernel_size=3, stride=1, padding=1),
            ))
            if in_channel == out_channel:
                residuals.append(
                    Identity(),
                )
            else:
                residuals.append(
                    conv_block(in_channel, out_channel, kernel_size=1, stride=1, padding=0)
                )

        self.scale = conv_block(out_channel * len(self.kernel_style), self.raw_out_channel, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1))
        self.blocks = nn.ModuleList(blocks)
        self.residuals = nn.ModuleList(residuals)
        
        if self.raw_in_channel != self.raw_out_channel:
            self.skip = conv_block(self.raw_in_channel, self.raw_out_channel, kernel_size=1, stride=1, padding=0)
        else:
            self.skip = None

        self.embedding = nn.Embedding(num_steps, self.raw_out_channel)
    
    def forward(self, x: torch.Tensor, t: torch.Tensor):
        """
        x: (batch_size, channel, depth, width, height)
        t: (batch_size, )
        """
        if self.skip:
            res = self.skip(x)
        else:
            res = x

        outputs = []
        if self.raw_in_channel % self.split_num != 0 or self.raw_out_channel % self.split_num != 0:
            for i in range(len(self.kernel_style)):
                if i % 2:
                    outputs.append(self.blocks[i](x) * self.residuals[i](x))
                else:
                    outputs.append(self.blocks[i](x) - self.residuals[i](x))
        else:
            splits = torch.split(x, self.raw_in_channel // self.split_num, dim=1)
            for i in range(len(self.kernel_style)):
                if i % 2:
                    outputs.append(self.blocks[i](splits[i]) * self.residuals[i](splits[i]))
                else:
                    outputs.append(self.blocks[i](splits[i]) + self.residuals[i](splits[i]))
        x = self.scale(torch.cat(outputs, dim=1))
        
        x = torch.add(x, res)

        # (batch_size, channel) -> (batch_size, channel, 1, 1, 1)
        embedding: torch.Tensor = self.embedding(t)
        x = torch.add(x, embedding.unsqueeze_(2).unsqueeze_(2).unsqueeze_(2))

        return x, t