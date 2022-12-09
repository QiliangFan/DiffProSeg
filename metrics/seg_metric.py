import torch
from torch import nn



class Dice:

    def __init__(self):
        super().__init__()

    def __call__(self, img: torch.Tensor, gt: torch.Tensor):
        img = img.flatten(start_dim=1)
        gt = img.flatten(start_dim=1)

        inter = torch.sum(img * gt, dim=1)
        union = torch.sum(torch.pow(img, 2), dim=1) + torch.sum(torch.pow(gt, 2), dim=1)
        return torch.mean((2 * inter + 1) / (union + 1), dim=0)