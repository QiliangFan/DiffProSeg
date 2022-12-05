
import torch 
from torch import nn

class SegDiffLoss(nn.Module):
    
    def __init__(self):
        super().__init__()

        self.deta_loss = nn.SmoothL1Loss()

    def forward(
            self, 
            epsilon_theta: nn.Module, 
            img: torch.Tensor, 
            label: torch.Tensor, 
            alpha_bar: torch.Tensor,
            num_steps: int
        ):
        """
        the `t` in loss function is gnerated randomly, which should be as possiblily as different among cases.
        """
        epsilon = torch.randn_like(img)
        batch_size = img.shape[0]

        # 注意，如果下标是(N,), 从列表取出的元素shape也是(shape,);
        # 这样会导致系数无法和样本相乘，因此需要扩维（维度相同即可，但大小可为1将系数自动扩展）
        # (N,) -> (N, 1, 1, 1, 1)
        t = torch.randint(0, num_steps, size=(batch_size,)) 
        alpha_bar_t = alpha_bar[t.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)]

        x_t = torch.sqrt(alpha_bar_t) * label + torch.sqrt(1 - alpha_bar_t) * epsilon

        generated_noise = epsilon_theta(x_t, img, t)
        return self.deta_loss(epsilon - generated_noise)