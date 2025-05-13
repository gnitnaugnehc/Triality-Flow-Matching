import torch
import torch.nn as nn
import torch.nn.functional as F
from .algebra import octonion_mul

class OctonionConv2d(nn.Module):
    def __init__(self, in_oct, out_oct, kernel_size, padding=1):
        super().__init__()
        self.in_oct = in_oct
        self.out_oct = out_oct
        self.weight = nn.Parameter(
            torch.randn(out_oct, in_oct, 8, 8, *kernel_size)
        )
        self.bias = nn.Parameter(torch.zeros(out_oct, 8))
        self.padding = padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, in_oct*8, H, W] -> reshape to [B, in_oct, 8, H, W]
        B, C, H, W = x.shape
        x = x.view(B, self.in_oct, 8, H, W)

        out = []
        for i in range(self.out_oct):
            acc = 0
            for j in range(self.in_oct):
                for p in range(8):
                    for q in range(8):
                        w = self.weight[i, j, p, q]
                        xq = x[:, j, q]
                        acc = acc + F.conv2d(
                            xq.unsqueeze(1),
                            w.unsqueeze(1),
                            padding=self.padding
                        )
            # reshape accumulator to [B,8,H,W], add bias
            oct_comp = acc.view(B, 8, H, W) + self.bias[i].view(1,8,1,1)
            out.append(oct_comp)
        # concat and flatten back to [B, out_oct*8, H, W]
        out = torch.stack(out, dim=1)  # [B, out_oct, 8, H, W]
        return out.view(B, self.out_oct*8, H, W)

class OctonionLinear(nn.Module):
    def __init__(self, in_oct, out_oct):
        super().__init__()
        self.in_oct = in_oct
        self.out_oct = out_oct
        self.weight = nn.Parameter(torch.randn(out_oct, in_oct, 8, 8))
        self.bias = nn.Parameter(torch.zeros(out_oct, 8))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, in_oct*8]
        B = x.size(0)
        x = x.view(B, self.in_oct, 8)
        y = torch.zeros(B, self.out_oct, 8, device=x.device)
        for i in range(self.out_oct):
            for j in range(self.in_oct):
                y[:, i] = y[:, i] + octonion_mul(x[:, j], self.weight[i, j])
        y = y + self.bias.unsqueeze(0)
        return y.view(B, self.out_oct*8)
