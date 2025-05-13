import torch.nn as nn
import torch.nn.functional as F
from .layers import OctonionConv2d, OctonionLinear

class OctonionFlowField(nn.Module):
    def __init__(self, num_oct=1, hidden_oct=4):
        super().__init__()
        self.conv1 = OctonionConv2d(num_oct, hidden_oct, kernel_size=(3,3))
        self.conv2 = OctonionConv2d(hidden_oct, hidden_oct, kernel_size=(3,3))
        # after two 2×2 pools: spatial dims 32→16→8
        flatten_size = hidden_oct * 8 * 8
        self.fc = OctonionLinear(flatten_size, num_oct * 32 * 32 // 8)

    def forward(self, t, x):
        # x: [B, num_oct*8, 32, 32]
        h = F.relu(self.conv1(x))
        h = F.avg_pool2d(h, 2)
        h = F.relu(self.conv2(h))
        h = F.avg_pool2d(h, 2)
        B = x.size(0)
        h = h.view(B, -1)
        out = self.fc(h)
        return out.view_as(x)
