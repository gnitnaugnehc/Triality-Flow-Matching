import torch
import torch.nn.functional as F
from torch.optim import Adam
from .model import OctonionFlowField
from .data import get_cifar10
from .transforms import triality_transform

def train(epochs=20, lr=1e-3, device='cuda'):
    loader = get_cifar10(train=True)
    model = OctonionFlowField().to(device)
    opt = Adam(model.parameters(), lr=lr)

    for ep in range(1, epochs+1):
        total = 0
        for imgs, _ in loader:
            imgs = imgs.to(device)*2 - 1  # normalize to [-1,1]
            B = imgs.size(0)
            # pad 3→8 channels
            x0 = torch.zeros(B, 8,32,32, device=device)
            x0[:, :3] = imgs
            x0 = x0.unsqueeze(1).expand(-1,1,-1,-1,-1)
            x0 = x0.reshape(B, -1,32,32)

            t = torch.rand(B, device=device).view(B,1)
            eps = torch.randn_like(x0)
            xt = torch.sqrt(1-t**2).view(B,1,1,1)*x0 + t.view(B,1,1,1)*eps
            grad = (eps - t.view(B,1,1,1)*x0) / torch.sqrt(1-t**2).view(B,1,1,1)

            vt = model(t, xt)
            # equivariance loss
            Gxt = triality_transform(xt)
            Gvt = triality_transform(vt)
            vt_G = model(t, Gxt)
            loss = F.mse_loss(vt, grad) + 0.1 * F.mse_loss(vt_G, Gvt)

            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()*B

        print(f"Epoch {ep}/{epochs} — avg loss {total/len(loader.dataset):.4f}")

if __name__ == '__main__':
    train()
