import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader

def get_cifar10(batch_size=128, num_workers=4, train=True):
    transform = T.Compose([T.ToTensor()])
    ds = torchvision.datasets.CIFAR10(
        root='./data', train=train, download=True, transform=transform
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=train, num_workers=num_workers)
