import torch.nn as nn
import torch.nn.functional as F

from models.model import Model


class SimpleNet(Model):
    def __init__(self, in_channels=1, kernel_size=5, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 20, kernel_size, 1)
        self.conv2 = nn.Conv2d(20, 50, kernel_size, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.fc2 = nn.Linear(500, num_classes)

    def features(self, x):
         x = F.relu(self.conv1(x))
         x = F.max_pool2d(x, 2, 2)
         x = F.relu(self.conv2(x))
         x = F.max_pool2d(x, 2, 2)
         return x

    def forward(self, x, latent=False):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        if x.requires_grad:
            x.register_hook(self.activations_hook)
        x = x.view(-1, 4 * 4 * 50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        out = F.log_softmax(x, dim=1)
        if latent:
            return out, x
        else:
            return out

class SimpleNetCIFAR(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(128, 20, 2, 1)
        self.conv2 = nn.Conv2d(20, 50, 2, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.fc2 = nn.Linear(500, num_classes)

    # def features(self, x):
    #     x = F.relu(self.conv1(x))
    #     x = F.max_pool2d(x, 2, 1)
    #     x = F.relu(self.conv2(x))
    #     x = F.max_pool2d(x, 2, 1)
    #     return x

    def forward(self, x, latent=False):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        if x.requires_grad:
            x.register_hook(self.activations_hook)
        x = x.view(-1, 4 * 4 * 50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        out = F.log_softmax(x, dim=1)
        if latent:
            return out, x
        else:
            return out