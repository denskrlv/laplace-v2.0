import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18

class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18, self).__init__()
        # Load base ResNet18
        self.model = resnet18(pretrained=False)
        # Change input conv layer to accept 1 channel (instead of 3)
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # Adjust final fully connected layer for 10 classes (MNIST)
        self.model.fc = nn.Linear(self.model.fc.in_features, 10)

    def forward(self, x):
        return self.model(x)
