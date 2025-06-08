import torch.nn as nn
import torch.nn.functional as F

class WideLeNet(nn.Module):
    def __init__(self):
        super(WideLeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 24, kernel_size=5)   # Output: (24, 28, 28)
        self.conv2 = nn.Conv2d(24, 64, kernel_size=5)  # Output: (64, 12, 12)
        self.fc1 = nn.Linear(64 * 5 * 5, 256)          # Updated to 1600
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))  # 32x32 → 14x14
        x = F.relu(F.max_pool2d(self.conv2(x), 2))  # 14x14 → 5x5
        x = x.view(-1, 64 * 5 * 5)  # Flatten from (64, 5, 5) = 1600
        x = F.relu(self.fc1(x))
        return self.fc2(x)