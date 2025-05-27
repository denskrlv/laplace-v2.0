import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    #simple two layer MLP
    def __init__(self, input_dim=784, hidden_dim=100, output_dim=10):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        x = x.view(x.size(0), -1)  #flatten images
        return self.net(x)