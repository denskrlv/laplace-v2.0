from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import random
import numpy as np
import torch
import torch.nn as nn
from .widelenet import WideLeNet

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def train_model(seed):
    set_seed(seed)

    # Load MNIST
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    model = WideLeNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(15):  # Keep it short to save time
        model.train()
        total_loss, correct = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x.size(0)
            correct += (output.argmax(1) == y).sum().item()

        acc = correct / len(train_loader.dataset)
        print(f"[Seed {seed}] Epoch {epoch+1}, Loss: {total_loss:.2f}, Accuracy: {acc:.4f}")

    return model

# Train two models on different seeds
model1 = train_model(seed=0)
model2 = train_model(seed=42)
model3 = train_model(seed=123)

# Save models
torch.save(model1.state_dict(), "widenet_seed0.pth")
torch.save(model2.state_dict(), "widenet_seed42.pth")