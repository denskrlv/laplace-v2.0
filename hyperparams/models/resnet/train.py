import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import random
import os

from .resnet18 import ResNet18 # Assumes you saved the class in resnet_mnist.py

# ------------------- Settings -------------------
BATCH_SIZE = 128
NUM_EPOCHS = 15
SEEDS = [42, 0, 123]
SAVE_DIR = './checkpoints'
os.makedirs(SAVE_DIR, exist_ok=True)

# ------------------- Reproducibility -------------------
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# ------------------- Data Loaders -------------------
transform = transforms.Compose([
    transforms.Resize(32),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST(root='.', train=True, download=True, transform=transform)
test_dataset  = datasets.MNIST(root='.', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

# ------------------- Training & Evaluation -------------------
def train_one_epoch(model, optimizer, criterion, loader, device):
    model.train()
    total_loss = 0
    for data, target in loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        loss = criterion(model(data), target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def evaluate(model, loader, device):
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            pred = model(data).argmax(dim=1)
            correct += pred.eq(target).sum().item()
    return correct / len(loader.dataset)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

for seed in SEEDS:
    print(f"\n=== Training with seed {seed} ===")
    set_seed(seed)

    model = ResNet18().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss = train_one_epoch(model, optimizer, criterion, train_loader, device)
        test_acc = evaluate(model, test_loader, device)
        print(f"Seed {seed} | Epoch {epoch} | Loss: {train_loss:.4f} | Test Acc: {test_acc*100:.2f}%")

    ckpt_path = os.path.join(SAVE_DIR, f'resnet18_mnist_seed{seed}.pth')
    torch.save(model.state_dict(), ckpt_path)
    print(f"→ Saved model to {ckpt_path}")
