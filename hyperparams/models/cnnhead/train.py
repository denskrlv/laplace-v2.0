import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import random
import os
from .cnnhead import SmallCNN, LinearHead, FrozenCNNWithHead

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def train_frozen_linear(seed, cnn_epochs=5, head_epochs=15):
    set_seed(seed)
    os.makedirs("models", exist_ok=True)

    # Data
    transform = transforms.Compose([transforms.ToTensor()])
    train_set = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    train_loader = DataLoader(train_set, batch_size=64, shuffle=True)

    # Initialize backbone and linear head
    backbone = SmallCNN().to(device)
    head = LinearHead().to(device)
    loss_fn = nn.CrossEntropyLoss()

    # Step 1: Pre-train CNN with a temporary head
    print(f"Seed {seed}: Pretraining CNN...")
    temp_head = LinearHead().to(device)
    optimizer = torch.optim.Adam(list(backbone.parameters()) + list(temp_head.parameters()), lr=1e-3)
    backbone.train()
    temp_head.train()
    for epoch in range(cnn_epochs):
        total_loss, correct = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            features = backbone(x)
            output = temp_head(features)
            loss = loss_fn(output, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x.size(0)
            correct += (output.argmax(1) == y).sum().item()
        acc = correct / len(train_loader.dataset)
        print(f"[Seed {seed}] CNN Epoch {epoch+1}, Loss: {total_loss:.2f}, Accuracy: {acc:.4f}")

    # Freeze CNN
    for param in backbone.parameters():
        param.requires_grad = False
    backbone.eval()

    # Step 2: Train only linear head
    print(f"Seed {seed}: Training linear head...")
    optimizer = torch.optim.Adam(head.parameters(), lr=1e-3)
    for epoch in range(head_epochs):
        head.train()
        total_loss, correct = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                features = backbone(x)
            logits = head(features)
            loss = loss_fn(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x.size(0)
            correct += (logits.argmax(1) == y).sum().item()
        acc = correct / len(train_loader.dataset)
        print(f"[Seed {seed}] Head Epoch {epoch+1}, Loss: {total_loss:.2f}, Accuracy: {acc:.4f}")

    # Save backbone, head, and wrapped model
    torch.save(backbone.state_dict(), f"models/cnn_backbone_seed{seed}.pth")
    torch.save(head.state_dict(), f"models/linear_head_seed{seed}.pth")

    model = FrozenCNNWithHead(backbone, head)
    torch.save(model.state_dict(), f"models/final_model_seed{seed}.pth")

for seed in [0, 42, 123]:
    train_frozen_linear(seed=seed)