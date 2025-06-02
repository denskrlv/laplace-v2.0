# --- experiments/utils/training_utils.py ---
import time
import torch
import torch.optim as optim
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR

def train_map_model(model, train_loader, device, lr=0.1, epochs=100, weight_decay=5e-4):
    print(f"--- Starting MAP Model Training ({model.__class__.__name__}) ---")
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    total_start_time = time.time()

    for epoch in range(epochs):
        epoch_start = time.time()
        model.train()
        epoch_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        scheduler.step()
        avg_loss = epoch_loss / len(train_loader)
        epoch_time = time.time() - epoch_start
        remaining_epochs = epochs - epoch - 1
        estimated_remaining = epoch_time * remaining_epochs
        current_lr = scheduler.get_last_lr()[0]

        print(f"MAP Training: Epoch {epoch+1}/{epochs}, Avg Loss: {avg_loss:.4f}, LR: {current_lr:.1e}, "
              f"Epoch Time: {epoch_time:.2f}s, Estimated Remaining: {estimated_remaining:.2f}s")

    total_time = time.time() - total_start_time
    print(f"--- MAP Model Training Finished in {total_time:.2f}s ---")
    return model