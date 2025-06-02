# --- laplace_paper_experiments/utils/training_utils.py ---
import torch
import torch.optim as optim
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm.auto import tqdm  # Import tqdm


def train_map_model(model, train_loader, device, lr=0.1, epochs=100, weight_decay=5e-4):
    print(f"--- Starting MAP Model Training ({model.__class__.__name__}) ---")
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    # Outer loop for epochs with tqdm
    epoch_pbar = tqdm(range(epochs), desc="MAP Training Progress")

    for epoch in epoch_pbar:
        model.train()
        total_epoch_loss = 0
        num_batches = 0

        # Inner loop for batches with tqdm
        batch_pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}", leave=False)
        for batch_idx, (data, target) in enumerate(batch_pbar):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            total_epoch_loss += loss.item()
            num_batches += 1

            # Update inner progress bar postfix with current batch loss
            if batch_idx % 10 == 0:  # Update less frequently to avoid too much flicker
                batch_pbar.set_postfix_str(f"Batch Loss: {loss.item():.4f}")

        scheduler.step()
        avg_epoch_loss = total_epoch_loss / num_batches
        current_lr = scheduler.get_last_lr()[0]

        # Update outer progress bar description or postfix with epoch summary
        epoch_pbar.set_postfix_str(f"Last Epoch Avg Loss: {avg_epoch_loss:.4f}, LR: {current_lr:.1e}")
        # Optionally, print a summary line less frequently if needed, e.g., every 5-10 epochs
        if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
            print(
                f"MAP Training: Epoch {epoch + 1}/{epochs} completed. Avg Loss: {avg_epoch_loss:.4f}, LR: {current_lr:.1e}")

    print(f"--- MAP Model Training Finished after {epochs} epochs ---")
    return model