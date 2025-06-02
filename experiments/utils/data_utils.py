# --- laplace_paper_experiments/scripts/data_utils.py ---
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.data import DataLoader, random_split
import torch

def get_mnist_loaders(batch_size=128, test_batch_size=1000, val_split=10000, seed=42):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    full_train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)

    num_train = len(full_train_dataset)
    train_size = num_train - val_split

    train_subset, val_subset = random_split(full_train_dataset, [train_size, val_split],
                                            generator=torch.Generator().manual_seed(42)) # for reproducibility

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=False) # pin_memory False for MPS warning
    val_loader = DataLoader(val_subset, batch_size=test_batch_size, shuffle=False, num_workers=1, pin_memory=False)
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False, num_workers=1, pin_memory=False)

    print(f"MNIST: Train samples: {len(train_subset)}, Val samples: {len(val_subset)}, Test samples: {len(test_dataset)}")
    return train_loader, val_loader, test_loader