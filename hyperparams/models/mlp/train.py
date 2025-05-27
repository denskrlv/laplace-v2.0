import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
from .mlp import MLP

def train(model, device, loader, optimizer, criterion, epoch):
    #model in training mode
    model.train()
    #loop over training batches, move data and labels to GPU if available
    for batch_idx, (x, y) in enumerate(loader):
        x, y = x.to(device), y.to(device)

        # run forward pass and computes the loss (cross-entropy)
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)

        #Backpropagation
        loss.backward()
        optimizer.step()

        #log
        if batch_idx % 100 == 0:
            print(f"Train Epoch: {epoch} [{batch_idx * len(x)}/{len(loader.dataset)}] Loss: {loss.item():.6f}")

def test(model, device, loader, criterion):
    #Evaluation mode
    model.eval()
    loss = 0
    correct = 0

    #Inference without computing gradients
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            output = model(x)
            loss += criterion(output, y).item() * x.size(0)
            pred = output.argmax(dim=1)
            correct += pred.eq(y).sum().item()
    #Compute total test loss and count how many predictions are correct
    loss /= len(loader.dataset)
    accuracy = 100. * correct / len(loader.dataset)

    #Compute average loss and accuracy
    print(f"\nTest set: Avg loss: {loss:.4f}, Accuracy: {correct}/{len(loader.dataset)} ({accuracy:.2f}%)\n")
    return accuracy

def main():
    #Define hyperparameters
    batch_size = 128
    epochs = 20
    lr = 1e-3
    model_path = os.path.join(os.path.dirname(__file__), "pretrained", "mlp_mnist.pth")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # MNIST transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    #download datasets
    train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_data = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    
    #Wrap datasets into PyTorch DataLoaders for batching
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, pin_memory=True)
    test_loader = DataLoader(test_data, batch_size=256, shuffle=False, pin_memory=True)

    # Model, loss, optimizer
    model = MLP().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Training loop
    for epoch in range(1, epochs + 1):
        train(model, device, train_loader, optimizer, criterion, epoch)
        test(model, device, test_loader, criterion)

    # Save model
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save(model.state_dict(), model_path)
    print(f"Saved trained model to {model_path}")

if __name__ == '__main__':
    main()
