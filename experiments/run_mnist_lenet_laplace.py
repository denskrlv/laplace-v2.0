# --- laplace_paper_experiments/experiments/run_mnist_lenet_laplace.py ---
import torch
# Standard library imports that might be needed by the imported functions
import torch.nn # For type hinting if used in signatures
import torch.optim # For type hinting
from torch.utils.data import DataLoader # For type hinting

# Assuming laplace2-v2.0 is installed or in PYTHONPATH
from laplace import Laplace

# --- IMPORT FROM YOUR NEW STRUCTURE ---
from models.lenet_model import LeNet
from utils.data_utils import get_mnist_loaders
from utils.training_utils import train_map_model
from utils.evaluation_utils import evaluate_model

if __name__ == '__main__':
    # Set the device to GPU if available, otherwise mps otherwise CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # Hyperparameters from Appendix C (for LeNet on MNIST)
    BATCH_SIZE = 128
    MAP_EPOCHS = 100
    MAP_LR = 0.1      # Initial learning rate for Adam
    MAP_WEIGHT_DECAY = 5e-4
    LA_PRIOR_PRECISION = 50.0

    # 1. Load Data
    # get_mnist_loaders is now imported
    train_loader, val_loader, test_loader = get_mnist_loaders(batch_size=BATCH_SIZE, val_split=10000)

    # 2. Train MAP model (Using the imported LeNet)
    map_model_instance = LeNet(num_classes=10)
    # train_map_model is now imported
    map_model_instance = train_map_model(
        map_model_instance, train_loader, device,
        lr=MAP_LR, epochs=MAP_EPOCHS, weight_decay=MAP_WEIGHT_DECAY
    )

    # evaluate_model is now imported
    map_results = evaluate_model(map_model_instance, test_loader, device, model_name="MAP Baseline (Imported LeNet)", num_classes=10)

    # 3. Apply and Fit Post-hoc Laplace (Last-Layer KFAC)
    print("\n--- Applying Post-hoc Laplace (Last-Layer KFAC) to Imported LeNet ---")
    la_model = Laplace(
        map_model_instance,
        'classification',
        subset_of_weights='last_layer',
        hessian_structure='diag',
        prior_precision=LA_PRIOR_PRECISION
    )
    la_model.fit(train_loader)
    print("--- Laplace Model Fitting Finished ---")

    print("\n--- Tuning Laplace Prior Precision ---")
    # Ensure your val_loader is passed correctly
    la_model.optimize_prior_precision(
        method='marglik',  # Or 'gridsearch' if you prefer and provide a loss for CV
        val_loader=val_loader,
        pred_type='glm',  # GLM predictive is usually used for this
        link_approx='probit',
        verbose=True
    )

    # 4. Evaluate Laplace Model
    la_results = evaluate_model(la_model, test_loader, device, model_name="Laplace on Imported LeNet (LL KFAC)", num_classes=10)

    print("\n--- Summary ---")
    print(f"MAP (Imported LeNet): Acc={map_results['accuracy']:.4f}, NLL={map_results['nll']:.4f}, ECE_L1={map_results['ece_l1']:.4f}")
    print(f"LA on Imported LeNet (LL KFAC): Acc={la_results['accuracy']:.4f}, NLL={la_results['nll']:.4f}, ECE_L1={la_results['ece_l1']:.4f}")