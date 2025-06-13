import torch
from laplace import Laplace
from tests.utils.data_utils import get_adult_loaders
from tests.baselines.vanilla.models.mlp_tabular import MLPTabular
import warnings

warnings.filterwarnings('ignore')

print("--- Starting Quick Compatibility Check for Adult Dataset ---")

# 1. Set up device and data path
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
data_path = './data'

# 2. Load data to get one batch and the number of features
try:
    print("Loading Adult dataset...")
    (train_loader, _, _), num_features = get_adult_loaders(
        data_path=data_path, batch_size=32, device=device
    )
    X, y = next(iter(train_loader))
    print("Data loaded successfully.")
    print(f"Input batch shape: {X.shape}")
    print(f"Target batch shape: {y.shape}")
except Exception as e:
    print(f"\n[ERROR] Failed to load data: {e}")
    exit()

# 3. Instantiate the model
try:
    print("\nInstantiating MLPTabular model...")
    model = MLPTabular(num_features=num_features, num_classes=2).to(device)
    print("Model instantiated successfully.")
except Exception as e:
    print(f"\n[ERROR] Failed to create model: {e}")
    exit()

# 4. Instantiate the Laplace class with the model
try:
    print("\nInstantiating Laplace with the model...")
    # We will test the last-layer Laplace, as it's a common use case.
    la = Laplace(model, 'classification', subset_of_weights='last_layer', hessian_structure='diag')
    print("Laplace instantiated successfully.")
except Exception as e:
    print(f"\n[ERROR] Failed to instantiate Laplace: {e}")
    exit()

# 5. The core test: Fit Laplace on a single batch
try:
    print("\nFitting Laplace on a single batch of data...")
    # This will perform the forward and backward passes needed to compute curvature.
    la.fit(train_loader) # The fit method can take a dataloader directly
    print("\n✅ Quick check successful! The Laplace library is compatible with your model and data.")
except Exception as e:
    print(f"\n[ERROR] The compatibility check failed during the fit step: {e}")

print("\n--- End of Check ---")