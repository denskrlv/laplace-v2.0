# --- laplace_paper_experiments/experiments/run_mnist_lenet_laplace.py ---
import torch
# Standard library imports
import torch.nn
import torch.optim
from torch.utils.data import DataLoader
import time  # For timestamping results file
import os
import pandas as pd  # For saving results to CSV

# Assuming laplace-v2.0 is installed or in PYTHONPATH
from laplace import Laplace
from laplace.utils.metrics import RunningNLLMetric  # Directly import the correct metric

# --- IMPORT FROM YOUR PROJECT STRUCTURE ---
from models.lenet_model import LeNet
from utils.data_utils import get_mnist_loaders
from utils.training_utils import train_map_model
from utils.evaluation_utils import evaluate_model

import numpy as np
import torch.backends.cudnn as cudnn


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        if torch.cuda.device_count() > 1:
            torch.cuda.manual_seed_all(seed)
        cudnn.deterministic = True
        cudnn.benchmark = False


if __name__ == '__main__':
    SEED = 42
    set_seed(SEED)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # Hyperparameters
    BATCH_SIZE = 128
    MAP_EPOCHS = 100  # Set to 100 for proper MAP training before LA experiments
    MAP_LR = 0.1
    MAP_WEIGHT_DECAY = 5e-4
    INITIAL_LA_PRIOR_PRECISION = 1.0  # Initial guess for tuning, will be optimized

    # --- Results Collection ---
    all_results_list = []  # Use a different name to avoid conflict if 'all_results' is a module
    results_dir = "results_mnist"  # Specific directory for these results
    os.makedirs(results_dir, exist_ok=True)

    # 1. Load Data
    train_loader, val_loader, test_loader = get_mnist_loaders(
        batch_size=BATCH_SIZE, val_split=10000, seed=SEED
    )

    # 2. Train MAP model (ONCE)
    print("\n--- Training MAP Model ---")
    map_model_instance = LeNet(num_classes=10)
    map_model_instance = train_map_model(
        map_model_instance, train_loader, device,
        lr=MAP_LR, epochs=MAP_EPOCHS, weight_decay=MAP_WEIGHT_DECAY
    )

    print("\n--- Evaluating MAP Model ---")
    map_eval_results = evaluate_model(map_model_instance, test_loader, device, model_name="MAP Baseline (LeNet)",
                                      num_classes=10)

    current_run_data = {
        "method_name": "MAP",
        "subset_of_weights": "all",  # N/A for MAP, but for table consistency
        "hessian_structure": "N/A",
        "optimized_prior_precision": "N/A",
        **map_eval_results
    }
    all_results_list.append(current_run_data)

    # --- Define Laplace Configurations to Test ---
    laplace_configurations = [
        {'name_suffix': 'LL-Diag', 'subset_of_weights': 'last_layer', 'hessian_structure': 'diag'},
        {'name_suffix': 'LL-KFAC', 'subset_of_weights': 'last_layer', 'hessian_structure': 'kron'},
        {'name_suffix': 'All-Diag', 'subset_of_weights': 'all', 'hessian_structure': 'diag'},
        {'name_suffix': 'All-KFAC', 'subset_of_weights': 'all', 'hessian_structure': 'kron'},
        # {'name_suffix': 'LL-Full (LA*)', 'subset_of_weights': 'last_layer', 'hessian_structure': 'full'}, # Potentially LA*
    ]

    # --- Loop Through Laplace Configurations ---
    for config in laplace_configurations:
        model_eval_name = f"Laplace LeNet ({config['name_suffix']}, Tuned Prior)"
        print(f"\n--- Applying Post-hoc Laplace ({config['name_suffix']}) to LeNet ---")
        la_model = None
        try:
            la_model = Laplace(
                map_model_instance,
                'classification',
                subset_of_weights=config['subset_of_weights'],
                hessian_structure=config['hessian_structure'],
                prior_precision=INITIAL_LA_PRIOR_PRECISION
            )
            print(f"Fitting Laplace model ({config['name_suffix']})...")
            la_model.fit(train_loader)
            print("Laplace model fitting finished.")

            print(f"\nTuning Prior Precision for {config['name_suffix']} (Gridsearch)...")

            loss_for_gridsearch = RunningNLLMetric().to(device)

            la_model.optimize_prior_precision(
                method='gridsearch',
                val_loader=val_loader,
                loss=loss_for_gridsearch,
                pred_type='glm',
                link_approx='probit',
                verbose=True
            )
            optimized_prior_prec = la_model.prior_precision.item() if la_model.prior_precision.numel() == 1 else la_model.prior_precision.tolist()
            print(f"Optimized prior precision for {config['name_suffix']}: {optimized_prior_prec}")

            # Evaluate Laplace Model
            print(f"\n--- Evaluating {model_eval_name} ---")
            la_eval_results = evaluate_model(la_model, test_loader, device,
                                             model_name=model_eval_name,
                                             num_classes=10)

            current_run_data = {
                "method_name": f"LA ({config['name_suffix']})",
                "subset_of_weights": config['subset_of_weights'],
                "hessian_structure": config['hessian_structure'],
                "optimized_prior_precision": optimized_prior_prec,
                **la_eval_results
            }
            all_results_list.append(current_run_data)

        except Exception as e:
            print(f"AN ERROR OCCURRED DURING LAPLACE PHASE for {config['name_suffix']}: {e}")
            import traceback

            traceback.print_exc()
            all_results_list.append({
                "method_name": f"LA ({config['name_suffix']})",
                "subset_of_weights": config['subset_of_weights'],
                "hessian_structure": config['hessian_structure'],
                "optimized_prior_precision": "Error",
                "accuracy": float('nan'), "nll": float('nan'), "ece_l1": float('nan')
            })

    # --- Save All Results ---
    results_df = pd.DataFrame(all_results_list)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    results_filename = os.path.join(results_dir, f"mnist_lenet_laplace_all_configs_{timestamp}.csv")
    results_df.to_csv(results_filename, index=False)
    print(f"\nAll results saved to: {results_filename}")
    print("\nFinal Results Summary Table:")
    print(results_df.to_string())  # .to_string() for better console printing of DataFrame

    print("\n--- MNIST LeNet Experiments Finished ---")
    print("You should now have a CSV file with results for MAP and various Laplace configurations.")
    print("Next, you can adapt this workflow for CIFAR-10 and WRN.")