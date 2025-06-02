# --- laplace_paper_experiments/experiments/run_mnist_lenet_laplace.py ---
import torch
# Standard library imports
import torch.nn
import torch.optim
from torch.utils.data import DataLoader
import time
import os
import pandas as pd

from laplace import Laplace
from laplace.utils.metrics import RunningNLLMetric

from models.lenet_model import LeNet
from utils.data_utils import get_mnist_loaders  # Ensure this file is also updated
from utils.training_utils import train_map_model
from utils.evaluation_utils import evaluate_model

if __name__ == '__main__':

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # Hyperparameters
    BATCH_SIZE = 128
    MAP_EPOCHS = 100
    MAP_LR = 0.1
    MAP_WEIGHT_DECAY = 5e-4
    INITIAL_LA_PRIOR_PRECISION = 1.0

    all_results_list = []
    results_dir = "results_mnist"
    os.makedirs(results_dir, exist_ok=True)

    # checkpoint_dir and map_model_checkpoint_path logic for MAP model remains
    checkpoints_dir = "checkpoints_mnist"
    os.makedirs(checkpoints_dir, exist_ok=True)
    map_model_filename = f"map_lenet_mnist_epochs{MAP_EPOCHS}_lr{MAP_LR}_wd{MAP_WEIGHT_DECAY}.pth"
    map_model_checkpoint_path = os.path.join(checkpoints_dir, map_model_filename)

    # 1. Load Data
    # Removed seed argument from get_mnist_loaders call
    train_loader, val_loader, test_loader = get_mnist_loaders(
        batch_size=BATCH_SIZE, val_split=10000
    )

    # 2. Load or Train MAP model
    map_model_instance = LeNet(num_classes=10).to(device)

    if os.path.exists(map_model_checkpoint_path):  # Checkpoint name no longer includes seed
        print(f"\n--- Loading Pre-trained MAP Model from: {map_model_checkpoint_path} ---")
        map_model_instance.load_state_dict(torch.load(map_model_checkpoint_path, map_location=device))
    else:
        print(f"\n--- No Pre-trained MAP Model found at: {map_model_checkpoint_path} ---")
        print("--- Training MAP Model ---")
        map_model_instance = train_map_model(
            map_model_instance, train_loader, device,
            lr=MAP_LR, epochs=MAP_EPOCHS, weight_decay=MAP_WEIGHT_DECAY
        )
        print(f"--- Saving Trained MAP Model to: {map_model_checkpoint_path} ---")
        torch.save(map_model_instance.state_dict(), map_model_checkpoint_path)

    print("\n--- Evaluating MAP Model ---")
    map_eval_results = evaluate_model(map_model_instance, test_loader, device, model_name="MAP Baseline (LeNet)",
                                      num_classes=10)

    current_run_data = {
        "method_name": "MAP",
        "subset_of_weights": "all",
        "hessian_structure": "N/A",
        "optimized_prior_precision": "N/A",
        "map_epochs": MAP_EPOCHS,
        **map_eval_results
    }
    all_results_list.append(current_run_data)

    laplace_configurations = [
        {'name_suffix': 'LL-Diag', 'subset_of_weights': 'last_layer', 'hessian_structure': 'diag'},
        {'name_suffix': 'LL-KFAC', 'subset_of_weights': 'last_layer', 'hessian_structure': 'kron'},
        {'name_suffix': 'All-Diag', 'subset_of_weights': 'all', 'hessian_structure': 'diag'},
        {'name_suffix': 'All-KFAC', 'subset_of_weights': 'all', 'hessian_structure': 'kron'},
    ]

    for config in laplace_configurations:
        model_eval_name = f"Laplace LeNet ({config['name_suffix']}, Tuned Prior)"
        print(f"\n--- Applying Post-hoc Laplace ({config['name_suffix']}) to LeNet ---")
        la_model = None
        try:
            map_model_instance.to(device)
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

            print(f"\n--- Evaluating {model_eval_name} ---")
            la_eval_results = evaluate_model(la_model, test_loader, device,
                                             model_name=model_eval_name,
                                             num_classes=10)

            current_run_data = {
                "method_name": f"LA ({config['name_suffix']})",
                "subset_of_weights": config['subset_of_weights'],
                "hessian_structure": config['hessian_structure'],
                "optimized_prior_precision": optimized_prior_prec,
                "map_epochs": MAP_EPOCHS,
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
                "map_epochs": MAP_EPOCHS,
                "accuracy": float('nan'), "nll": float('nan'), "ece_l1": float('nan')
            })

    results_df = pd.DataFrame(all_results_list)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    results_filename = os.path.join(results_dir, f"mnist_lenet_laplace_all_configs_{timestamp}.csv")
    results_df.to_csv(results_filename, index=False)
    print(f"\nAll results saved to: {results_filename}")
    print("\nFinal Results Summary Table:")
    print(results_df.to_string())

    print("\n--- MNIST LeNet Experiments Finished ---")