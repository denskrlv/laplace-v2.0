# --- laplace_paper_experiments/scripts/evaluation_utils.py ---
import torch
import torch.nn.functional as F
from torchmetrics import CalibrationError # Make sure torchmetrics is installed

@torch.no_grad()
def evaluate_model(model_wrapper, test_loader, device, model_name="Model", num_classes=10):
    if isinstance(model_wrapper, torch.nn.Module):
        model_wrapper.eval()

    all_probs = []
    all_targets = []

    # Ensure num_classes is passed or derived correctly
    ece_metric_l1 = CalibrationError(task="multiclass", num_classes=num_classes, n_bins=15, norm='l1').to(device)

    for data, target in test_loader:
        data, target = data.to(device), target.to(device)

        if isinstance(model_wrapper, torch.nn.Module): # Standard model (MAP)
            output_logits = model_wrapper(data)
            probs = torch.softmax(output_logits, dim=-1)
        else: # Assuming it's a Laplace model
            probs = model_wrapper(data, pred_type='glm', link_approx='probit')

        all_probs.append(probs)
        all_targets.append(target)
        ece_metric_l1.update(probs, target)

    probs_cat = torch.cat(all_probs)
    targets_cat = torch.cat(all_targets)

    preds_classes = torch.argmax(probs_cat, dim=1)
    accuracy = (preds_classes == targets_cat).float().mean().item()
    nll = F.nll_loss(torch.log(probs_cat.clamp(min=1e-9)), targets_cat, reduction='mean').item()
    ece_l1 = ece_metric_l1.compute().item()

    print(f"\n--- {model_name} Evaluation Results ---")
    print(f"Accuracy: {accuracy*100:.2f}%")
    print(f"NLL: {nll:.4f}")
    print(f"ECE (L1): {ece_l1:.4f}")

    return {"accuracy": accuracy, "nll": nll, "ece_l1": ece_l1}