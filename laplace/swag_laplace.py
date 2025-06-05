from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.utils import parameters_to_vector

from laplace.baselaplace import DiagLaplace
from laplace.utils.swag import SWAG


class SWAGLaplace(DiagLaplace):

    _key = ("all", "swag_laplace")

    def __init__(
        self,
        model: nn.Module,
        likelihood: str,
        n_models: int = 20,
        start_epoch: int = 0,
        swa_freq: int = 1,
        swa_lr: float = 0.05,
        max_num_models: int = 20,
        var_clamp: float = 1e-30,
        prior_precision=1.0, # Example, ensure these match DiagLaplace needs
        temperature=1.0,     # Example
        device = None,
        **kwargs
    ):
        super().__init__(model, likelihood, prior_precision=prior_precision, 
            temperature=temperature, **kwargs)

        self.device = device if device is not None else next(model.parameters()).device
        
        # Initialize SWAG
        self.swag = SWAG(
            model=model,
            n_models=n_models,
            start_epoch=start_epoch,
            swa_freq=swa_freq,
            swa_lr=swa_lr,
            max_num_models=max_num_models,
            var_clamp=var_clamp,
            device=self.device
        )
        
        # Initialize storage for SWAG statistics
        self._init_swag_storage()

    def _init_swag_storage(self):
        self.swag_mean = None
        self.swag_covariance = None

    def _init_H(self):
        n_params = sum(p.numel() for p in self.model.parameters())
        self.H = torch.zeros(
            n_params, device=self._device, dtype=self._dtype
        )

    def fit(self, 
            train_loader: DataLoader,
            override: bool = True,  # Move to match parent class ordering
            progress_bar: bool = False,  # Move to match parent class ordering 
            optimizer: torch.optim.Optimizer | None = None,
            criterion: nn.Module | None = None,
            epochs: int | None = None,
            start_epoch: int = 0,
            **kwargs  # To catch any other arguments
        ):
        # Extract parameters from kwargs if they were passed that way,
        # otherwise use the explicitly passed ones.
        opt = optimizer if optimizer is not None else kwargs.pop('optimizer', None)
        crit = criterion if criterion is not None else kwargs.pop('criterion', None)
        eps = epochs if epochs is not None else kwargs.pop('epochs', None)

        if opt is not None and crit is not None and eps is not None:
            self.train_swag(
                train_loader=train_loader,
                optimizer=opt,
                criterion=crit,
                epochs=eps,
                start_epoch=start_epoch,
                progress_bar=progress_bar,
                **kwargs # Pass remaining kwargs
            )
        else:
            raise ValueError(
                "SWAGLaplace.fit requires 'optimizer', 'criterion', and 'epochs' "
                "to be provided for SWAG training."
            )
    
    def train_swag(
        self,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        epochs: int,
        start_epoch: int = 0,
        progress_bar: bool = False,
        **kwargs
    ):
        # First, train the model using SWAG
        self.swag.fit(
            train_loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            epochs=epochs,
            start_epoch=start_epoch,
            progress_bar=progress_bar
        )
        
        # Get SWAG statistics
        var, (U, S) = self.swag.get_covariance()
        
        # Store SWAG mean
        self.swag_mean = [mean.clone() for mean in self.swag.mean]
        
        # Store SWAG covariance (diagonal + low-rank)
        self.swag_covariance = {
            'var': var,
            'U': U,
            'S': S
        }
        
        # Compute Laplace approximation using SWAG statistics
        self._compute_laplace_approximation()

    def _compute_laplace_approximation(self):
        # Set the model parameters to SWAG mean
        # self.swag_mean is a list of tensors, aligned with model.parameters()
        with torch.no_grad():
            for model_param, mean_val in zip(self.model.parameters(), self.swag_mean):
                model_param.data.copy_(mean_val)
        
        # After setting model parameters to SWAG mean, update self.mean
        # self.mean is the flattened vector of parameters, used by BaseLaplace/ParametricLaplace
        self.mean = parameters_to_vector(self.model.parameters()).detach()

        # Initialize Hessian (which is a 1D vector for diagonal approximation)
        self._init_H()  # self.H is now torch.zeros(n_params_total, device=self._device, dtype=self._dtype)

    def sample(self, n_samples: int = 100, generator: torch.Generator | None = None) -> torch.Tensor:
        # Ensure the model is fitted, which sets up self.mean and self.H
        if not hasattr(self, 'H') or self.H is None:
            raise RuntimeError("Laplace approximation not fitted. Call 'fit' first to compute H.")
        
        if self.mean is None:
            # This should not happen if fit() was called, as _compute_laplace_approximation sets self.mean.
            raise RuntimeError(
                "Laplace mean (self.mean) is not set. "
                "Ensure 'fit' has been called and that _compute_laplace_approximation "
                "correctly sets self.mean."
            )
        
        return super().sample(n_samples=n_samples, generator=generator)

    # def _add_laplace_correction(self):
    #     """Add Laplace correction to the SWAG sample."""
    #     # Sample from Laplace approximation
    #     laplace_sample = super().sample(n_samples=1)
        
    #     # Add correction to current parameters
    #     for param, correction in zip(self.params, laplace_sample):
    #         param.data += correction

    def evaluate(self, data_loader: DataLoader) -> float:
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in data_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        return 100. * correct / total
