from __future__ import annotations

import logging
from copy import deepcopy
from typing import Optional, Tuple

from tqdm import tqdm
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from laplace.utils.matrix import Kron, KronDecomposed


class SWAG:
    """Stochastic Weight Averaging-Gaussian (SWAG) implementation.
    
    SWAG builds upon Stochastic Weight Averaging (SWA) by fitting a Gaussian distribution
    to the collected weight samples during training. This provides a way to estimate
    uncertainty in the model's predictions.

    Parameters
    ----------
    model : nn.Module
        The neural network model to apply SWAG to
    n_models : int, default=20
        Number of models to collect for SWAG
    start_epoch : int, default=0
        Epoch to start collecting models
    swa_freq : int, default=1
        Frequency of model collection in epochs
    swa_lr : float, default=0.05
        Learning rate for SWA
    max_num_models : int, default=20
        Maximum number of models to store
    var_clamp : float, default=1e-30
        Minimum value for variance to ensure numerical stability
    """

    def __init__(
        self,
        model: nn.Module,
        n_models: int = 20,
        start_epoch: int = 0,
        swa_freq: int = 1,
        swa_lr: float = 0.05,
        max_num_models: int = 20,
        var_clamp: float = 1e-30,
        device=None,
    ):
        self.model = model
        self.n_models = n_models
        self.start_epoch = start_epoch
        self.swa_freq = swa_freq
        self.swa_lr = swa_lr
        self.max_num_models = max_num_models
        self.var_clamp = var_clamp
        self.device = device if device is not None else next(model.parameters()).device
        self.model = self.model.to(self.device)

        # Initialize storage for model parameters
        self._init_storage()

    def _init_storage(self):
        """Initialize storage for model parameters and statistics."""
        # Get model parameters that require gradients
        self.params = [p for p in self.model.parameters() if p.requires_grad]
        
        # Initialize storage for mean and second moment
        self.mean = [torch.zeros_like(p) for p in self.params]
        self.sq_mean = [torch.zeros_like(p) for p in self.params]
        
        # Initialize storage for collected models
        self.collected_models = []
        self.n_models_collected = 0

    def collect_model(self, epoch: int):
        """Collect a model snapshot if conditions are met.
        
        Parameters
        ----------
        epoch : int
            Current training epoch
        """
        if epoch >= self.start_epoch and epoch % self.swa_freq == 0:
            if self.n_models_collected < self.max_num_models:
                # Create a copy of the current model state
                model_state = deepcopy(self.model.state_dict())
                self.collected_models.append(model_state)
                self.n_models_collected += 1
                
                # Update running statistics
                self._update_statistics()

    def _update_statistics(self):
        """Update running statistics (mean and second moment) of model parameters."""
        for i, param in enumerate(self.params):
            # Update mean
            self.mean[i] = (self.n_models_collected - 1) / self.n_models_collected * self.mean[i] + \
                          1 / self.n_models_collected * param.data
            
            # Update second moment
            self.sq_mean[i] = (self.n_models_collected - 1) / self.n_models_collected * self.sq_mean[i] + \
                             1 / self.n_models_collected * param.data ** 2

    def get_covariance(self, rank: int = 20) -> Tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Compute the covariance matrix using a low-rank plus diagonal approximation.
        
        Parameters
        ----------
        rank : int, default=20
            Rank of the low-rank approximation
            
        Returns
        -------
        tuple
            (diagonal variance, low-rank factors)
        """
        # Compute diagonal variance
        var = [torch.clamp(sq - mean ** 2, min=self.var_clamp) 
               for mean, sq in zip(self.mean, self.sq_mean)]
        
        # Compute low-rank factors
        if rank > 0:
            # Stack all parameter differences
            param_diffs = []
            for model_state in self.collected_models[-rank:]:
                self.model.load_state_dict(model_state)
                diff = [p.data - mean for p, mean in zip(self.params, self.mean)]
                param_diffs.append(torch.cat([d.flatten() for d in diff]))
            
            # Stack differences into a matrix
            D = torch.stack(param_diffs)
            
            # Compute SVD
            U, S, _ = torch.linalg.svd(D, full_matrices=False)
            
            # Return diagonal variance and low-rank factors
            return var, (U, S)
        
        return var, (None, None)

    def sample(self, scale: float = 0.5) -> None:
        """Sample a new set of parameters from the SWAG posterior.
        
        Parameters
        ----------
        scale : float, default=0.5
            Scale factor for the sampled parameters
        """
        var, (U, S) = self.get_covariance()
        
        # Sample from diagonal variance
        for i, param in enumerate(self.params):
            param.data = self.mean[i] + scale * torch.randn_like(param) * torch.sqrt(var[i])
        
        # Add low-rank perturbation if available
        if U is not None and S is not None:
            # Sample from low-rank approximation
            z = torch.randn(S.shape[0], device=S.device)
            perturbation = (U * S.unsqueeze(0)) @ z
            
            # Apply perturbation to parameters
            start_idx = 0
            for param in self.params:
                param_size = param.numel()
                param.data += scale * perturbation[start_idx:start_idx + param_size].view(param.shape)
                start_idx += param_size

    def fit(
        self,
        train_loader: DataLoader,
        optimizer: Optimizer,
        criterion: nn.Module,
        epochs: int,
        start_epoch: int = 0,
        progress_bar: bool = False,
    ):
        """Train the model using SWAG.
        
        Parameters
        ----------
        train_loader : DataLoader
            Training data loader
        optimizer : Optimizer
            Optimizer for training
        criterion : nn.Module
            Loss function
        epochs : int
            Number of training epochs
        start_epoch : int, default=0
            Starting epoch number
        progress_bar : bool, default=False
            Whether to show progress bar
        """
        self.model.train()
        
        # Import tqdm if progress bar is requested
        if progress_bar:
            try:
                from tqdm import tqdm
                epoch_iter = tqdm(range(start_epoch, epochs), desc="Epochs")
            except ImportError:
                print("Warning: tqdm not installed. Progress bar disabled.")
                progress_bar = False
                epoch_iter = range(start_epoch, epochs)
        else:
            epoch_iter = range(start_epoch, epochs)
        
        for epoch in epoch_iter:
            running_loss = 0.0
            
            # Process batches without inner progress bar
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                # Update running loss
                running_loss += loss.item()
                
            # Update epoch progress bar with loss info if using tqdm
            if progress_bar:
                epoch_iter.set_postfix({"loss": f"{running_loss/len(train_loader):.4f}"})
            
            # Collect model if conditions are met
            self.collect_model(epoch)
            
            # Update learning rate for SWA
            if epoch >= self.start_epoch:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = self.swa_lr
            
            # Print epoch summary if not using progress bar
            if not progress_bar:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader):.4f}")

def fit_diagonal_swag_var(model: nn.Module, train_loader: DataLoader, n_epochs: int = 100) -> list[torch.Tensor]:
    """Fit diagonal SWAG variance to a model.
    
    Parameters
    ----------
    model : nn.Module
        The neural network model
    train_loader : DataLoader
        Training data loader
    n_epochs : int, default=100
        Number of training epochs
        
    Returns
    -------
    list[torch.Tensor]
        List of diagonal variances for each parameter
    """
    # Initialize SWAG
    swag = SWAG(
        model=model,
        n_models=20,  # Default number of models
        start_epoch=0,
        swa_freq=1,
        max_num_models=20
    )
    
    # Initialize optimizer and criterion
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # Train using SWAG
    swag.fit(
        train_loader=train_loader,
        optimizer=optimizer,
        criterion=criterion,
        epochs=n_epochs
    )
    
    # Get diagonal variance
    var, _ = swag.get_covariance(rank=0)  # rank=0 to get only diagonal variance
    
    return var

