from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from laplace.baselaplace import BaseLaplace
from laplace.utils.swag import SWAG


class SWAGLaplace(BaseLaplace):
    """Laplace approximation with SWAG (Stochastic Weight Averaging-Gaussian).
    
    This class combines the benefits of SWAG with Laplace approximation to provide
    a more robust uncertainty estimation. It uses SWAG to collect model samples
    during training and then applies Laplace approximation to estimate the posterior.

    Parameters
    ----------
    model : nn.Module
        The neural network model
    likelihood : str
        Likelihood type ('classification' or 'regression')
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
        likelihood: str,
        n_models: int = 20,
        start_epoch: int = 0,
        swa_freq: int = 1,
        swa_lr: float = 0.05,
        max_num_models: int = 20,
        var_clamp: float = 1e-30,
        device = None,
        **kwargs
    ):
        super().__init__(model, likelihood, **kwargs)

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
        """Initialize storage for SWAG statistics."""
        self.swag_mean = None
        self.swag_covariance = None

    def _init_H(self):
        """Initialize Hessian approximation.
        
        For SWAG-Laplace, we use a diagonal Hessian approximation
        which will be updated during model training.
        """
        n_params = sum(p.numel() for p in self.model.parameters())
        self.H = torch.zeros(
            n_params, device=self._device, dtype=self._dtype
        )

    def fit(self, train_loader: DataLoader, *args, **kwargs):
        """Fit the Laplace approximation.
        
        This overrides the BaseLaplace fit method with a compatible signature.
        
        Parameters
        ----------
        train_loader : DataLoader
            Training data loader
        *args, **kwargs : 
            Additional arguments passed to train_swag method
            
        Returns
        -------
        self
        """
        # Extract optimizer and criterion if provided in kwargs
        optimizer = kwargs.pop('optimizer', None)
        criterion = kwargs.pop('criterion', None)
        epochs = kwargs.pop('epochs', 100)
        start_epoch = kwargs.pop('start_epoch', 0)
        progress_bar = kwargs.pop('progress_bar', False)
        
        # If optimizer and criterion are provided, use SWAG training
        if optimizer is not None and criterion is not None:
            self.train_swag(
                train_loader=train_loader,
                optimizer=optimizer,
                criterion=criterion,
                epochs=epochs,
                start_epoch=start_epoch,
                progress_bar=progress_bar,
                **kwargs
            )
        else:
            # Fall back to standard Laplace fit method if no optimizer/criterion provided
            super().fit(train_loader, *args, **kwargs)
        
        return self
    
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
        """Train the model using SWAG and compute Laplace approximation.
        
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
        """Compute Laplace approximation using SWAG statistics."""
        # Set the model parameters to SWAG mean
        for param, mean in zip(self.params, self.swag_mean):
            param.data.copy_(mean)
        
        # Initialize Hessian using SWAG covariance
        self._init_H()
        
        # Update Hessian with SWAG covariance information
        var, (U, S) = self.swag_covariance['var'], (
            self.swag_covariance['U'],
            self.swag_covariance['S']
        )
        
        # Add diagonal variance to Hessian
        for i, param in enumerate(self.params):
            self.H[i] += 1.0 / var[i]
        
        # Add low-rank approximation if available
        if U is not None and S is not None:
            # Compute low-rank update to Hessian
            D = torch.stack([p.flatten() for p in self.params])
            H_update = U @ torch.diag(S) @ U.T
            H_update = H_update.view_as(D)
            
            # Add to Hessian
            for i, param in enumerate(self.params):
                self.H[i] += H_update[i].view(param.shape)

    def sample(self, n_samples: int = 1) -> list[torch.Tensor]:
        """Sample from the SWAG-Laplace posterior.
        
        Parameters
        ----------
        n_samples : int, default=1
            Number of samples to generate
            
        Returns
        -------
        list[torch.Tensor]
            List of sampled parameter sets
        """
        samples = []
        for _ in range(n_samples):
            # Sample from SWAG
            self.swag.sample()
            
            # Add Laplace correction
            self._add_laplace_correction()
            
            # Store sample
            samples.append([p.clone() for p in self.params])
        
        return samples

    def _add_laplace_correction(self):
        """Add Laplace correction to the SWAG sample."""
        # Sample from Laplace approximation
        laplace_sample = super().sample(n_samples=1)
        
        # Add correction to current parameters
        for param, correction in zip(self.params, laplace_sample):
            param.data += correction

    def evaluate(self, data_loader: DataLoader) -> float:
        """Evaluate the model's accuracy on the given data loader."""
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
