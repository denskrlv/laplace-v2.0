from __future__ import annotations

from typing import Any, Literal, MutableMapping, Type
import warnings

import torch
from torch import nn
from torch.autograd import grad
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from torch.utils.data import DataLoader

import tqdm

from laplace.baselaplace import ParametricLaplace
from laplace.curvature.curvature import CurvatureInterface
from laplace.utils.enums import Likelihood
from laplace import FullLaplace

def _flatten(tensor_list):
    """Vectorise a list of tensors, skipping None entries and
    making every slice contiguous."""
    return torch.cat([t.reshape(-1) for t in tensor_list if t is not None])


class SubspaceLaplace(ParametricLaplace):
    """
    Instead of defining a Gaussian posterior over the entire parameter space,
    it performs inference only in a K-dimensional subspace (K << P)
    defined by basis vectors.
    The remaining directions are kept fixed at the MAP estimate.

    Works with:
    - "hessian_eig"
    - "pca_sgd"
    - "random"

    Parameters:
        - model : torch.nn.Module

        - likelihood : Likelihood or str in {'classification', 'regression'}

        - subspace_dim : int, default=20

        - subspace_method : str, default='hessian_eig'

        - n_eig_samples : int, default=100 (for 'hessian_eig')

        - sigma_noise : torch.Tensor or float, default=1 (must be 1 for classification)

        - prior_precision : torch.Tensor or float, default=1 (= weight decay)

        - prior_mean : torch.Tensor or float, default=0

        - temperature : float, default=1

        - enable_backprop: bool, default=False

        - dict_key_x: str, default='input_ids'

        - dict_key_y: str, default='labels'

        - backend : subclasses of `laplace.curvature.CurvatureInterface`

        - backend_kwargs : dict, default=None

    """

    # key to map to correct subclass of BaseLaplace, (subset of weights, Hessian structure)
    _key = ("all", "subspace")

    def __init__(
        self,
        model: nn.Module,
        likelihood: Likelihood | str,
        subspace_dim: int = 20,
        subspace_method: Literal["hessian_eig", "pca_sgd", "random"] = "hessian_eig",
        n_eig_samples: int = 100,
        sigma_noise: float | torch.Tensor = 1.0,
        prior_precision: float | torch.Tensor = 1.0,
        prior_mean: float | torch.Tensor = 0.0,
        temperature: float = 1.0,
        enable_backprop: bool = False,
        dict_key_x: str = "input_ids",
        dict_key_y: str = "labels",
        backend: Type[CurvatureInterface] | None = None,
        backend_kwargs: dict[str, Any] | None = None,
    ):

        self.subspace_dim = subspace_dim
        super().__init__(
            model,
            likelihood,
            sigma_noise=sigma_noise,
            prior_precision=prior_precision,
            prior_mean=prior_mean,
            temperature=temperature,
            enable_backprop=enable_backprop,
            dict_key_x=dict_key_x,
            dict_key_y=dict_key_y,
            backend=backend,
            backend_kwargs=backend_kwargs,
        )

        self.subspace_dim = min(subspace_dim, self.n_params)
        if self.subspace_dim < subspace_dim:
            warnings.warn(
                f"Requested subspace dimension {subspace_dim} exceeds parameter count {self.n_params}. "
                f"Using {self.subspace_dim} instead."
            )

        self.subspace_method = subspace_method
        self.n_eig_samples = n_eig_samples

        # Subspace basis and projections
        self.U = None
        self.H_proj = None
        self._posterior_scale = None

    def _curv_closure(
        self,
        X: torch.Tensor | MutableMapping[str, Any],
        y: torch.Tensor,
        N: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Use the same “full” Hessian/GGN as FullLaplace does
        return self.backend.full(X, y, N=N)

    def _init_H(self) -> None:
        self.H = torch.zeros(
            self.subspace_dim, self.subspace_dim, device=self._device, dtype=self._dtype
        )

    def _init_subspace_basis(self, train_loader: DataLoader) -> None:
        if self.subspace_method == "random":
            # Random orthonormal basis
            U = torch.randn(self.n_params, self.subspace_dim, device=self._device, dtype=self._dtype)
            U, _ = torch.linalg.qr(U)  # Orthonormalize
            self.U = U

        elif self.subspace_method == "hessian_eig":
            # Estimate top eigenvectors of the Hessian using power iteration
            self.U = self._compute_hessian_eigenvectors(train_loader)

        elif self.subspace_method == "pca_sgd":
            # Use principal components from SGD trajectory
            # Note: This would require storing SGD iterates which we don't have access to here
            # For now, we'll fall back to random initialization with a warning
            warnings.warn(
                "SGD trajectory-based PCA subspace not implemented yet. "
                "Falling back to Hessian eigenvectors."
            )
            self.U = self._compute_hessian_eigenvectors(train_loader)

        else:
            raise ValueError(f"Unknown subspace method: {self.subspace_method}")

    def _compute_hessian_eigenvectors(self, train_loader: DataLoader) -> torch.Tensor:
        # Use stochastic power iteration to approximate top eigenvectors
        device = self._device
        dtype = self._dtype
        n_params = self.n_params
        k = self.subspace_dim

        # Initialize random orthonormal basis
        Q = torch.randn(n_params, k, device=device, dtype=dtype)
        Q, _ = torch.linalg.qr(Q)  # Orthonormalize

        # Power iteration steps
        num_steps = self.n_eig_samples
        for _ in range(num_steps):
            # Compute Hessian-vector products for each column of Q
            HQ = torch.zeros_like(Q)

            for i in range(k):
                # Project vector into parameter space
                q = Q[:, i]
                vector_to_parameters(q, self.params)

                # Sample mini-batch
                try:
                    X, y = next(iter(train_loader))
                    X, y = X.to(device), y.to(device)
                except (ValueError, TypeError):
                    # Handle dict-type data for e.g. Huggingface models
                    data = next(iter(train_loader))
                    X, y = data, data[self.dict_key_y].to(device)

                # Compute Hessian-vector product using the backend
                HQ[:, i], _ = self._hessian_vector_product(X, y, q)

                # Reset parameters to MAP
                vector_to_parameters(self.mean, self.params)

            # QR decomposition for orthogonalization
            Q, _ = torch.linalg.qr(HQ)

        # Final multiplication to get better eigenvector estimates
        HQ = torch.zeros_like(Q)
        for i in range(k):
            q = Q[:, i]
            vector_to_parameters(q, self.params)

            try:
                X, y = next(iter(train_loader))
                X, y = X.to(device), y.to(device)
            except (ValueError, TypeError):
                data = next(iter(train_loader))
                X, y = data, data[self.dict_key_y].to(device)

            HQ[:, i], _ = self._hessian_vector_product(X, y, q)
            vector_to_parameters(self.mean, self.params)

        # Perform an eigendecomposition on the small matrix Q^T HQ to extract eigenvalues
        QHQ = torch.matmul(Q.t(), HQ)  # [k, k]
        eigvals, eigvecs = torch.linalg.eigh(QHQ)

        # Sort by descending eigenvalues
        idx = torch.argsort(eigvals, descending=True)
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]

        # Get the final eigenvectors in the original space
        U = torch.matmul(Q, eigvecs)

        # Ensure orthonormality
        U, _ = torch.linalg.qr(U)

        return U

    def _hessian_vector_product(
        self,
        X: torch.Tensor | MutableMapping[str, Any],
        y: torch.Tensor,
        v: torch.Tensor,
    ) -> tuple[torch.Tensor, None]:
        if hasattr(self.backend, "hvp"):
            return self.backend.hvp(X, y, v), None

        # forward + scalar loss
        out = self.model(X)
        loss = self.backend.lossfunc(out, y) * self.backend.factor

        # first-order gradient
        g = grad(loss, self.params, create_graph=True)
        g_flat = _flatten(g)

        # dot with v
        dot = (g_flat * v).sum()

        # second-order gradient
        Hv = grad(dot, self.params, retain_graph=False)
        Hv_flat = _flatten(Hv).detach()

        return Hv_flat, None

    # def fit(self, train_loader, override=True, progress_bar=True):
    #     # Temporarily force full‐space H
    #     print(1)
    #     orig_init_H = type(self)._init_H
    #     type(self)._init_H = FullLaplace._init_H
    #     # Accumulate full Hessian
    #     print(2)
    #     super().fit(train_loader, override=override, progress_bar=progress_bar)
    #     # Restore proper init for subspace
    #     print(3)
    #     type(self)._init_H = orig_init_H
    #     # Build U and project
    #     print(4)
    #     self._init_subspace_basis(train_loader)
    #     H_full = self.H
    #     self.H = self.U.T @ H_full @ self.U
    #     # Reset posterior scale
    #     print(5)
    #     self._posterior_scale = None

    def fit(self, train_loader, override=True, progress_bar=True):
        """
        1.  Find the sub-space basis **before** any curvature is stored.
        2.  Estimate the K×K Hessian inside that sub-space *on the fly*
            with Hessian–vector products; never materialise the full matrix.
        """
        if override:
            self._init_H()                      # empty K×K matrix

        # (1) Sub-space basis
        self._init_subspace_basis(train_loader)  # fills self.U  (P×K)

        # (2) Accumulate projected curvature
        self.model.eval()
        N = len(train_loader.dataset)

        # ------ progress-bar wrapper ---------------------------------
        pbar = tqdm.tqdm(train_loader,
                         disable=not progress_bar,
                         desc="[Subspace LA] accumulating Hessian")
        # --------------------------------------------------------------

        for X, y in pbar:
            X, y = X.to(self._device), y.to(self._device)

            # For every basis vector u_j compute H u_j and
            # inner-products u_iᵀ (H u_j)  →  add to K×K matrix
            for j in range(self.subspace_dim):
                u_j = self.U[:, j]
                H_u_j, _ = self._hessian_vector_product(X, y, u_j)
                proj = self.U.t().mv(H_u_j)        # (K,)
                self.H[:, j] += proj * (train_loader.batch_size / N)
        print('done')
        self._posterior_scale = None

    def _compute_scale(self) -> None:
        # Posterior precision in subspace = H (projected Hessian) + prior_precision
        posterior_precision = self.H + self.prior_precision * torch.eye(
            self.subspace_dim, device=self._device, dtype=self._dtype
        )

        # Compute scale matrix (Cholesky of inverse precision)
        jitter = 1e-6
        for _ in range(5): # try up to 5 times
            try:
                L = torch.linalg.cholesky(posterior_precision + jitter * torch.eye(
                    self.subspace_dim, device=self._device, dtype=self._dtype))
                self._posterior_scale = torch.linalg.inv(L.T)
            except RuntimeError:
                # Handle non-PD matrix (add small diagonal component and retry)
                jitter *= 10
                # posterior_precision += jitter * torch.eye(
                #     self.subspace_dim, device=self._device, dtype=self._dtype
                # )
                # L = torch.linalg.cholesky(posterior_precision)
                # self._posterior_scale = torch.linalg.inv(L.t())

    @property
    def posterior_scale(self) -> torch.Tensor:
        if self._posterior_scale is None:
            self._compute_scale()
        return self._posterior_scale

    @property
    def posterior_covariance(self) -> torch.Tensor:
        scale = self.posterior_scale
        return scale @ scale.t()

    @property
    def posterior_precision(self) -> torch.Tensor:
        if self.H is None:
            raise ValueError("Laplace not fit yet.")
        return self.H + self.prior_precision * torch.eye(
            self.subspace_dim, device=self._device, dtype=self._dtype
        )

    def sample(
        self, n_samples: int = 100, generator: torch.Generator | None = None
    ) -> torch.Tensor:
        # Draw samples in the subspace: z ~ N(0, P_z^-1)
        scale = self.posterior_scale
        z = torch.randn(
            n_samples, self.subspace_dim, generator=generator,
            device=self._device, dtype=self._dtype
        )

        # Project back to parameter space: θ = θ_MAP + U z
        subspace_samples = z @ scale.t()  # [n_samples, subspace_dim]
        param_samples = self.mean.unsqueeze(0) + torch.matmul(subspace_samples, self.U.t())

        return param_samples

    def functional_variance(self, Js: torch.Tensor) -> torch.Tensor:
        self._check_jacobians(Js)

        # Project Jacobians into subspace
        J_proj = torch.matmul(Js, self.U)

        # Covariance in subspace
        subspace_cov = self.posterior_covariance

        # Functional variance
        functional_var = torch.matmul(
            torch.matmul(J_proj, subspace_cov),
            torch.transpose(J_proj, -2, -1)
        )

        return functional_var

    def functional_covariance(self, Js: torch.Tensor) -> torch.Tensor:
        n_batch, n_outs, n_params = Js.shape
        Js = Js.reshape(n_batch * n_outs, n_params)

        # Project Jacobians into subspace
        J_proj = torch.matmul(Js, self.U)  # [batch_size * output_dim, subspace_dim]

        # Covariance
        cov = torch.matmul(
            torch.matmul(J_proj, self.posterior_covariance),
            J_proj.t()
        )

        return cov

    def _check_jacobians(self, Js: torch.Tensor) -> None:
        if not isinstance(Js, torch.Tensor):
            raise ValueError("Jacobians have to be torch.Tensor.")
        if not Js.device == self._device:
            raise ValueError("Jacobians need to be on the same device as Laplace.")


# from laplace.laplace import _LAPLACE_LOOKUP
# _LAPLACE_LOOKUP[('all', 'subspace')] = SubspaceLaplace
