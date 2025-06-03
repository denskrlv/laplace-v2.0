import torch, sys
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.append("..")

from laplace.subspace_laplace import SubspaceLaplace
from laplace.utils.enums import Likelihood
from laplace.curvature.asdfghjkl import AsdfghjklHessian

def toy_loader():
    # 100 points, 2-D input, binary labels
    X = torch.randn(100, 2)
    y = (X.sum(dim=1) > 0).long()
    dataset = TensorDataset(X, y)
    return DataLoader(dataset, batch_size=10, shuffle=True)

def test_subspace_laplace_shapes(toy_loader):
    model = nn.Sequential(nn.Linear(2, 4), nn.ReLU(), nn.Linear(4, 2))
    # ensure grad on all params
    for p in model.parameters(): p.requires_grad_(True)

    la = SubspaceLaplace(
        model,
        likelihood=Likelihood.CLASSIFICATION,
        subspace_dim=3,
        subspace_method="hessian_eig",
        n_eig_samples=5,
        backend=AsdfghjklHessian,
    )
    # fit the Laplace in the subspace
    la.fit(toy_loader)

    # posterior precision / covariance live in 3×3
    P = la.posterior_precision
    C = la.posterior_covariance
    assert P.shape == (3, 3)
    assert C.shape == (3, 3)

    # sampling returns full-dim param draws
    samples = la.sample(n_samples=7)
    assert samples.shape == (7, la.n_params)

    # build a random batch and compute functional variance/covariance
    Xb, yb = next(iter(toy_loader))
    # get Jacobians via the backend
    Js, _ = la.backend.jacobians(Xb)
    fv = la.functional_variance(Js)
    fc = la.functional_covariance(Js)
    # should be [batch, out, out] and [batch*out, batch*out]
    assert fv.shape[0] == Xb.size(0)
    assert fv.shape[1] == la.n_outputs
    assert fc.shape == (Xb.size(0)*la.n_outputs, Xb.size(0)*la.n_outputs)


if __name__ == "__main__":
    loader = toy_loader()
    try:
        test_subspace_laplace_shapes(loader)
        print("✅  All assertions passed.")
    except AssertionError as e:
        print("❌  Assertion failed:", e)
        sys.exit(1)

