import torch
from laplace import Laplace
from laplace.utils.enums import HessianStructure, SubsetOfWeights

# Load model and train_loader
# ...

# Apply Subspace Laplace approximation
la = Laplace(
    model=model,
    likelihood='classification',
    subset_of_weights=SubsetOfWeights.ALL,
    hessian_structure=HessianStructure.SUBSPACE,
    subspace_dim=20,
    subspace_method='hessian_eig'
)

# Fit the approximation
la.fit(train_loader)

# Make predictions with uncertainty
pred_mean, pred_var = la(test_data, pred_type='glm')