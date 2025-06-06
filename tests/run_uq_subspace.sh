#!/bin/zsh
cd tests

SUBSPACE_DIM=20
SUBSPACE_METHOD=random
EIG_STEPS=100
SUBSPACE_ARGS="--method subspace \
               --subspace_dim ${SUBSPACE_DIM} \
               --subspace_method ${SUBSPACE_METHOD} \
               --eig_steps ${EIG_STEPS}"
seed=42

# ---- Subspace Laplace ----
python uq.py --data_root ~/Datasets --benchmark R-MNIST --model LeNet \
            --models_root models $SUBSPACE_ARGS --model_seed $seed

echo 'Running OOD detection ...'

python uq.py --data_root ~/Datasets --benchmark MNIST-OOD --model LeNet \
        --models_root models $SUBSPACE_ARGS --model_seed $seed

