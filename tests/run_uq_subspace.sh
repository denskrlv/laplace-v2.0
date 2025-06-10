#!/bin/bash
cd tests
SUBSPACE_DIM=20
SUBSPACE_METHOD=random
EIG_STEPS=100
SUBSPACE_ARGS="--method subspace \
               --subspace_dim ${SUBSPACE_DIM} \
               --subspace_method ${SUBSPACE_METHOD} \
               --eig_steps ${EIG_STEPS}"
SEED=6
DATA=~/Datasets

# python uq.py --data_root "$DATA" \
#         --benchmark R-MNIST --model LeNet \
#         --models_root models ${SUBSPACE_ARGS} --model_seed "$SEED"


python uq.py --data_root "$DATA" \
        --benchmark MNIST-OOD --model LeNet \
        --models_root models ${SUBSPACE_ARGS} --model_seed "$SEED"

