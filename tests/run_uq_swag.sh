#!/bin/bash
cd tests
SUBSPACE_DIM=20
SUBSPACE_METHOD=random
EIG_STEPS=100
SUBSPACE_ARGS="--method swag_laplace \
               --subspace_dim ${SUBSPACE_DIM} \
               --subspace_method ${SUBSPACE_METHOD} \
               --eig_steps ${EIG_STEPS}
               --batch_size 128"

DATA_ROOT="$HOME/projects/laplace-v2.0/data" #vm data
#DATA_ROOT="data"  #local data

#for seed in 6 12 13 523 972394; do
#  python uq.py --data_root "$DATA_ROOT" \
#          --benchmark MNIST-OOD --model LeNet \
#          --models_root models ${SUBSPACE_ARGS} --model_seed "$seed"
#done

# # =========================================================
for seed in 6 12 13 523 972394; do
  python uq.py --data_root "$DATA_ROOT" \
          --benchmark CIFAR-10-C --model WRN16-4 \
          --models_root models ${SUBSPACE_ARGS} --model_seed "$seed"
done