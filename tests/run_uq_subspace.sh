#!/bin/bash
cd tests
SUBSPACE_DIM=20
SUBSPACE_METHOD=random
EIG_STEPS=100

# Arguments for MNIST (can use 'glm')
MNIST_ARGS="--method subspace \
            --subspace_dim ${SUBSPACE_DIM} \
            --subspace_method ${SUBSPACE_METHOD} \
            --eig_steps ${EIG_STEPS} \
            --batch_size 128"

# Arguments for CIFAR-10-C (use 'nn' to avoid memory error)
CIFAR_ARGS="--method subspace \
            --pred_type nn \
            --subspace_dim ${SUBSPACE_DIM} \
            --subspace_method ${SUBSPACE_METHOD} \
            --eig_steps ${EIG_STEPS} \
            --batch_size 32" # Keep a small batch size for CIFAR-10

DATA_ROOT="$HOME/projects/laplace-v2.0/data"

# This part should still work fine
#echo "Running Subspace Laplace on MNIST-OOD..."
#for seed in 6 12 13 523 972394; do
#  python uq.py --data_root "$DATA_ROOT" \
#          --benchmark MNIST-OOD --model LeNet \
#          --models_root models ${MNIST_ARGS} --model_seed "$seed"
#done

# This is the part that was failing
echo "Running Subspace Laplace on CIFAR-10-C..."
for seed in 6 12 13 523 972394; do
  python uq.py --data_root "$DATA_ROOT" \
          --benchmark CIFAR-10-C --model WRN16-4 \
          --models_root models ${CIFAR_ARGS} --model_seed "$seed"
done