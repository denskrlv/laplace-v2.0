#!/bin/bash
cd tests
SUBSPACE_DIM=20
SUBSPACE_METHOD=random
EIG_STEPS=100
SUBSPACE_ARGS="--method subspace \
               --subspace_dim ${SUBSPACE_DIM} \
               --subspace_method ${SUBSPACE_METHOD} \
               --eig_steps ${EIG_STEPS}"

for seed in 6 12 13 523 972394; do
        python uq.py --data_root ~/Datasets \
        --benchmark MNIST-OOD --model LeNet \
        --models_root models ${SUBSPACE_ARGS} --model_seed $seed
done

for seed in 6 12 13 523 972394; do
        python uq.py --data_root ~/Datasets \
        --benchmark R-MINST --model LeNet \
        --models_root models ${SUBSPACE_ARGS} --model_seed $seed
done

# for seed in 6 12 13 523 972394; do
#         python uq.py --data_root ~/Datasets \
#         --benchmark CIFAR-10-OOD --model WRN16-4 \
#         --models_root models ${SUBSPACE_ARGS} --model_seed $seed
# done