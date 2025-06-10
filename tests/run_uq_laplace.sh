#!/bin/bash
cd tests

DATA_ROOT="$HOME/projects/laplace-v2.0/data"
#DATA_ROOT="data"

# =========================================================
#  R-MNIST
# =========================================================
#for seed in 6 12 13 523 972394; do
##  python uq.py --data_root "${DATA_ROOT}" --benchmark R-MNIST --model LeNet \
##         --models_root models --method laplace --model_seed $seed --subset_of_weights all
#
#  python uq.py --data_root "${DATA_ROOT}" --benchmark R-MNIST --model LeNet \
#         --models_root models --method laplace --model_seed $seed --subset_of_weights last_layer
#done

# =========================================================
#  CIFAR-10-C
# =========================================================
for seed in 6 12 13 523 972394; do
#  python uq.py --data_root "${DATA_ROOT}" --benchmark CIFAR-10-C --model WRN16-4 \
#         --models_root models --method laplace --model_seed $seed --subset_of_weights all
  python uq.py --data_root "${DATA_ROOT}" --benchmark CIFAR-10-C --model WRN16-4 \
         --models_root models --method laplace --model_seed $seed --subset_of_weights last_layer
done

# =========================================================
#  MNIST OOD DETECTION
# =========================================================
for seed in 6 12 13 523 972394; do
#  python uq.py --data_root "${DATA_ROOT}" --benchmark MNIST-OOD --model LeNet \
#         --models_root models --method laplace --model_seed $seed --subset_of_weights all
  python uq.py --data_root "${DATA_ROOT}" --benchmark MNIST-OOD --model LeNet \
         --models_root models --method laplace --model_seed $seed  --subset_of_weights last_layer
done

# =========================================================
#  CIFAR-10 OOD DETECTION
# =========================================================
for seed in 6 12 13 523 972394; do
#  python uq.py --data_root "${DATA_ROOT}" --benchmark CIFAR-10-OOD --model WRN16-4 \
#         --models_root models --method laplace --model_seed $seed --subset_of_weights all
  python uq.py --data_root "${DATA_ROOT}" --benchmark CIFAR-10-OOD --model WRN16-4 \
         --models_root models --method laplace --model_seed $seed --subset_of_weights last_layer
done
