#!/bin/bash
cd tests

# =========================================================
#  R-MNIST
# =========================================================
for seed in 6 12 13 523 972394; do
  python uq.py --data_root ~/Datasets --benchmark R-MNIST --model LeNet \
         --models_root models --method laplace --model_seed $seed --subset_of_weights all --backend kazuki

#   python uq.py --data_root ~/Datasets --benchmark R-MNIST --model LeNet \
#          --models_root models --method laplace --model_seed $seed --subset_of_weights last_layer
done

# =========================================================
#  CIFAR-10-C
# =========================================================
for seed in 6 12 13 523 972394; do
  python uq.py --data_root ~/Datasets/alt --benchmark CIFAR-10-C --model WRN16-4 \
         --models_root models --method laplace --model_seed $seed --subset_of_weights all --backend kazuki

#   python uq.py --data_root ~/Datasets/alt --benchmark CIFAR-10-C --model WRN16-4 \
#          --models_root models --method laplace --model_seed $seed --subset_of_weights last_layer
done

# =========================================================
#  MNIST OOD DETECTION
# =========================================================
for seed in 6 12 13 523 972394; do
  python uq.py --data_root ~/Datasets --benchmark MNIST-OOD --model LeNet \
         --models_root models --method laplace --model_seed $seed --subset_of_weights all --backend kazuki

#   python uq.py --data_root ~/Datasets --benchmark MNIST-OOD --model LeNet \
#          --models_root models --method laplace --model_seed $seed  --subset_of_weights last_layer
done

# =========================================================
#  CIFAR-10 OOD DETECTION
# =========================================================
for seed in 6 12 13 523 972394; do

  python uq.py --data_root ~/Datasets --benchmark CIFAR-10-OOD --model WRN16-4 \
         --models_root models --method laplace --model_seed $seed --subset_of_weights all --backend kazuki

#   python uq.py --data_root ~/Datasets --benchmark CIFAR-10-OOD --model WRN16-4 \
#          --models_root models --method laplace --model_seed $seed --subset_of_weights last_layer
done
