#!/bin/bash

# This script runs a single instance of each main experiment to verify the setup.
# It uses a single seed (6) for consistency. If any command fails, the
# script will exit immediately.

set -e

# Define a function for consistent logging
log_header() {
    echo "========================================================="
    echo " $1"
    echo "========================================================="
}

# --- Define correct paths relative to the project root ---
MODELS_ROOT="./tests/models"
DATA_ROOT="./data"
CONFIGS_ROOT="./tests/configs"


# --- MNIST Benchmark Test Runs (R-MNIST) ---
log_header "Testing R-MNIST Benchmark"
MODEL_SEED=6
BENCHMARK="R-MNIST"

# MAP
log_header "[${BENCHMARK}] Running: MAP Baseline"
python tests/uq.py --benchmark ${BENCHMARK} --method map --model LeNet \
    --model_seed ${MODEL_SEED} --data_root ${DATA_ROOT} --models_root ${MODELS_ROOT} --download

# Laplace (Default)
log_header "[${BENCHMARK}] Running: Default Laplace (LA)"
python tests/uq.py --benchmark ${BENCHMARK} --config ${CONFIGS_ROOT}/post_hoc_laplace/mnist_default.yaml \
    --model_seed ${MODEL_SEED} --data_root ${DATA_ROOT} --models_root ${MODELS_ROOT} --download

# Laplace (OOD-Optimized LA*)
log_header "[${BENCHMARK}] Running: OOD-Optimized Laplace (LA*)"
python tests/uq.py --benchmark ${BENCHMARK} --config ${CONFIGS_ROOT}/post_hoc_laplace/mnist_bestood.yaml \
    --model_seed ${MODEL_SEED} --data_root ${DATA_ROOT} --models_root ${MODELS_ROOT} --download


# --- CIFAR-10 Benchmark Test Runs (CIFAR-10-C) ---
log_header "Testing CIFAR-10-C Benchmark"
BENCHMARK="CIFAR-10-C"
MODEL="WRN16-4"

# MAP
log_header "[${BENCHMARK}] Running: MAP Baseline"
python tests/uq.py --benchmark ${BENCHMARK} --method map --model ${MODEL} \
    --model_seed ${MODEL_SEED} --data_root ${DATA_ROOT} --models_root ${MODELS_ROOT} --download

# Laplace (Default)
log_header "[${BENCHMARK}] Running: Default Laplace (LA)"
python tests/uq.py --benchmark ${BENCHMARK} --config ${CONFIGS_ROOT}/post_hoc_laplace/cifar10_default.yaml \
    --model_seed ${MODEL_SEED} --data_root ${DATA_ROOT} --models_root ${MODELS_ROOT} --download

# Laplace (OOD-Optimized LA*)
log_header "[${BENCHMARK}] Running: OOD-Optimized Laplace (LA*)"
python tests/uq.py --benchmark ${BENCHMARK} --config ${CONFIGS_ROOT}/post_hoc_laplace/cifar10_bestood.yaml \
    --model_seed ${MODEL_SEED} --data_root ${DATA_ROOT} --models_root ${MODELS_ROOT} --download


log_header "All test runs completed successfully!"
