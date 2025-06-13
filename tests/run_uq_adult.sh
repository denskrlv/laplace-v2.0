#!/bin/bash

echo "Starting Adult Dataset Experiment Suite"
echo "---------------------------------------"

# === Configuration ===
# Assumes the script is run from the project root (e.g., ./run_uq_adult.sh)
DATA_ROOT="./data"
MODEL_NAME="MLPTabular"
BENCHMARK="Adult"
RESULTS_ROOT="./results/${BENCHMARK}"
MODELS_ROOT="./models/${BENCHMARK}" # We won't be saving models, but good practice

# Ensure results directory exists
mkdir -p $RESULTS_ROOT

COMMON_ARGS="--data_root ${DATA_ROOT} \
             --benchmark ${BENCHMARK} \
             --model ${MODEL_NAME} \
             --models_root ${MODELS_ROOT} \
             --batch_size 256"

# === 1. BASELINE EXPERIMENTS ===
echo ""
echo "--> RUNNING BASELINE MAP EXPERIMENT"
python3 tests/uq.py ${COMMON_ARGS} --method map --run_name "${BENCHMARK}/map_baseline"

echo ""
echo "--> RUNNING BASELINE LAPLACE (LAST-LAYER) EXPERIMENT"
# This corresponds to the default "LA" in the paper
python3 tests/uq.py ${COMMON_ARGS} --method laplace --subset_of_weights last_layer --run_name "${BENCHMARK}/laplace_ll_baseline"


# === 2. DOMAIN SHIFT EXPERIMENTS ===
echo ""
echo "------------------------------------------------"
echo "--> RUNNING DOMAIN SHIFT: MALE-TO-FEMALE"
python3 tests/uq.py ${COMMON_ARGS} --method laplace --subset_of_weights last_layer --domain_shift_gender male_to_female --run_name "${BENCHMARK}/shift_male_to_female"

echo ""
echo "--> RUNNING DOMAIN SHIFT: FEMALE-TO-MALE"
python3 tests/uq.py ${COMMON_ARGS} --method laplace --subset_of_weights last_layer --domain_shift_gender female_to_male --run_name "${BENCHMARK}/shift_female_to_male"


# === 3. NOISE INTENSITY EXPERIMENTS ===
echo ""
echo "------------------------------------------------"
echo "--> RUNNING NOISE INTENSITY EXPERIMENTS"
for intensity in 0.1 0.25 0.5 0.75 1.0; do
    echo ""
    echo "    -> Noise Intensity: ${intensity}"
    python3 tests/uq.py ${COMMON_ARGS} --method laplace --subset_of_weights last_layer --noise_intensity ${intensity} --run_name "${BENCHMARK}/noise_${intensity}"
done

echo ""
echo "---------------------------------------"
echo "All Adult dataset experiments complete."
echo "Results are saved in ${RESULTS_ROOT}"