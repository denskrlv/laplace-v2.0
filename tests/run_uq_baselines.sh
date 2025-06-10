#!/bin/bash
cd tests
DATA_ROOT="$HOME/projects/laplace-v2.0/data"
#DATA_ROOT="data"
# =========================================================
#  R-MNIST
# =========================================================
for seed in 6 12 13 523 972394; do

  python uq.py --data_root "${DATA_ROOT}" --benchmark R-MNIST --model LeNet \
         --models_root models --method map --model_seed $seed

  python uq.py --data_root "${DATA_ROOT}" --benchmark R-MNIST --model LeNet \
         --models_root models --method ensemble --nr_components 5 --model_seed $seed

  python uq.py --data_root "${DATA_ROOT}" --benchmark R-MNIST \
         --model LeNet-BBB-flipout --models_root models/bbb/flipout \
         --method bbb --model_seed $seed

  python uq.py --data_root "${DATA_ROOT}" --benchmark R-MNIST \
         --model LeNet-CSGHMC --models_root models/csghmc \
         --method csghmc --model_seed $seed

done
#
# =========================================================
#  CIFAR-10-C
# =========================================================
for seed in 6 12 13 523 972394; do

  python uq.py --data_root "${DATA_ROOT}" --benchmark CIFAR-10-C --model WRN16-4 \
         --models_root models --method map --model_seed $seed

  python uq.py --data_root "${DATA_ROOT}" --benchmark CIFAR-10-C --model WRN16-4 \
         --models_root models --method ensemble --nr_components 5 --model_seed $seed

  python uq.py --data_root "${DATA_ROOT}" --benchmark CIFAR-10-C \
         --model WRN16-4-BBB-flipout --models_root models/bbb/flipout \
         --method bbb --model_seed $seed

  python uq.py --data_root "${DATA_ROOT}" --benchmark CIFAR-10-C \
         --model WRN16-4-CSGHMC --models_root models/csghmc \
         --method csghmc --model_seed $seed

done

# =========================================================
#  MNIST OOD DETECTION
# =========================================================
for seed in 6 12 13 523 972394; do

  python uq.py --data_root "${DATA_ROOT}" --benchmark MNIST-OOD --model LeNet \
         --models_root models --method map --model_seed $seed

  python uq.py --data_root "${DATA_ROOT}" --benchmark MNIST-OOD --model LeNet \
         --models_root models --method ensemble --nr_components 5 --model_seed $seed

  python uq.py --data_root "${DATA_ROOT}" --benchmark MNIST-OOD \
         --model LeNet-BBB-flipout --models_root models/bbb/flipout \
         --method bbb --model_seed $seed

  python uq.py --data_root "${DATA_ROOT}" --benchmark MNIST-OOD \
         --model LeNet-CSGHMC --models_root models/csghmc \
         --method csghmc --model_seed $seed

done

# =========================================================
#  CIFAR-10 OOD DETECTION
# =========================================================
for seed in 6 12 13 523 972394; do

  python uq.py --data_root "${DATA_ROOT}" --benchmark CIFAR-10-OOD --model WRN16-4 \
         --models_root models --method map --model_seed $seed

  python uq.py --data_root "${DATA_ROOT}" --benchmark CIFAR-10-OOD --model WRN16-4 \
         --models_root models --method ensemble --nr_components 5 --model_seed $seed

  python uq.py --data_root "${DATA_ROOT}" --benchmark CIFAR-10-OOD \
         --model WRN16-4-BBB-flipout --models_root models/bbb/flipout \
         --method bbb --model_seed $seed

  python uq.py --data_root "${DATA_ROOT}" --benchmark CIFAR-10-OOD \
         --model WRN16-4-CSGHMC --models_root models/csghmc \
         --method csghmc --model_seed $seed

done

# =========================================================
#  SWAG
# =========================================================
python uq.py --data_root "${DATA_ROOT}" --benchmark MNIST-OOD --model LeNet \
       --models_root models --method swag --n_samples 30 --seed 711 --model_seed 6

python uq.py --data_root "${DATA_ROOT}" --benchmark CIFAR-10-OOD --model WRN16-4 \
       --models_root models --method swag --n_samples 30 --seed 711 --model_seed 6
