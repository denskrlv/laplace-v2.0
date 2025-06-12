#!/bin/bash

# Create necessary directories
mkdir -p data
mkdir -p tests/models

echo "Please download the models from https://drive.google.com/file/d/17cI3dhconEwLj5J3XTEgbPlzYTUoSxAk/view?usp=share_link"
echo "and extract them to the 'models' directory."
echo "Then run: docker-compose up -d"
echo "To execute a test script: docker-compose exec laplace ./tests/run_uq_baselines.sh"