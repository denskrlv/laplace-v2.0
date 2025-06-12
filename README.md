# laplace-v2.0

## How to Run

1. Make sure you have Docker and Docker Compose installed
2. Run the setup script: `./setup_docker.sh`
3. Download models from [Google Drive](https://drive.google.com/file/d/17cI3dhconEwLj5J3XTEgbPlzYTUoSxAk/view?usp=share_link)
4. Extract them to the `models` directory
5. Build and start the container: `docker-compose up -d`
6. Run experiments:
   - `docker-compose exec laplace ./tests/run_uq_baselines.sh`
   - `docker-compose exec laplace ./tests/run_uq_laplace.sh`
   - `docker-compose exec laplace ./tests/run_uq_subspace.sh`