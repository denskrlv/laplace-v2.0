#!/bin/bash

# Build the Docker images
docker compose build

# Start the containers in detached mode
docker compose up -d

exit 0