# Use a PyTorch base image with CUDA support
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    python3 \
    python3-pip \
    python3-dev \
    vim \
    && rm -rf /var/lib/apt/lists/* \

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip3 install --upgrade pip
RUN pip3 install torch
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy the project files
COPY . .

# Install the package in development mode
RUN pip3 install -e .

# Set up directory for models
RUN mkdir -p tests/models

# Default command: bash
CMD ["/bin/bash"]