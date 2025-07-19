# ==============================================================================
# FILE: Dockerfile
# DESCRIPTION: Blueprint for building the project's Docker image.
# ==============================================================================
# Use the official NVIDIA CUDA image as a base for GPU support
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# Set environment variables to avoid interactive prompts during installation
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies: build tools, OpenFOAM, and PyVista requirements
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    unzip \
    xvfb \
    libgl1-mesa-glx \
    libxrender1 \
    # Add OpenFOAM repository and install
    && curl -s https://dl.openfoam.com/add-debian-repo.sh | bash \
    && apt-get install -y openfoam2206-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
ENV CONDA_DIR /opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p $CONDA_DIR && \
    rm ~/miniconda.sh
ENV PATH=$CONDA_DIR/bin:$PATH

# Create the project directory
WORKDIR /app

# Copy the environment file and create the conda environment
COPY environment.yml .
RUN conda env create -f environment.yml

# Make conda environment available in shell
RUN echo "conda activate aura_env" >> ~/.bashrc
ENV PATH /opt/conda/envs/aura_env/bin:$PATH

# Copy the rest of the project files into the workdir
COPY . .

# Set the default command to keep the container running
CMD [ "tail", "-f", "/dev/null" ]
