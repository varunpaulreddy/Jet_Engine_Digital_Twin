# Jet_Engine_Digital_Twin

Aura: An AI-Powered Flow Predictor
Aura is a proof-of-concept project demonstrating the use of a Graph Neural Network (GNN) as a surrogate model to predict high-fidelity Computational Fluid Dynamics (CFD) results from low-fidelity inputs. The model learns the complex physics of turbulent flow around a T161 low-pressure turbine blade, enabling near-instantaneous prediction of results that would typically require immense supercomputing resources.

The core of this project uses the public DNS dataset from the PyFR MTU-T161 Simulation as the "ground truth" and learns to enhance results from a much faster RANS simulation run in OpenFOAM.

## Technology Stack
Backend: Python 3.10

AI Framework: PyTorch

GNN Library: PyTorch Geometric (PyG)

Data Processing & Visualization: PyVista, NumPy

CFD Solvers: OpenFOAM (for low-fidelity inputs), PyFR (for processing high-fidelity source data)

## Project Setup
### 1. Environment Setup
It is highly recommended to use a Conda environment to manage dependencies.

Bash

# Create and activate the conda environment
conda create -n aura_env python=3.10
conda activate aura_env

# Install Python packages
pip install torch torchvision torchaudio
pip install torch_geometric
pip install pyvista pandas jupyterlab
### 2. Install External Software
This project requires OpenFOAM and PyFR for the data generation pipeline. Please follow their official documentation for installation:

OpenFOAM Installation Guide

PyFR Installation Guide

## Workflow & Usage
The project is broken down into a sequential workflow, from data acquisition to model inference.

### 1. Data Acquisition & Processing
This phase prepares the paired low-fidelity/high-fidelity dataset for training.

Download Source Data: Place the mesh.pyfrm and snapshot-*.pyfrs files from the AWS bucket into a data/raw/ directory.

Generate Ground Truth: Use the pyfr export command to convert the .pyfrs snapshot and mesh into a .vtu file (e.g., ground_truth_train.vtu).

Generate Input Data: Use OpenFOAM and the same mesh to run a low-fidelity RANS simulation. Export the result as a corresponding .vtu file (e.g., input_data_train.vtu).

Create Graph Objects: Run the data processing script to convert the paired .vtu files into PyG graph objects.

Bash

python src/data_processor.py
This will save the processed .pt files into the data/processed/ directory.

### 2. Model Training
Once the data is processed, you can train the GNN model.

Bash

python src/train.py --epochs 100 --learning_rate 0.001
Trained model weights will be saved to the models/ directory. Monitor the output for the training and validation loss to track model performance.

### 3. Evaluation & Inference
To evaluate the model on unseen data or to run inference on a new low-fidelity input:

Evaluate: Run the evaluation script, pointing it to your validation dataset and trained model weights.

Bash

python src/evaluate.py --model_path models/best_model.pt
Inference: To generate a high-fidelity prediction from a new low-fidelity input, use the inference script.

Bash

python src/inference.py --input_file path/to/new_lowfi_data.vtu --output_file path/to/prediction.vtu
This will create a new .vtu file containing the AI-enhanced flow field, which can be visualized in ParaView or PyVista.

## Project Structure
AURA_PROJECT/
├── data/
│   ├── raw/          # Raw .pyfrs, .pyfrm files
│   ├── processed/    # Processed .pt graph objects
│   └── output/       # Generated .vtu predictions
├── models/           # Saved model checkpoints (.pt)
├── notebooks/        # Jupyter notebooks for exploration
├── src/
│   ├── __init__.py
│   ├── data_processor.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   └── inference.py
└── README.md


Aura Project: Dockerized Action Plan
This plan integrates Docker into the entire project lifecycle, ensuring a reproducible and clean development environment. All commands should be run from the root of your project directory.

## Phase 0: Environment Containerization (The New Foundation)
Objective: Build a single, portable Docker image containing all software dependencies for the project.

Tasks:

Create Project Structure: Organize your project directory as follows:

AURA_PROJECT/
├── data/
├── models/
├── notebooks/
├── src/
├── Dockerfile          # You will create this
├── docker-compose.yml  # You will create this
└── environment.yml     # You will create this

Define Python Environment: Create a file named environment.yml. This will tell Conda exactly which packages to install inside your Docker container.

name: aura_env
channels:
  - pytorch
  - pyg
  - conda-forge
dependencies:
  - python=3.10
  - pip
  - pytorch
  - torchvision
  - torchaudio
  - pyg
  - pyvista
  - pandas
  - jupyterlab
  - vtk

Create the Dockerfile: Create the Dockerfile provided in the code block below. This is the blueprint for your environment.

Create the Docker Compose File: Create the docker-compose.yml file provided below. This is your easy-to-use control panel for the container.

Build the Docker Image: This command reads your Dockerfile and builds the environment. This will take a long time (30-60 minutes) the first time you run it.

docker-compose build

Start the Container: This command starts your container in the background.

docker-compose up -d

Verify the Container: Check that your container is running.

docker ps

You should see an entry for aura_project_aura-dev_1.

## Phase 1: Data Acquisition & Generation (Inside Docker)
Objective: Use the tools inside your container to generate the paired dataset.

Tasks:

Enter the Container: Get a bash shell inside your running container.

docker-compose exec aura-dev bash

You are now "inside" your pre-configured lab. All subsequent commands in this plan are run from this shell.

Download Source Files: Use the AWS CLI (already installed in the container) to download the data into the mounted /app/data directory.

aws s3 cp s3://pyfr-mtu-t161-dns-data/mesh.pyfrm ./data/raw/ --no-sign-request
aws s3 cp s3://pyfr-mtu-t161-dns-data/snapshot-247.000.pyfrs ./data/raw/ --no-sign-request
# ... and so on for your validation file

Generate High-Fidelity Target: Use the container's PyFR installation.

pyfr export data/raw/mesh.pyfrm data/raw/snapshot-247.000.pyfrs data/processed/ground_truth_train.vtu

Generate Low-Fidelity Input: Use the container's OpenFOAM installation to run your RANS simulation and generate data/processed/input_data_train.vtu.

## Phase 2 & 3: Model Development & Training (Inside Docker)
Objective: Interactively develop your data processing scripts and train your GNN model.

Tasks:

Start Jupyter Lab: From a new, separate terminal on your host machine (not inside the container), run the following command to start a Jupyter Lab server inside the container.

docker-compose exec aura-dev jupyter-lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root

Access Jupyter: The command will output a URL with a token. Copy and paste it into your local web browser to access the Jupyter interface running inside the container. You can now create notebooks in the /app/notebooks directory to develop your data processing and model-building scripts.

Process Data: Write your graph construction script as a notebook or .py file. Run it from within Jupyter or from the container's bash shell to process the .vtu files and save the graph objects in data/processed/.

Train the Model: Once your scripts are ready, run the training from the container's bash shell. Your GPU will be automatically detected.

python src/train.py

## Phase 4: Validation & Demonstration
Objective: Evaluate the model and generate the final visuals.

Tasks:

Run Evaluation: Use your validation dataset to test the model's performance.

docker-compose exec aura-dev python src/evaluate.py

Run Inference: Generate a prediction from a low-fidelity input.

docker-compose exec aura-dev python src/inference.py --input_file data/processed/input_data_validation.vtu --output_file data/output/prediction.vtu

Visualize: The output file (prediction.vtu) will appear in the data/output folder on your host machine. You can now open it with your local installation of ParaView to create the final comparison visuals.
