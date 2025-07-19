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
