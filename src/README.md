# Source Code (`src`)

This directory contains the main source code for the Compound Figure Separation project.

## Top-Level Scripts

- **`demo_app.py`**: 
  - The main Streamlit demonstration application.
  - Allows users to select models, run inference on test images or uploaded files, and visualize results with ground truth comparisons.
  - Run with: `streamlit run src/demo_app.py`

- **`prepare.py`**:
  - A one-click setup script to prepare the environment.
  - Downloads the necessary datasets (using `src/utils/download_dataset.py`), updates dataset paths to absolute, and performs integrity checks.
  - Run with: `python src/prepare.py`

- **`train.py`**:
  - A CLI script for training YOLO models.
  - Allows easy reproduction of training runs with customizable parameters (epochs, batch size, image size, dataset).
  - Run with: `python src/train.py --help`

## Directory Structure

- **`notebooks/`**: Contains Jupyter notebooks for exploration, baseline training, experimentation, and result visualization. See `src/notebooks/README.md` for details.
- **`utils/`**: Contains utility scripts for data download/upload, path management, and other helper tools. See `src/utils/README.md` for details.
- **`generators/`**: Contains scripts or tools used for generating synthetic data or processing specific data types (if applicable).
- **`archive/`**: Contains deprecated or older scripts and experiments.