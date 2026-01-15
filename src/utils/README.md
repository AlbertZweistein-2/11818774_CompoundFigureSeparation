# Utility Scripts

This directory contains helper scripts and tools for dataset management, processing, and other utility tasks for the Compound Figure Separation project.

## Core Utilities

### Data Management
- **`download_dataset.py`**: 
  - Downloads the datasets from the Hugging Face repository `TobiPoni/CompoundFigureSeparation`.
  - Supports downloading all data, standard training folders (04, 05, 06), or specific folders.
  - Usage: `python src/utils/download_dataset.py --select default`

- **`download_models.py`**:
  - Downloads trained YOLO models from the Hugging Face repository `TobiPoni/BaseCompoundFigureSeparator`.
  - Usage: `python src/utils/download_models.py all` or `python src/utils/download_models.py <model_name>`

- **`upload_data.py`**:
  - Uploads local datasets to the Hugging Face repository.
  - Handles zipping of large folders to avoid file count limits.

### Path Management
- **`relativizeSplitPaths.py`**: 
  - Updates YOLO dataset configuration files (`data.yaml`) and split text files (`train.txt`, `val.txt`, `test.txt`) to use relative paths. 
  - Essential for making the dataset portable between different machines.

- **`absolutizeSplitPaths.py`**:
  - Converts relative paths in dataset configuration to absolute paths.
  - Useful if a specific tool or environment requires absolute paths.

## Legacy & Helper Tools

- **`imageViewer.py`**: A simple Streamlit app to browse through large folders of images.
- **`SCI3000Extractor.py`**: Tool to extract figures and captions from SCI-3000 PDFs using JSON annotations.
- **`SCI3000Grader.py`**: Streamlit app for grading figures (accepted/compound/questionable).
- **`SCI3000SingleClassificationReview.py`**: Streamlit app for classifying single images.
