# Notebooks

This directory contains the Jupyter notebooks for the Compound Figure Separation pipeline.

## Run Order

1.  **[01_Extraction_SCI3000.ipynb](01_Extraction_SCI3000.ipynb)**
    *   **Purpose**: Extracts real compound figures and metadata from SCI-3000 source PDFs.
    *   **Input**: SCI-3000 PDF dataset.
    *   **Output**: Extracted figures in `dataset/03_intermediate/SCI-3000_real-compound/`.

2.  **[02_Synthetic_Data_Generation.ipynb](02_Synthetic_Data_Generation.ipynb)** (Optional/Pre-requisite)
    *   **Purpose**: Generates synthetic compound figures by stitching atomic plots.
    *   **Output**: Synthetic images in `dataset/03_intermediate/`.

3.  **[03_Dataset_Assembly.ipynb](03_Dataset_Assembly.ipynb)**
    *   **Purpose**: Merges real and synthetic data, creates standard splits (train/val/test), and converts annotations to YOLO format.
    *   **Input**: Intermediate datasets from steps 01 & 02.
    *   **Output**: Final YOLO-ready dataset in `dataset/04_all_classes/`.

4.  **[04_Train_YOLO_Baseline.ipynb](04_Train_YOLO_Baseline.ipynb)**
    *   **Purpose**: Trains the YOLO object detection model.
    *   **Input**: `dataset/04_all_classes/data.yaml`.
    *   **Output**: Trained weights in `runs/detect/<run_name>/weights/best.pt`.

5.  **[05_YOLO_result_visualization.ipynb](05_YOLO_result_visualization.ipynb)**
    *   **Purpose**: Quantitative and Qualitative Evaluation. Calculates metrics (mAP) and generates visualization plots.
@exe:"docker"
## Note
*   Ensure the `dataset/` directory is populated before running 03.
*   Generated model weights are stored in `runs/` (root) or `models/` (finalized).
