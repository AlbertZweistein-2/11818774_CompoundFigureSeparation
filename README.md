# 11818774_CompoundFigureSeparation

**Topic:** General Scientific Compound Figure Separation  
**Course:** Applied Deep Learning, TU Wien (WS2025)  
**Student:** Tobias Ponesch (11818774)

> **⚠️ CONTENT WARNING**  
> This dataset contains scientific figures extracted from research papers, which may include biomedical imagery (e.g., organ scans, anatomical diagrams, or surgical photos). Some users may find these images sensitive or disturbing.

---

## Content
1. [Problem Statement & Motivation](#1-problem-statement--motivation)
2. [Hugging Face Repositories (Data & Models)](#2-hugging-face-repositories-data--models)
3. [Interactive Demo Application](#3-interactive-demo-application)
4. [Quickstart: Training Preparation](#4-quickstart-training-preparation)
5. [Dataset Organization](#5-dataset-organization)
6. [Project Achievements](#6-project-achievements)
7. [Dataset Generation Methodology](#7-dataset-generation-methodology)
8. [Data Mixing & Splitting Strategy](#8-data-mixing--splitting-strategy)
9. [Detailed Assignment Documentation (Archive)](#9-detailed-assignment-documentation-archive)
    - [Assignment 2 Details](#91-assignment-2-hacking--baseline-results)
    - [Assignment 1 Proposal](#92-assignment-1-original-proposal--deliverables)
    - [Time Tracking](#93-time-tracking-estimate)
10. [Bibliography](#10-bibliography)

---

## 1 Problem Statement & Motivation
During research on extracting metadata from scientific charts in research papers, it became clear that a major bottleneck is the presence of **compound figures**. These are composite images containing multiple sub-figures or panels (charts, illustrations, biomedical scans) within a single frame.

To enable automated analysis or extraction from specific components, it is necessary to first **split these compound figures into their individual parts**. While existing research heavily favors medical imagery, this project focuses on **general scientific figures**, bridging the gap with a dedicated dataset and YOLO-based detection model.

---

## 2 Hugging Face Repositories (Data & Models)

The large-scale data and trained weights are hosted externally on Hugging Face:

- **🤗 Dataset:** [TobiPoni/CompoundFigureSeparation](https://huggingface.co/datasets/TobiPoni/CompoundFigureSeparation)
- **🤗 Models:** [TobiPoni/BaseCompoundFigureSeparator](https://huggingface.co/TobiPoni/BaseCompoundFigureSeparator)

**Local Download Tools:**
- Dataset: `python src/utils/download_dataset.py --select default`
- Models: `python src/utils/download_models.py all`

---

## 3 Interactive Demo Application

A Streamlit-based demo application is included to run inference on custom images or the test set. It features automatic model fetching from Hugging Face and side-by-side ground truth comparison.

**Run command:**
```bash
streamlit run src/demo_app.py
```

---

## 4 Quickstart: Training Preparation

Follow these steps to quickly set up the environment and start training:

1.  **Environment Setup:**
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    ```

2.  **One-Click Prepare:**  
    Downloads the training data, fixes relative paths to absolute, and checks integrity.
    ```bash
    python src/prepare.py
    ```

3.  **Start Training:**
    ```bash
    python src/train.py --data 05_selected_classes --epochs 50 --batch 16
    ```

---

## 5 Dataset Organization

The data is materialized into three specialized variants for different research needs:

- **`04_all_classes`**: The complete set of labeled classes (Chart, Illustration, Image, Table, Text, Subpanel, Legend, Titles, Axes).
- **`05_selected_classes`**: A high-level subset optimized for parsing multi-panel layouts (Chart, Illustration, Image, Table). **Recommended as a general-purpose detector.**
- **`06_compound_chart_splitter`**: A specialized dataset containing only charts and their internal sub-elements (X-Axis, Y-Axis, Legend, Title) for deep analysis of chart components.

---

## 6 Project Achievements

During this project, we:
1.  Extracted and curated a hybrid (real+synthetic) dataset for scientific figure detection.
2.  Implemented a robust data engineering pipeline for automated generation and assembly.
3.  Trained a suite of YOLOv11 models (Nano to Medium) achieving an mAP50-95 of 0.58 on general classes.
4.  Created a portable ecosystem with easy data/model downloads and an interactive demo app.

---

## 7 Dataset Generation Methodology

The dataset was constructed through a rigorous process of extraction, manual grading, and synthetic augmentation.

### 7.1 Extraction and Manual Grading
*   **Real Figure Extraction:** Figures were extracted from the *SCI-3000* dataset using PDF parsing tools.
*   **Initial Curation:** Approximately **2,500** figures were manually reviewed and labeled as "useful".
*   **Type Separation:** From this pool, **1,160** images were identified as singles (atomic) and approx. **700** were identified as real compound figures.

### 7.2 Annotation
*   **Real Compounds:** The 700 real compound figures were manually annotated with bounding boxes using **Label Studio** [[6]](#bibliography).
*   **Synthetic Plots:** **2,500** synthetic charts were generated using Matplotlib with varying layouts, noise, and styles to provide ground truth for chart internal elements.

### 7.3 Synthetic Assembly
*   **Stitched Figures:** The 1,160 real single figures were programmatically stitched into **~10,000 synthetic compound figures**, providing a massive increase in training variety with automatic labels.

---

## 8 Data Mixing & Splitting Strategy

The final training data utilizes a sophisticated mixing strategy to ensure high performance on rare classes and robust validation.

### 8.1 Oversampling & Stratified Splitting
*   **Real Data Focus:** The real compound figures (annotated in Label Studio) were split into Train, Val, and Test sets using **stratified splitting** based on the presence of sub-figure types.
*   **Training Augmentation:** The training set was supplemented with oversampled synthetic data to handle class imbalances.

### 8.2 Synthetic Injection for Rare Classes
*   **Class Balancing:** Classes that were under-represented in the real compound set (e.g., **Tables**) were boosted using synthetic samples.
*   **Validation Integrity:** Specifically for **Tables**, 50 synthetic samples were injected into each of the Validation and Test sets to ensure the model's ability to generalize to these types could be measured reliably.

---

## 9 Detailed Assignment Documentation (Archive)

### 9.1 Assignment 2: Hacking & Baseline Results

For Assignment 2, the focus was on **Data Engineering** and establishing a **Baseline Model**.
- **Metric:** mAP50-95 (Target 0.50, Achieved 0.58).
- **Optimization:** Scaling from Nano $\to$ Small and increasing resolution from 640px $\to$ 960px was critical.
- **Hardware:** Training on RTX 3090, ~1.7h for 40 epochs.

### 9.2 Assignment 1: Original Proposal & Deliverables

#### deliverables
- Topic: General Scientific Compound Figure Separation.
- Project type: Bring your own data (synthetic + real-world).
- Idea: Build and release a dataset for **general** compound figures and fine-tune a YOLO detector.

#### Original Work-Breakdown Structure
| Task | What’s included | Estimate |
|---|---|---|
| Dataset: synthetic generation | composition scripts, layouts, noise/spacing | **5-6 days** |
| Dataset: real extraction | PDF parsing (Docling), filtering, curation | **3-4 days** |
| Dataset: manual labeling | Label the extracted figures | **5-6 days** |
| Model design & setup | Select YOLO variant, configure data pipeline | **1–2 days** |
| Training & fine-tuning | baseline runs, tweaks, checkpoints | **2-4 days** |
| Evaluation | metrics, error analysis on real test set | **2-4 days** |
| Minimal application | demo to run on new images | **1-2 days** |
| Final report | concise write-up | **1-2 days** |
| Presentation | 6–8 slides with demo screenshots | **1 day** |

### 9.3 Time Tracking (Estimate)
| Task | Time Spent |
|---|---|
| Data review and initial evaluation| ~8h |
| SCI-3000 Data extraction and first evaluation | ~ 10h |
| Dataset Generation (Synthetic) | ~5h |
| Real Compound Figure labeling (Label Studio) | ~20h |
| Pipeline & Splitting Logic | ~8h |
| Model Training & Debugging | ~8h |
| Repository Cleanup and Documentation | ~Xh |
| Final Dataset preparation and release | ~Xh |
| **Total** | **~XXh** |

---

## 10 Bibliography
[1] Tianyuan Yao, et al. **Compound Figure Separation of Biomedical Images with Side Loss.** *arXiv preprint arXiv:2107.08650*, 2021.

[2] Tianyuan Yao, et al. **Compound Figure Separation of Biomedical Images: Mining Large Datasets for Self-supervised Learning.** *arXiv preprint arXiv:2208.14357*, 2022.

[3] Pengyuan Li, et al. **Compound image segmentation of published biomedical figures.** Oxford Bioinformatics, 2018.

[4] Satoshi Tsutsui, David Crandall **A Data Driven Approach for Compound Figure Separation Using Convolutional Neural Networks** *arXiv preprint arXiv:1703.05105*, 2017.

[5] Noah Siegel, et al. **FigureSeer: Parsing Result-Figures in Research Papers** *Springer Nature Link*, 2016.

[6] Tkachenko, M., Malyuk, M., Holonychev, A., Liubimov, N., & Shlyapnikov, N. (2020-2022). **Label Studio: Data labeling software.** https://github.com/heartexlabs/label-studio
