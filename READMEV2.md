# 11818774_CompoundFigureSeparation

**Topic:** General Scientific Compound Figure Separation  
**Course:** Applied Deep Learning, TU Wien (WS2025)  
**Student:** Tobias Ponesch (11818774)

> **⚠️ CONTENT WARNING** > This dataset contains scientific figures extracted from research papers, which may include biomedical imagery (e.g., organ scans, anatomical diagrams, or surgical photos). Some users may find these images sensitive or disturbing.

---

## Content
- [11818774\_CompoundFigureSeparation](#11818774_compoundfigureseparation)
  - [Content](#content)
  - [1 Problem Statement \& Motivation](#1-problem-statement--motivation)
  - [2 Background \& Definitions](#2-background--definitions)
  - [3 Relevant Scientific Literature](#3-relevant-scientific-literature)
  - [4 Scope](#4-scope)
  - [4 Summary](#4-summary)
  - [5 Demo Application](#5-demo-application)
  - [6 Results](#6-results)
  - [7 Detailed Documentation](#7-detailed-documentation)
    - [7.1 Setup \& Installation](#71-setup--installation)
    - [7.2 Quickstart](#72-quickstart)
    - [7.3 Paths \& Portability](#73-paths--portability)
    - [7.4 Dataset Overview](#74-dataset-overview)
    - [7.5 Pipeline (Raw → Assemble → Train → Eval)](#75-pipeline-raw--assemble--train--eval)
    - [7.6 Tests](#76-tests)
    - [7.7 Data Generation \& Curation](#77-data-generation--curation)
    - [7.8 Assignment 2: Hacking \& Baseline Results](#78-assignment-2-hacking--baseline-results)
      - [7.8.1 Deliverables (Assignment 2)](#781-deliverables-assignment-2)
      - [7.8.2 Data Engineering Strategy](#782-data-engineering-strategy)
      - [7.8.3 Baseline Model \& Optimization](#783-baseline-model--optimization)
      - [7.8.4 System Specification \& Runtime](#784-system-specification--runtime)
      - [7.8.5 Time Tracking (Estimate)](#785-time-tracking-estimate)
    - [7.9 Assignment 1: Original Proposal \& Deliverables](#79-assignment-1-original-proposal--deliverables)
      - [7.9.1 Exact Deliverables for Assignment 1](#791-exact-deliverables-for-assignment-1)
        - [7.9.1.1 References (≥2 papers)](#7911-references-2-papers)
        - [7.9.1.2 Topic decision](#7912-topic-decision)
        - [7.9.1.3 Project type](#7913-project-type)
        - [7.9.1.4 Written summary](#7914-written-summary)
      - [7.9.2 Work-Breakdown Structure (Original Estimate)](#792-work-breakdown-structure-original-estimate)
      - [7.9.3 Plan for now (Original ToDos)](#793-plan-for-now-original-todos)
    - [7.10 Future Work](#710-future-work)
    - [7.11 Hardware Used](#711-hardware-used)
    - [7.12 Bibliography](#712-bibliography)

---

## 1 Problem Statement & Motivation
During my bachelor thesis on extracting metadata from scientific charts in research papers, I encountered the challenge of separating **compound figures** into their individual **sub-figures** or **panels**. Scientific figures often combine multiple charts, illustrations, or images within a single composite figure. To analyze or extract information from specific parts of such figures, it is essential to first **split compound figures into their individual components**.

This project exists because most downstream tasks (chart understanding, table extraction, caption alignment, OCR, figure-type classification) **assume individual figures as input**, while real papers frequently bundle multiple panels together. Without robust separation, automation pipelines break at the first step, and scientific figures remain largely inaccessible for large-scale analytics.

## 2 Background & Definitions
**Compound figure separation** is the task of detecting all sub-figures within a larger composite image and returning bounding boxes for each panel. In practice, panels can be separated by white gaps, sub-caption labels (A, B, C), or shared axes; sometimes there is no clear separator at all.

Common layout patterns include:
- Grid layouts (e.g., 2x2 panels with consistent spacing)
- Mixed-size panels (one large overview + multiple insets)
- Tightly packed charts with shared axes or minimal whitespace
- Overlaid annotations, legends, and sub-figure labels that cross panel boundaries

Why it is hard:
- The same figure type can appear at drastically different scales.
- Visual separators are inconsistent across disciplines and publishers.
- Text elements (axes, legends, panel labels) often overlap multiple panels.
- Models trained on medical imagery do not generalize to general scientific plots or diagrams.

## 3 Relevant Scientific Literature

Most existing research focuses on **medical images** (X-rays, MRI). The standard dataset is the **ImageCLEF Medical dataset**. Approaches typically rely on domain-specific features or side loss [[1]](#bibliography)[[2]](#bibliography)[[3]](#bibliography) and do not generalize well to **general scientific figures** (charts, illustrations, tables).

Some research has explored general figures [[4]](#bibliography)[[5]](#bibliography), but publicly available implementations (e.g., [Compound Figure Separator](https://github.com/apple2373/figure-separator?tab=readme-ov-file)) often perform poorly on mixed chart types. Currently, there are **no publicly available datasets** dedicated to general scientific figure separation that include bounding box annotations for complex layouts.

## 4 Scope
The goal of this project is to create a **dataset and model for general scientific figure separation**.

**Key Objectives:**
1.  **Synthetic Dataset:** Programmatically combine charts/plots (from sources like SciCap or SCI-3000) into composite figures with auto-generated bounding box labels.
2.  **Real-World Dataset:** Extract figures from PDFs (using tools like Docling or SCI-3000) and manually annotate them.
3.  **Model Training:** Fine-tune a YOLO model (YOLOv11) on the combined dataset.

**Outcome:**
* A reproducible **Data Engineering Pipeline**.
* A **hybrid dataset** (Synthetic + Real-World).
* A trained **YOLOv11 baseline model**.

## 4 Summary
This project addresses the lack of a general-purpose solution for compound figure separation. By combining **synthetic data generation** with **manual annotation** of real-world scientific figures, it establishes a robust pipeline for training detection models. The resulting resources (code, dataset, model weights) aim to support further research in automated figure understanding.

**What you can do with this repository:**
- Train or fine-tune YOLO models for compound figure separation.
- Rebuild or customize datasets from the 01/02/03 pipeline assets.
- Run the Streamlit demo for interactive inference and visualization.
- Extend the dataset with additional real-world labeling to boost performance.

---

## 5 Demo Application
The repository includes a Streamlit demo for interactive inference and visualization.

- **Run the app:** `streamlit run src/demo_app.py`
- **What it does:** pick a model, run inference on a test image or your own upload, and compare predictions vs. ground truth.
- **Hugging Face assets:**  
  - Dataset: https://huggingface.co/datasets/TobiPoni/CompoundFigureSeparation  
  - Models: https://huggingface.co/TobiPoni/BaseCompoundFigureSeparator  

## 6 Results
**Baseline metrics (validation set):**

| Run | mAP50-95 | mAP50 | Precision | Recall | Fitness |
|---|---:|---:|---:|---:|---:|
| compound_yolo11n_sim2real | 0.474717 | 0.589498 | 0.561775 | 0.595361 | 0.474717 |
| compound_yolo11s_sim2real | 0.589258 | 0.652078 | 0.667178 | 0.612863 | 0.589258 |
| compound_FINAL_medium_1280 | 0.571545 | 0.618877 | 0.734589 | 0.587923 | 0.571545 |
| compound_FINAL_medium_1280_SMART_FINETUNE | 0.571202 | 0.638409 | 0.743074 | 0.607866 | 0.571202 |
| comparison_small_1280_batch16 | 0.558806 | 0.640242 | 0.689162 | 0.584029 | 0.558806 |
| compound_figure_separator_selected_classes_yol... | 0.888112 | 0.948805 | 0.919226 | 0.937864 | 0.888112 |
| compound_chart_splitter_yolo11n_1280_batch16 | 0.436243 | 0.621764 | 0.648992 | 0.583571 | 0.436243 |
| compound_chart_splitter_yolo11s_1280_batch16 | 0.432764 | 0.587210 | 0.601707 | 0.523441 | 0.432764 |
| compound_chart_splitter_yolo11m_1280_batch8 | 0.436168 | 0.598296 | 0.554434 | 0.576579 | 0.436168 |

Only result metrics are shown here; hyperparameters are intentionally omitted for brevity.

Result plots (PR curves, loss curves, qualitative predictions) will be added from `assets/`.

## 7 Detailed Documentation

### 7.1 Setup & Installation
- **Python:** 3.12 (>=3.10 should work)
- **Env:** `python -m venv .venv && source .venv/bin/activate`
- **Install:** `pip install --upgrade pip` then `pip install -r requirements.txt` (for CUDA wheels follow https://pytorch.org if GPU is available; CPU-only also works for inference/tests).

### 7.2 Quickstart
1) **Download datasets (optional if you clone the repo with data):**  
   `python src/utils/download_dataset.py --repo_id TobiPoni/CompoundFigureSeparation`
2) **Train a baseline model:**  
   `python src/train.py --dataset 04_all_classes --img 960 --epochs 40 --batch 16 --model yolo11s`
3) **Run the demo app:**  
   `streamlit run src/demo_app.py`

### 7.3 Paths & Portability
- Dataset configs are now relative: see [dataset/04_all_classes/data.yaml](dataset/04_all_classes/data.yaml), [dataset/05_selected_classes/data.yaml](dataset/05_selected_classes/data.yaml), and [dataset/06_compound_chart_splitter/data.yaml](dataset/06_compound_chart_splitter/data.yaml).
- To rewrite split lists to relative paths (if you copied data to a new machine): run `python src/utils/make_splits_relative.py` once after syncing the repo (handles 04/05/06 splits).

### 7.4 Dataset Overview
This repository provides three **pre-datasets** (01/02/03) to support custom dataset generation, plus three **final training datasets** (04/05/06) ready for YOLO training.

**Pre-datasets (for re-generation / customization):**
- `dataset/01_raw/`: Raw extracted figures and metadata (e.g., SCI-3000 exports, initial figure candidates, PDFs/JSONs).  
- `dataset/02_assets/`: Curated assets used for synthesis (backgrounds, chart snippets, styles, templates, and layout ingredients).  
- `dataset/03_intermediate/`: Intermediate outputs from the pipeline (real compound figures, Label Studio exports, synthetic stitcher outputs, and temporary splits).

These three folders are intended to let you **rebuild a custom dataset** by changing filters, class mappings, synthesis parameters, or labeling policies.

**Final datasets (ready for training/eval):**
- `dataset/04_all_classes/`: Full label set (all annotated classes).  
- `dataset/05_selected_classes/`: High-level classes only (coarser taxonomy for more robust detection).  
- `dataset/06_compound_chart_splitter/`: Chart-only subset + chart sub-elements (axes, legends, etc.).

**Dataset organization (YOLO format):**
- `images/` and `labels/` follow standard YOLO structure.  
- `train.txt`, `val.txt`, `test.txt` store split lists (relative paths).
- `data.yaml` defines class names and split files.

**Limitations:** The real-world compound set is still relatively small. Performance will likely benefit from additional labeling and broader real-world coverage.

### 7.5 Pipeline (Raw → Assemble → Train → Eval)
1) **Extraction (Real):** Use [src/notebooks/01_Extraction_SCI3000.ipynb](src/notebooks/01_Extraction_SCI3000.ipynb) with SCI-3000 PDFs + JSON annotations → exports figures + metadata into `dataset/03_intermediate/SCI-3000_real-compound/`.
2) **Synthetic Generation:** Use [src/generators/SCI3000SyntheticCompoundStitcher.py](src/generators/SCI3000SyntheticCompoundStitcher.py) and [src/generators/CompoundPlotGenerator.py](src/generators/CompoundPlotGenerator.py) to create synthetic compound figures → outputs in `dataset/03_intermediate/SCI-3000_synthetic-generated/` and `dataset/03_intermediate/SyntheticCompoundPlots/`.
3) **Assembly:** Run [src/notebooks/03_Dataset_Assembly.ipynb](src/notebooks/03_Dataset_Assembly.ipynb) to merge real + synthetic, generate YOLO splits and labels → materialized in `dataset/04_all_classes/` and `dataset/05_selected_classes/` (and optional `dataset/06_compound_chart_splitter/`).
4) **Training:** Use [src/notebooks/04_Train_YOLO_Baseline.ipynb](src/notebooks/04_Train_YOLO_Baseline.ipynb) with `data.yaml` from the chosen dataset → training outputs in `runs/detect/...`.
5) **Evaluation/Visualization:** Use [src/notebooks/05_YOLO_result_visualization.ipynb](src/notebooks/05_YOLO_result_visualization.ipynb) to plot metrics and qualitative examples from `runs/detect/...`.

### 7.6 Tests
- **Dataset integrity:** `pytest tests/test_dataset_integrity.py` checks YOLO label files vs. images. Override dataset root via env: `DATASET_ROOT=dataset pytest tests/test_dataset_integrity.py`.
- **GPU availability:** `pytest tests/test_gpu_availability.py` (skips automatically if keine GPU). Force skip on CPU/CI: `SKIP_GPU_TEST=1 pytest tests/test_gpu_availability.py`.

### 7.7 Data Generation & Curation
This is the end-to-end data story of the project:

1) **Extraction (SCI-3000):** figures were extracted from SCI-3000 PDFs with associated metadata.  
2) **Curation:** ~2,500 figures were manually reviewed for usefulness.  
3) **Real compounds:** ~700 real compound figures were annotated with bounding boxes using Label Studio [[6]](#bibliography).  
4) **Singles:** ~1,160 atomic figures were labeled as single panels.  
5) **Synthetic compounds:** those single panels were stitched into ~10,000 synthetic compound figures (auto-labeled).  
6) **Synthetic plots:** ~2,500 charts were generated via Matplotlib with varying layouts, styles, and noise.  
7) **Mixing & splitting:** real and synthetic data were combined with oversampling, and real compounds were stratified into train/val/test.  
8) **Gap-filling:** underrepresented classes (e.g., tables) were supplemented using synthetic samples (e.g., 50 for val + 50 for test).  
9) **Final sets:** data were packaged into 04/05/06 with class mappings and task-specific class granularity.

Example images (stitched compounds, real compounds, synthetic plots) will be added to this section.

---

### 7.8 Assignment 2: Hacking & Baseline Results

For Assignment 2, the focus shifted from planning to **Data Engineering**, establishing a **Baseline Model**, and optimizing for hardware constraints.

#### 7.8.1 Deliverables (Assignment 2)
- **Error Metric:** mAP50-95 (COCO style), evaluated on the held-out validation split of the assembled YOLO dataset (see dataset/04_all_classes).
- **Target vs. Achieved:** Target mAP50-95 = **0.50**; Achieved = **0.58** with YOLOv11s @ 960px.
- **Hardware/Runtime:** Local RTX 3090, ~1.7h for 40 epochs (see specs below).
- **Time Tracking:** See table in section 7.8.5 (total ≈ 59h).

#### 7.8.2 Data Engineering Strategy
Instead of purely tuning hyperparameters, the core complexity of this assignment was **Data-Centric AI**.
* **Critical Review:** Existing datasets like *MediCaT* were reviewed but discarded due to imprecise bounding boxes found during manual inspection.
* **Multi-Task Curation:** The new dataset (based on *SCI-3000*) was designed for three downstream tasks:
    1.  **Binary Classification:** Compound vs. Atomic figures (~2,500 images labeled).
    2.  **Figure Type Classification:** Categorizing atomic figures (Charts, Illustrations, etc.).
    3.  **Object Detection:** Manual annotation of **700 compound figures** in Label Studio.
* **Pipeline:** A custom pipeline merges these real-world samples with synthetically generated figures to handle class imbalances (e.g., rare "Table" or "Shared Axis" classes).

#### 7.8.3 Baseline Model & Optimization
* **Model:** YOLOv11s (Small)
* **Input Resolution:** 960px (Optimized to resolve small axis text)
* **Error Metric:** mAP50-95 (mean Average Precision)
* **Target:** 0.50
* **Achieved:** **0.58**

**Optimization Strategy:**
The optimization involved scaling the model architecture (Nano $\to$ Small) and increasing input resolution (640px $\to$ 960px). The resolution increase was critical to resolving fine-grained elements like shared axes and labels, which were lost at lower resolutions.

#### 7.8.4 System Specification & Runtime
Training was performed on a local workstation to leverage high VRAM for larger batch sizes and resolutions.

* **GPU:** NVIDIA GeForce RTX 3090 (24 GB VRAM)
* **CPU:** AMD Ryzen 7 5800X (8 Cores / 16 Threads)
* **RAM:** 32 GB DDR4 3200MHz
* **Environment:** Python 3.12, PyTorch 2.9, CUDA 12.8

**Runtime:**
* Total training time (40 Epochs): **~1.7 hours**
* Inference speed: ~2ms per image (TensorRT/FP16)

#### 7.8.5 Time Tracking (Estimate)
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

### 7.9 Assignment 1: Original Proposal & Deliverables
*(Content submitted for Assignment 1)*

#### 7.9.1 Exact Deliverables for Assignment 1

##### 7.9.1.1 References (≥2 papers)
See [Bibliography](#bibliography) section below.

##### 7.9.1.2 Topic decision
**General Scientific Compound Figure Separation** (split multi-panel research figures into sub-figures/panels).

##### 7.9.1.3 Project type
**Bring your own data** (synthetic + real-world set) with a **YOLO-based baseline** (fine-tuning + evaluation).

##### 7.9.1.4 Written summary
- **Idea & approach (short):** Build and release a dataset for **general** (non-medical) compound figures and fine-tune a **YOLO** detector to predict sub-figure bounding boxes; evaluate on real research figures.
- **Dataset (to use/collect):** - **Synthetic set:** programmatically compose charts/plots/diagrams/other images into compound figures; auto-generate bounding boxes.  
  - **Real-world set:** extract figures from PDFs (e.g., via Docling) and **manually** annotate sub-figure boxes.  
  - **Target size:** Still to be determined.

#### 7.9.2 Work-Breakdown Structure (Original Estimate)
| Task | What’s included | Estimate |
|---|---|---|
| Dataset: synthetic generation | composition scripts, layouts, noise/spacing, auto-labels | **5-6 days** |
| Dataset: real extraction | PDF parsing (Docling), filtering, curation | **3-4 days** |
| Dataset: manual labeling | Label the extracted figures | **5-6 days** |
| Model design & setup | Select YOLO variant, configure data pipeline, choose suitable loss function | **1–2 days** |
| Training & fine-tuning | baseline runs, tweaks, checkpoints | **2-4 days** |
| Evaluation | metrics, error analysis on real test set | **2-4 days** |
| Minimal application | demo to run on new images | **1-2 days** |
| Final report | concise write-up (method, data, results, limits) | **1-2 days** |
| Presentation | 6–8 slides with demo screenshots | **1 day** |

#### 7.9.3 Plan for now (Original ToDos)
* [x] Gather Datasets to use (Completed in Assignment 2)
    * [x] Find open source dataset of figures from different papers to combine for the synthetical dataset.
    * [x] Find a dataset with real figures and use real compound figures and label them.
* [x] Generate synthetic dataset from selected datasets
* [x] Generate dataset from real compound figures with bounding boxes.

---

### 7.10 Future Work
Current limitations involve the semantic mapping of separated figures to their caption descriptions. Future implementations could utilize an OCR-based post-processing pipeline or multimodal models like LayoutLM to align sub-figures with their textual context.

### 7.11 Hardware Used
- **GPU:** NVIDIA GeForce RTX 3090 (24 GB VRAM)
- **CPU:** AMD Ryzen 7 5800X (8 Cores / 16 Threads)
- **RAM:** 32 GB DDR4 3200MHz
- **Software:** Python 3.12, PyTorch 2.9, CUDA 12.8

<a id="bibliography"></a>
### 7.12 Bibliography
[1] Tianyuan Yao, et al. **Compound Figure Separation of Biomedical Images with Side Loss.** *arXiv preprint arXiv:2107.08650*, 2021. [https://arxiv.org/abs/2107.08650](https://arxiv.org/abs/2107.08650); **GitHub**: https://github.com/hrlblab/ImageSeperation

[2] Tianyuan Yao, et al. **Compound Figure Separation of Biomedical Images: Mining Large Datasets for Self-supervised Learning.** *arXiv preprint arXiv:2208.14357*, 2022. https://arxiv.org/abs/2208.14357;

[3] Pengyuan Li, et al. **Compound image segmentation of published biomedical figures.** Oxford Bioinformatics, Volume 34, Issue 7, April 2018, Pages 1192 - 1199. https://academic.oup.com/bioinformatics/article/34/7/1192/4430539; **Github**: https://github.com/pengyuanli/FigSplit?tab=readme-ov-file

[4] Satoshi Tsutsui, David Crandall **A Data Driven Approach for Compound Figure Separation Using Convolutional Neural Networks** *arXiv preprint arXiv:1703.05105*, 2017. https://arxiv.org/abs/1703.05105; **GitHub**: https://github.com/apple2373/figure-separator?tab=readme-ov-file; Website: https://vision.soic.indiana.edu/figure-separator/

[5] Noah Siegel, et al. **FigureSeer: Parsing Result-Figures in Research Papers** *Springer Nature Link*, 2016. https://link.springer.com/chapter/10.1007/978-3-319-46478-7_41; **GitHub**: https://github.com/allenai/figureseer

[6] **Label Studio: Data Labeling Tool.** https://labelstud.io/; **GitHub**: https://github.com/HumanSignal/label-studio
