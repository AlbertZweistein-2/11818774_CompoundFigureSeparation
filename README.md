# 11818774_CompoundFigureSeparation

**Topic:** General Scientific Compound Figure Separation  
**Course:** Applied Deep Learning, TU Wien (WS2025)  
**Student:** Tobias Ponesch (11818774)

> **⚠️ CONTENT WARNING** > This dataset contains scientific figures extracted from research papers, which may include biomedical imagery (e.g., organ scans, anatomical diagrams, or surgical photos). Some users may find these images sensitive or disturbing.

---

## Content
1. [Problem Statement](#1-problem-statement)
2. [Relevant Scientific Literature](#2-relevant-scientific-literature)
3. [Scope](#3-scope)
4. [Summary](#4-summary)
5. [Assignment 2: Hacking & Baseline Results](#5-assignment-2-hacking--baseline-results)
6. [Assignment 1: Original Proposal & Deliverables](#6-assignment-1-original-proposal--deliverables)
7. [Future Work](#7-future-work)
8. [Bibliography](#bibliography)

---

## 1 Problem Statement
During my bachelor thesis on extracting metadata from scientific charts in research papers, I encountered the challenge of separating **compound figures** into their individual **sub-figures** or **panels**. Scientific figures often combine multiple charts, illustrations, or images within a single composite figure. To analyze or extract information from specific parts of such figures, it is essential to first **split compound figures into their individual components**.

## 2 Relevant Scientific Literature
Most existing research focuses on **medical images** (X-rays, MRI). The standard dataset is the **ImageCLEF Medical dataset**. Approaches typically rely on domain-specific features or side loss [[1]](#bibliography)[[2]](#bibliography)[[3]](#bibliography) and do not generalize well to **general scientific figures** (charts, illustrations, tables).

Some research has explored general figures [[4]](#bibliography)[[5]](#bibliography), but publicly available implementations (e.g., [Compound Figure Separator](https://github.com/apple2373/figure-separator?tab=readme-ov-file)) often perform poorly on mixed chart types. Currently, there are **no publicly available datasets** dedicated to general scientific figure separation that include bounding box annotations for complex layouts.

## 3 Scope
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

---

## 5 Assignment 2: Hacking & Baseline Results

For Assignment 2, the focus shifted from planning to **Data Engineering**, establishing a **Baseline Model**, and optimizing for hardware constraints.

### 5.1 Data Engineering Strategy
Instead of purely tuning hyperparameters, the core complexity of this assignment was **Data-Centric AI**.
* **Critical Review:** Existing datasets like *MediCaT* were reviewed but discarded due to imprecise bounding boxes found during manual inspection.
* **Multi-Task Curation:** The new dataset (based on *SCI-3000*) was designed for three downstream tasks:
    1.  **Binary Classification:** Compound vs. Atomic figures (~2,500 images labeled).
    2.  **Figure Type Classification:** Categorizing atomic figures (Charts, Illustrations, etc.).
    3.  **Object Detection:** Manual annotation of **700 compound figures** in Label Studio.
* **Pipeline:** A custom pipeline merges these real-world samples with synthetically generated figures to handle class imbalances (e.g., rare "Table" or "Shared Axis" classes).

### 5.2 Baseline Model & Optimization
* **Model:** YOLOv11s (Small)
* **Input Resolution:** 960px (Optimized to resolve small axis text)
* **Error Metric:** mAP50-95 (mean Average Precision)
* **Target:** 0.50
* **Achieved:** **[INSERT FINAL mAP HERE]**

**Optimization Strategy:**
The optimization involved scaling the model architecture (Nano $\to$ Small) and increasing input resolution (640px $\to$ 960px). The resolution increase was critical to resolving fine-grained elements like shared axes and labels, which were lost at lower resolutions.

### 5.3 System Specification & Runtime
Training was performed on a local workstation to leverage high VRAM for larger batch sizes and resolutions.

* **GPU:** NVIDIA GeForce RTX 3090 (24 GB VRAM)
* **CPU:** AMD Ryzen 7 5800X (8 Cores / 16 Threads)
* **RAM:** 32 GB DDR4 3200MHz
* **Environment:** Python 3.12, PyTorch 2.9, CUDA 12.8

**Runtime:**
* Total training time (40 Epochs): **[INSERT TIME, e.g. ~1.5 hours]**
* Inference speed: ~2ms per image (TensorRT/FP16)

### 5.4 Time Tracking (Estimate)
| Task | Time Spent |
|---|---|
| Data review and initial evaluation| ~8h |
| SCI-3000 Data extraction and first evaluation | ~ 10h |
| Dataset Generation (Synthetic) | ~5h |
| Real Compound Figure labeling (Label Studio) | ~20h |
| Pipeline & Splitting Logic | ~8h |
| Model Training & Debugging | ~8h |

---

## 6 Assignment 1: Original Proposal & Deliverables
*(Content submitted for Assignment 1)*

### 6.1 Exact Deliverables for Assignment 1

#### 5.1 References (≥2 papers)
See [Bibliography](#bibliography) section below.

#### 5.2 Topic decision
**General Scientific Compound Figure Separation** (split multi-panel research figures into sub-figures/panels).

#### 5.3 Project type
**Bring your own data** (synthetic + real-world set) with a **YOLO-based baseline** (fine-tuning + evaluation).

#### 5.4 Written summary
- **Idea & approach (short):** Build and release a dataset for **general** (non-medical) compound figures and fine-tune a **YOLO** detector to predict sub-figure bounding boxes; evaluate on real research figures.
- **Dataset (to use/collect):** - **Synthetic set:** programmatically compose charts/plots/diagrams/other images into compound figures; auto-generate bounding boxes.  
  - **Real-world set:** extract figures from PDFs (e.g., via Docling) and **manually** annotate sub-figure boxes.  
  - **Target size:** Still to be determined.

### 6.2 Work-Breakdown Structure (Original Estimate)
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

### 6.3 Plan for now (Original ToDos)
* [x] Gather Datasets to use (Completed in Assignment 2)
    * [x] Find open source dataset of figures from different papers to combine for the synthetical dataset.
    * [x] Find a dataset with real figures and use real compound figures and label them.
* [x] Generate synthetic dataset from selected datasets
* [x] Generate dataset from real compound figures with bounding boxes.

---

## 7 Future Work
Current limitations involve the semantic mapping of separated figures to their caption descriptions. Future implementations could utilize an OCR-based post-processing pipeline or multimodal models like LayoutLM to align sub-figures with their textual context.

## Bibliography
[1] Tianyuan Yao, et al. **Compound Figure Separation of Biomedical Images with Side Loss.** *arXiv preprint arXiv:2107.08650*, 2021. [https://arxiv.org/abs/2107.08650](https://arxiv.org/abs/2107.08650); **GitHub**: https://github.com/hrlblab/ImageSeperation

[2] Tianyuan Yao, et al. **Compound Figure Separation of Biomedical Images: Mining Large Datasets for Self-supervised Learning.** *arXiv preprint arXiv:2208.14357*, 2022. https://arxiv.org/abs/2208.14357;

[3] Pengyuan Li, et al. **Compound image segmentation of published biomedical figures.** Oxford Bioinformatics, Volume 34, Issue 7, April 2018, Pages 1192 - 1199. https://academic.oup.com/bioinformatics/article/34/7/1192/4430539; **Github**: https://github.com/pengyuanli/FigSplit?tab=readme-ov-file

[4] Satoshi Tsutsui, David Crandall **A Data Driven Approach for Compound Figure Separation Using Convolutional Neural Networks** *arXiv preprint arXiv:1703.05105*, 2017. https://arxiv.org/abs/1703.05105; **GitHub**: https://github.com/apple2373/figure-separator?tab=readme-ov-file; Website: https://vision.soic.indiana.edu/figure-separator/

[5] Noah Siegel, et al. **FigureSeer: Parsing Result-Figures in Research Papers** *Springer Nature Link*, 2016. https://link.springer.com/chapter/10.1007/978-3-319-46478-7_41; **GitHub**: https://github.com/allenai/figureseer