# Compound Figure Separation

**Topic:** General Scientific Compound Figure Separation  
**Course:** Applied Deep Learning, TU Wien (WS2025)  
**Student:** Tobias Ponesch (11818774)

> **⚠️ CONTENT WARNING**  
> This dataset contains scientific figures extracted from research papers, which may include biomedical imagery (e.g., organ scans, anatomical diagrams, or surgical photos). Some users may find these images sensitive or disturbing.

---

## Content
[1 Problem Statement & Motivation](#1-problem-statement--motivation)<br>
[2 Data- & Model- Download (Huggingface)](#2-data--model--download-huggingface) <br>
[3 Interactive Demo Application](#3-interactive-demo-application) <br>
[4 Quickstart: Training Preparation](#4-quickstart-training-preparation) <br>
<span style="margin-left: 20px;"></span>
[4.1 Environment Setup](#41-environment-setup) <br>
<span style="margin-left: 20px;"></span>
[4.2 Start Training](#42-start-training) <br>
[5 Repository Structure](#5-repository-structure) <br>
<span style="margin-left: 20px;"></span>
[5.1 Dataset Overview](#51-dataset-overview) <br>
<span style="margin-left: 20px;"></span>
[5.2 Folder Structure](#52-folder-structure) <br>
[6 Baseline Models]() <br>
<span style="margin-left: 20px;"></span>
[6.1 YOLOv11 and Configurations]() <br>
<span style="margin-left: 20px;"></span>
[6.2 Model Performance]() <br>
[7 Learnings]() <br>
[8 Future Work]() <br>
[9 Detailed Documentation]() <br>
<span style="margin-left: 20px;"></span>
[9.1 Relevant Scientific Literature]() <br>
<span style="margin-left: 20px;"></span>
[9.2 Explored existing Datasets]() <br>
<span style="margin-left: 20px;"></span>
[9.3 Detailed Dataset Generation Strategy]() <br>
<span style="margin-left: 40px;"></span>
[9.3.1 SCI-3000 Figure Extraction]() <br>
<span style="margin-left: 40px;"></span>
[9.3.2 Real Compound Figure Labeling (Labelstudio)]() <br>
<span style="margin-left: 40px;"></span>
[9.3.3 Synthetic Compound Figure Stitching]() <br>
<span style="margin-left: 40px;"></span>
[9.3.4 Synthetic Plot Generation (matplotlib)]() <br>
<span style="margin-left: 40px;"></span>
[9.3.5 Dataset Assembly Strategy]() <br>
<span style="margin-left: 40px;"></span>
[9.3.6 ]() <br>
<span style="margin-left: 20px;"></span>
[9.X Time Tracking]() <br>
[10 Bibliography](#10-bibliography)

---

## 1 Problem Statement & Motivation

During my bachelor thesis on extracting metadata from scientific charts, I encountered a significant bottleneck: the prevalence of **compound figures**. These are composite images that bundle multiple sub-figures or panels, such as charts, illustrations, or biomedical scans, into a single Figure.

This project addresses the critical need for automated **compound figure separation**. Most downstream tasks, including **chart understanding, OCR, table extraction, and figure-type classification**, assume individual figures as input. Without robust separation, automation pipelines break at the first step, leaving scientific figures largely inaccessible for large-scale analytics.

While existing research heavily favors medical imagery, this project focuses on **general scientific figures**, bridging the gap with a dedicated dataset and a YOLO-based detection model capable of splitting complex figures into their individual components.

---

## 2 Data- & Model- Download <img src="docs/assets/hf-logo.png" style="vertical-align: bottom" height="25px">
The created datasets and trained models are hosted for download on [Hugging Face](https://huggingface.co/) (README Files still to publish):

The **Dataset** can be downloaded as multiple ZIP files, including images and labels at:
[<img src="docs/assets/hf-logo.png" style="vertical-align: bottom" height="20px"> CompoundFigureSeparation](https://huggingface.co/datasets/TobiPoni/CompoundFigureSeparation)

The "to beat" **Baseline Models** can be downloaded at:
[<img src="docs/assets/hf-logo.png" style="vertical-align: bottom" height="20px"> BaseCompoundFigureSeparator](https://huggingface.co/TobiPoni/BaseCompoundFigureSeparator)

**Local Download Tools:**
You can also use the local scripts to easily setup your environment ([Quickstart: Training Preparation](#4-quickstart-training-preparation)) and download the
**[Dataset](src/utils/download_dataset.py):**
```bash
python src/utils/download_dataset.py --select default #To download datasets 04, 05 and 06
python src/utils/download_dataset.py <dataset_nr> #To download a specific dataset
```
and/or the **[Models](src/utils/download_models.py):**
```bash
python src/utils/download_models.py all #To download all models
python src/utils/download_models.py <model_name> #To download a specific model
```


---

## 3 Interactive Demo Application

The repository includes a [Streamlit-based demo application](src/demo_app.py) designed to demonstrate the practical application of the trained models and visualize the compound figure separation process.

**Key Features:**
- **Dynamic Model Selection:** Choose from various baseline models; the application automatically fetches them from Hugging Face upon selection.
- **Flexible Input:** Run inference on sample images from the included datasets or upload your own scientific figures for processing.
- **Side-by-Side Visualization:** Compare model-predicted bounding boxes against ground truth annotations to evaluate performance.
- **Configuration Transparency:** Displays the underlying model architecture and configuration (YOLO `.yaml`) directly in the interface.

To **run the demo** locally, you have to clone this repository and run the following command:
```bash
streamlit run src/demo_app.py
```

---

## 4 Quickstart: Training Preparation
### 4.1 Environment Setup
To quickly start training a model, you can execute the [prepare.py](src/prepare.py) script. This script ensures all dependencies are met, downloads the default datasets (04, 05, and 06), and converts file paths to absolute format so training can start immediately.

It is recommended to use a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate
python src/prepare.py
```

### 4.2 Start Training
Run the [train.py](src/train.py) script to train a YOLO model on your selected dataset.
```bash
python src/train.py --data 05_selected_classes --name "my_first_run" --epochs 50 --batch 16
```

**Training Options:**
For a full list of available arguments and their defaults, use the help flag:
```bash
python src/train.py -h
```
| Argument | Description |
| :--- | :--- |
| `--data` | Path to `data.yaml` or dataset folder name (e.g., `04_all_classes`) |
| `--name` | Name for this training run |
| `--model` | YOLO model variant (e.g., `yolo11n.pt`, `yolo11s.pt`) |
| `--epochs` | Number of training epochs |
| `--batch` | Batch size for training |
| `--imgsz` | Image size for inference (e.g., `960`) |
| `--device` | CUDA device (e.g. `0`) or `cpu` |

---

## 5 Repository Structure
### 5.1 Dataset Overview
The open sourced dataset covers six different versions of the dataset, all stored in the `dataset/` folder, downloadable from Hugging Face (see [Data- & Model- Download](#2-data--model--download)). Model ready datasets are 04_all_classes, 05_selected_classes and 06_selected_classes.

#### 5.1.1 Pre Datasets
The pre datasets are the datasets that were used to create the model ready datasets.
##### 5.1.1.1 `01_raw` (to be uploaded)
This directory contains the raw images, extracted from the SCI-3000 PDFs. There are a total of 9505 images extracted from the PDFs.

##### 5.1.1.2 `02_assets`
This directory contains the as **Singles** labeled figures from the SCI-3000 (~1.100 images) dataset, which were hand labeled as `Chart`, `Illustration`, `Image`, `Table` or `Other`. 
It also contains real compound Figures from the MedICaT dataset, which were not used yet, due to their inconsistent quality of their labels.

##### 5.1.1.3 `03_intermediate`
The intermediate directory contains
1. `SCI-3000_real-compound`:
This Folder contains the real compound figures extracted from the SCI-3000 dataset, including the label-studio export. ~ 700 images were labeled in label-studio with bounding boxes for 
- `Chart`
- `Illustration`
- `Image`
- `Other`
- `Shared legend`
- `Shared Title`
- `Shared X-Axis`
- `Shared Y-Axis`
- `Subpanel`
- `Other`
<img src="docs/assets/SCI-3000_examples/real_compound/0b81f25ec14c48f1bbbc3e6c51c603e5-14-fig-0.png" width="500">

2. `SCI-3000_synthetic-generated`: This directory contains the **10.000** synthetically stitched together compound images, that were generated using the singles from the SCI-3000 dataset, stored in `02_assets/SCI-3000_singles`. The folder contains the images, a yolo-labels folder and a `synthetic_labels.json` file.
<img src="docs/assets/SCI-3000_examples/synthetic_compound/synth_000000.jpg" width="500">
Read the [Detailed Documentation here]().

3. `SyntheticCompoundPlots`: During training tests, there was a lack of performance especially in identifying *Shared Titles*, *Shared Legends*, *Shared Axes* and *Sub Panels* in charts, therefore the dataset was extended by 2.500 synthetically generated compound charts, with varying sub-panel configurations. The folder contains the images and a yolo-labels folder.
<img src="docs/assets/SCI-3000_examples/snythetic_compound_plots/synth_plot_00000.png" width="500">
Read the [Detailed Documentation here]().

#### 5.1.2 Model Ready Datasets
The model ready datasets all include the images and a yolo-labels folder, as well as a `data.yaml` file, a `test.txt`, a `train.txt` and a `val.txt` file. All dataset splits were generated using the [04_Dataset_Assembly.ipynb](notebooks/04_Dataset_Assembly.ipynb) notebook and have a train/val/test split. The train split contains synthetically generated compound images and real compound images, while the val and test splits only contain real compound images.
##### 5.1.2.1 `04_all_classes`
This directory contains the final dataset, which is a combination of the real compound figures from the SCI-3000 dataset, the synthetically generated compound figures (stitched SCI-3000 Singles) and the synthetic compound plots. The folder contains the images and a yolo-labels folder. For this dataset, all classes were kept. This Folder contains
```
04_all_classes/
├──images/
├──yolo-labels/
├──data.yaml
├──test.txt
├──train.txt
└──val.txt
```
**Focus:** Sparate compound Figures into all their present sub-objects (e.g. `Chart`, `Illustration`, `Image`, `Table`, `Other`, `Shared legend`, `Shared title`, `Shared x-axis`, `Shared y-axis`, `Subpanel`)
> **⚠️ Important Note:** Due to class inbalances, oversampling and stratified splits were performed, to tackle bias in the training process. Also, 50 synthetic compound images including `Table` boxes were added to the validation and test set splits, due to the scarcity of real compound images with `Table` boxes.

##### 5.1.2.2 `05_selected_classes`
This directory contains the same images as `04_all_classes`, but only the following classes (high level) classes were kept:
- `Chart`
- `Illustration`
- `Image`
- `Other`
- `Table` (remapped)

```
05_selected_classes/
├──images/
├──yolo-labels/
├──data.yaml
├──test.txt
├──train.txt
└──val.txt
```
**Focus:** Separate compound Figures into their high level sub-objects (e.g. `Chart`, `Illustration`, `Image`, `Table`, `Other`)
> **⚠️ Important Note:** Due to class inbalances, oversampling and stratified splits were performed, to tackle bias in the training process. Also, 50 synthetic compound images including `Table` boxes were added to the validation and test set splits, due to the scarcity of real compound images with `Table` boxes.


##### 5.1.2.3 `06_compound_chart_splitter`
This directory contains the dataset that focuses on splitting charts into their sub-objects. It only contains a selection from the images of the `04_all_classes` dataset, namely the ones with only `Charts` in them. The labels then only contain the clases:
- `Shared Legend`
- `Shared Title`
- `Shared X-Axis`
- `Shared Y-Axis`
- `Subpanel`

```
06_compound_chart_splitter/
├──images/
├──yolo-labels/
├──data.yaml
├──test.txt
├──train.txt
└──val.txt
```

**Focus:** Separate Charts into their building blocks (subpanels, shared axes, shared titles, shared legends).
> **⚠️ Important Note:** Due to shortages of hand labeled compound charts, the validation and test splits contain only very few images, and there might be more work to do to improve the model's performance (e.g. less real compound charts in train dataset, hand label more real compound charts from SCI-3000, etc.).
### 5.2 Folder Structure
Below is a classic repository overview:
```
11818774_CompoundFigureSeparation/
├── dataset/                                # Where the dataset will be downloaded to
├── docs/
│   ├── assets/                             # Holds example images
│   └── metrics/                            # Holds results of the models on the test set
├── models/                                 # Where models will be downloaded to
├── src/                                    # Includes all the source code
│   ├── archive/                            # Old scripts
│   ├── generators/                         # Code for generators used to create synthetic data
│   ├── notebooks/                          # All notebooks used in the project
│   ├── utils/                              # Other utility scripts
│   ├── demo_app.py
│   ├── prepare.py
│   └── train.py
├── tests/                                  # Unit tests to verify dataset and GPU availability
│   ├── test_dataset_integrity.py
│   └── test_gpu_availability.py
├── data_links/                             # Linked large external datasets
│   └── data/
│       ├── medicat_release/
│       │   ├── figures/
│       │   ├── roco_references/
│       │   ├── medicat_grading.json        # Custom grading file for image quality
│       │   ├── s2_full_figures_oa_nonroco_combined_medical_top4_public.jsonl
│       │   └── subcaptions_public.jsonl
│       └── SCI-3000/
│           ├── Annotations/
│           ├── FiguresRaw/
│           ├── PDFs/
│           ├── interesting_pages.md
│           ├── LICENCE
│           ├── README.md
│           ├── sci-3000-page-metadata.parquet
│           └── sci-3000-pdf-metadata.parquet
└── requirements.txt                         # Project dependencies
```

---

## 6 Baseline Models
### 6.1 YOLOv11 and Configurations
For a first baseline, YOLOv11 was used for object detection. Multiple different configurations were tested on the 
### 6.2 Model Performance

---

## 7 Learnings


## 8 Future Work


## 9 Detailed Documentation


## 10 Bibliography
[1] Tianyuan Yao, et al. **Compound Figure Separation of Biomedical Images with Side Loss.** *arXiv preprint arXiv:2107.08650*, 2021.

[2] Tianyuan Yao, et al. **Compound Figure Separation of Biomedical Images: Mining Large Datasets for Self-supervised Learning.** *arXiv preprint arXiv:2208.14357*, 2022.

[3] Pengyuan Li, et al. **Compound image segmentation of published biomedical figures.** Oxford Bioinformatics, 2018.

[4] Satoshi Tsutsui, David Crandall **A Data Driven Approach for Compound Figure Separation Using Convolutional Neural Networks** *arXiv preprint arXiv:1703.05105*, 2017.

[5] Noah Siegel, et al. **FigureSeer: Parsing Result-Figures in Research Papers** *Springer Nature Link*, 2016.

[6] Tkachenko, M., Malyuk, M., Holonychev, A., Liubimov, N., & Shlyapnikov, N. (2020-2022). **Label Studio: Data labeling software.** https://github.com/heartexlabs/label-studio
