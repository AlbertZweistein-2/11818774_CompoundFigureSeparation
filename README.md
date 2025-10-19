# 11818774_CompoundFigureSeparation
## Content
[1 Problem Statement](#1-problem-statement)  
[2 Relevant Scientific Literature](#2-relevant-scientific-literature)  
[3 Scope](#3-scope)  
[4 Summary](#4-summary)  
[5 Deliverables for Assignment 1 of Applied Deep Learning Course, TU Wien](#5-deliverables-for-assignment-1-of-applied-deep-learning-course-tu-wien)  
[6 Bibliography](#6-bibliography)

## 1 Problem Statement  
During my bachelor thesis on extracting metadata from scientific charts in research papers, I encountered the challenge of separating **compound figures** into their individual **sub-figures** or **panels**. Scientific figures often combine multiple charts, diagrams, or images within a single composite figure. To analyze or extract information from specific parts of such figures, it is essential to first **split compound figures into their individual components**.

## 2 Relevant Scientific Literature  
Most existing research on compound figure separation focuses on **medical images**, such as X-rays or MRI scans. The most commonly used publicly available dataset is the **ImageCLEF Medical dataset**, from [ImageCLEF 2016](https://www.imageclef.org/2016). Several approaches have been proposed for compound figure separation, but nearly all are domain-specific to medical imaging and do not generalize well to **general scientific figures**.

Notable publications in this area include:  
- *Compound Figure Separation of Biomedical Images with Side Loss* [[1]](#bibliography)  
- *Compound Figure Separation of Biomedical Images: Mining Large Datasets for Self-supervised Learning* [[2]](#bibliography)
- *Compound Image Segmentation of Published Biomedical Figures* [[3]](#bibliography)

Some research has explored the separation of more general scientific figures:  
- *A Data-Driven Approach for Compound Figure Separation Using Convolutional Neural Networks* [[4]](#bibliography)  
- *FigureSeer: Parsing Result-Figures in Research Papers* [[5]](#bibliography)

In testing several publicly available implementations of these approaches, including the [**Compound Figure Separator**](https://github.com/apple2373/figure-separator?tab=readme-ov-file) developed in the scope of [[4]](#bibliography), I found that they perform poorly on general scientific figures. In particular, when **charts or plots** were included, the models often failed to separate sub-figures correctly.

Currently, there are **no publicly available datasets** dedicated to general scientific figure separation. The method described in [[4]](#bibliography) relied on **transfer learning** and an **automatically synthesized dataset**, evaluated primarily on ImageCLEF Medical.


## 3 Scope  
The goal of this project, carried out in the scope of the **Applied Deep Learning** course, is to create a **dataset and model for general scientific figure separation**.

To achieve this, the project will:  
1. **Create a synthetic dataset** of compound scientific figures by combining multiple charts and figures into composite images and automatically generating corresponding bounding box labels.  
2. **Collect a real-world dataset** of scientific figures extracted from research papers using tools such as **Docling**, or existing figure datasets, and manually annotate them with bounding boxes.  
3. **Fine-tune the latest YOLO model** on the combined dataset and evaluate its performance on a test set of real-world compound figures.

The final outcome will include:  
- The **dataset** (synthetic + real-world)  
- **Boilerplate training code** for YOLO  
- The **trained model** and **evaluation results**

All resources will be made **publicly available** on GitHub and/or Hugging Face to support further research on general scientific figure separation.

## 4 Summary  
This project addresses the lack of a general-purpose solution for **compound figure separation** in scientific publications.  
While existing methods and datasets focus almost exclusively on medical imaging, this project aims to extend the field by creating and publishing a **dataset and fine-tuned YOLO-based model** specifically designed for **general scientific figures**, including charts and mixed figure types.  

The approach combines **synthetic data generation**, **manual annotation**, and **deep learning fine-tuning**, resulting in a reproducible pipeline for training and evaluating figure separation models.  
All datasets, code, and models will be released as **open-source resources** to enable future research and practical applications in automated figure understanding.

## 5 Exact Deliverables for Assignment 1 of Applied Deep Learning Course, TU Wien

### 5.1 References (≥2 papers)
See [Bibliography](#6-bibliography) section below.

### 5.2 Topic decision
**General Scientific Compound Figure Separation** (split multi-panel research figures into sub-figures/panels).

### 5.3 Project type
**Bring your own data** (synthetic + real-world set) with a **YOLO-based baseline** (fine-tuning + evaluation).

### 5.4 Written summary
- **Idea & approach (short):** Build and release a dataset for **general** (non-medical) compound figures and fine-tune a **YOLO** detector to predict sub-figure bounding boxes; evaluate on real research figures.
- **Dataset (to use/collect):**  
  - **Synthetic set:** programmatically compose charts/plots/diagrams/other images into compound figures; auto-generate bounding boxes.  
  - **Real-world set:** extract figures from PDFs (e.g., via Docling) and **manually** annotate sub-figure boxes.  
  - **Target size:** Still to be determined.

### Work-Breakdown Structure
| Task | What’s included | Estimate |
|---|---|---|
| Dataset: synthetic generation | composition scripts, layouts, noise/spacing, auto-labels | **5-6 days** |
| Dataset: real extraction | PDF parsing (Docling), filtering, curation | **3-4 days** |
| Dataset: manual labeling | Label the extracted figures | **5-6 days** |
| Model design & setup | Select YOLO variant, configure data pipeline, choose suitable loss function | **1–2 days** |
| Training & fine-tuning | baseline runs, LR/aug tweaks, checkpoints | **2-4 days** |
| Evaluation | metrics (mAP@IoU, recall), error analysis on real test set | **2-4 days** |
| Minimal application | CLI/notebook demo to run on new images | **1-2 days** |
| Final report | concise write-up (method, data, results, limits) | **1-2 days** |
| Presentation | 6–8 slides with demo screenshots | **1 day** |

## 6 Bibliography
[1] Tianyuan Yao, et al. **Compound Figure Separation of Biomedical Images with Side Loss.** *arXiv preprint arXiv:2107.08650*, 2021. [https://arxiv.org/abs/2107.08650](https://arxiv.org/abs/2107.08650); **GitHub**: https://github.com/hrlblab/ImageSeperation

[2] Tianyuan Yao, et al. **Compound Figure Separation of Biomedical Images: Mining Large Datasets for Self-supervised Learning.** *arXiv preprint arXiv:2208.14357*, 2022. https://arxiv.org/abs/2208.14357;

[3] Pengyuan Li, et al. **Compound image segmentation of published biomedical figures.** Oxford Bioinformatics, Volume 34, Issue 7, April 2018, Pages 1192 - 1199. https://academic.oup.com/bioinformatics/article/34/7/1192/4430539; **Github**: https://github.com/pengyuanli/FigSplit?tab=readme-ov-file

[4] Satoshi Tsutsui, David Crandall **A Data Driven Approach for Compound Figure Separation Using Convolutional Neural Networks** *arXiv preprint arXiv:1703.05105*, 2017. https://arxiv.org/abs/1703.05105; **GitHub**: https://github.com/apple2373/figure-separator?tab=readme-ov-file; Website: https://vision.soic.indiana.edu/figure-separator/

[5] Noah Siegel, et al. **FigureSeer: Parsing Result-Figures in Research Papers** *Springer Nature Link*, 2016. https://link.springer.com/chapter/10.1007/978-3-319-46478-7_41; **GitHub**: https://github.com/allenai/figureseer


*Prepared by Tobias Ponesch (11818774) for the TU Wien course “Applied Deep Learning”, WS2025.*