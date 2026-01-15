# Generators

This directory contains scripts and configuration files for generating synthetic datasets of compound figures. These generators are essential for handling class imbalances and providing large-scale labeled training data.

## 1. Matplotlib-based Plot Generator

This generator creates synthetic multi-panel plots using Matplotlib, specifically designed to train detection models on chart components and layouts.

### Configuration (`CompoundPlotConfig.py`)
Controls the diversity and statistical properties of the generated images:

- **Quantities & Resolution:**
    - Generates **2,500** images by default (`NUM_IMAGES_TO_GENERATE`).
    - Output resolution is **120 DPI**, aiming for images between **700px** and **1400px** in size.
- **Layouts & Weights:** 
    - Supports grids from 1x2 up to 4x2.
    - Distribution weights favor: (2,2) at **22%**, (1,3) at **20%**, and (2,3) at **18%**.
- **Component Probabilities:**
    - **Global Title:** **75%** chance to include a shared title.
    - **Global Legend:** **55%** chance to include a shared legend.
- **Sharing Modes (X/Y axes):**
    - **Ticks Shared:** (**45%** probability) X shared per column, Y shared per row. Only the outer axes show labels/ticks.
    - **Title Only Shared:** (**35%** probability) Axis labels are shared, but all subplots keep interior ticks.
    - **No Sharing:** (**20%** probability) Each subplot is independent.
- **Titling Strategies:**
    - **Text-based:** (**70%** weight) Randomly sampled multi-word scientific titles.
    - **Enumerated:** (**30%** weight) Labels like "(a)", "Fig. X", or "I."
    - **Placement:** Titles are placed **Centered (60%)** or in the **Top-Left (40%)**.
- **Styling:** Unified font sizes for consistency:
    - Font Family: Random choice of sans-serif, serif, or monospace.
    - Sizes: Axes/Labels (**10pt**), Ticks/Legend (**9pt**), Figure Titles (**14pt**).
- **Class Mapping:** Uses a global schema (11 classes) to ensure coordination with other splits. This specific generator typically emits: `Chart (0)`, `Shared Legend (4)`, `Shared Title (5)`, `Shared X-Axis (6)`, `Shared Y-Axis (7)`, and `Subpanel (8)`.

### generator (`CompoundPlotGenerator.py`)
- Programmatically constructs subplots, populates them with random data (line, scatter, bar, hist, or boxplots), and applies color maps.
- **Accurate Bounding Boxes:** Uses Matplotlib's `get_tightbbox` and `get_window_extent` with the renderer to calculate pixel-perfect YOLO labels for every component.
- **Outputs:** Images and YOLO labels are written to `dataset/03_intermediate/SyntheticCompoundPlots`.

## 2. SCI-3000 Synthetic Stitcher

### Script (`SCI3000SyntheticCompoundStitcher.py`)
- **Action:** Stitches real single-panel assets from `dataset/02_assets/SCI-3000-Singles` into new compound figures.
- **Augmentation Goal:** Creates **~10,000 synthetic compound figures**.
- **Balanced Sampling:** Includes specific oversampling rules for rare classes to ensure the detection model sees enough examples of varied content types.
- **Output:** Materializes images and labels in `dataset/03_intermediate/SCI-3000_synthetic-generated`.

---

**Note:** Always check the `CompoundPlotConfig.py` file to adjust storage locations, DPI settings, or the number of images before running a generation pass.
