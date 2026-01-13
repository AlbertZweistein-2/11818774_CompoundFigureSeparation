## Generators

Quick overview of the synthetic figure generators:

- **CompoundPlotConfig.py**
	- Configuration for the matplotlib-based plot generator (output dir, layout/sharing weights, styles, class encoding).

- **CompoundPlotGenerator.py**
	- Creates synthetic compound plots with matplotlib using the config.
	- Writes images and YOLO labels to `cfg.OUTPUT_DIR` (see config).

- **SCI3000SyntheticCompoundStitcher.py**
	- Stitches single-panel assets from `dataset/02_assets/SCI-3000-Singles` into compound figures.
	- Includes oversampling rules for rare classes, random grids; outputs to `dataset/03_intermediate/SCI-3000_synthetic-generated`.

**Note:** Paths/parameters are defined in the configs/module headers. Adjust there if you change storage locations.
