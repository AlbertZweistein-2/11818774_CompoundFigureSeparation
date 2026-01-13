"""
Config for Compound Plot Generator
----------------------------------
Controls output location, sampling weights for layouts and sharing modes,
plot styles, and class mapping for YOLO labels. Tweaking values here changes
the diversity of generated synthetic compound plots.
"""

# Output root for generated images + YOLO labels

OUTPUT_DIR = "../../dataset/03_intermediate/SyntheticCompoundPlots"
NUM_IMAGES_TO_GENERATE = 2500

LAYOUTS = [(1, 2), (1, 3), (1, 4), (2, 2), (2, 3), (3, 2), (3, 3), (4, 2)]
LAYOUT_WEIGHTS = [0.14, 0.20, 0.08, 0.22, 0.18, 0.08, 0.06, 0.04]

# X sharing applies per column; Y sharing applies per row
PROBS = {
    "x_mode_col_shared_ticks": 0.45,
    "x_mode_col_shared_title_only": 0.35,
    "x_mode_none": 0.20,

    "y_mode_row_shared_ticks": 0.45,
    "y_mode_row_shared_title_only": 0.35,
    "y_mode_none": 0.20,

    "share_legend": 0.55,
    "share_title": 0.75,
}

# Shared Title content: either text title OR figure-label style
TITLE_CONTENT_STRATEGY = {"text": 0.70, "enum": 0.30}
TITLE_PLACEMENT_WEIGHTS = {"center": 0.60, "top_left": 0.40}

# Figure-label styles like (a), a), A., I, Figure 3
ENUM_TITLE_TEMPLATES = ["(X)", "X)", "X.", "X", "Figure X", "Fig. X"]
ENUM_TOKENS = ["lower", "upper", "roman", "number"]  # how to fill X

PLOT_TYPES = ["line", "scatter", "bar", "hist", "boxplot"]
COLOR_MAPS = ["viridis", "plasma", "inferno", "Greys", "Blues", "tab10"]

# Einheitliche Schriftgrößen (keine random axis sizes!)
STYLE = {
    "font.size": 10,
    "axes.titlesize": 10,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.titlesize": 14,  # for text titles
}

FONT_FAMILIES = ["sans-serif", "serif", "monospace"]

CLASS_MAP = {
    "Chart": 0,
    "Illustration": 1,
    "Image": 2,
    "Other": 3,
    "Shared Legend": 4,
    "Shared Title": 5,
    "Shared X-Axis": 6,
    "Shared Y-Axis": 7,
    "Subpanel": 8,
    "Table": 9,
    "Subplot": 10,  # exists in schema, but not emitted
}

DPI = 120
MIN_IMG_SIZE = 700
MAX_IMG_SIZE = 1400
