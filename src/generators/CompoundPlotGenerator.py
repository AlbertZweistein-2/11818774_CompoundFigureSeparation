"""
Compound Plot Generator
-----------------------
Generates synthetic compound plots (matplotlib) with YOLO labels. Config is
provided via CompoundPlotConfig.py. Outputs images and YOLO label files under
cfg.OUTPUT_DIR.
"""

import os
import random
import string
from typing import Optional, Tuple

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox
from tqdm import tqdm

import CompoundPlotConfig as cfg


IMG_DIR = os.path.join(cfg.OUTPUT_DIR, "images")
LBL_DIR = os.path.join(cfg.OUTPUT_DIR, "yolo-labels")
os.makedirs(IMG_DIR, exist_ok=True)
os.makedirs(LBL_DIR, exist_ok=True)

WORDS = [
    "Analysis", "Distribution", "Correlation", "Growth", "Variance", "Heatmap",
    "Spectroscopy", "Cluster", "Regression", "Frequency", "Response", "Yield",
    "Error", "Simulation", "Model", "Metric",
]


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def bbox_to_yolo(b: Bbox, w: int, h: int) -> Optional[str]:
    """Convert a matplotlib Bbox to YOLO xywh (normalized)."""
    if b is None:
        return None
    x0, y0, x1, y1 = b.x0, b.y0, b.x1, b.y1

    x_min = clamp(x0, 0, w)
    x_max = clamp(x1, 0, w)
    y_min = clamp(h - y1, 0, h)
    y_max = clamp(h - y0, 0, h)

    if x_max <= x_min or y_max <= y_min:
        return None

    cx = (x_min + x_max) / 2 / w
    cy = (y_min + y_max) / 2 / h
    bw = (x_max - x_min) / w
    bh = (y_max - y_min) / h
    return f"{cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"


def pick_layout() -> Tuple[int, int]:
    """Sample (rows, cols) layout based on config weights."""
    return random.choices(cfg.LAYOUTS, weights=cfg.LAYOUT_WEIGHTS, k=1)[0]


def compute_figsize(nr: int, nc: int) -> Tuple[float, float]:
    """Compute matplotlib figsize to hit min/max pixel constraints."""
    w_in = 3.2 * nc + 1.2
    h_in = 2.7 * nr + 1.3

    px_w = w_in * cfg.DPI
    px_h = h_in * cfg.DPI

    scale_up = max(cfg.MIN_IMG_SIZE / px_w, cfg.MIN_IMG_SIZE / px_h, 1.0)
    scale_down = min(cfg.MAX_IMG_SIZE / px_w, cfg.MAX_IMG_SIZE / px_h, 1.0)
    scale = min(scale_up, scale_down)

    return w_in * scale, h_in * scale


def gen_text_title() -> str:
    return " ".join(random.sample(WORDS, random.randint(2, 4)))


def gen_enum_token(kind: str) -> str:
    if kind == "lower":
        return random.choice(string.ascii_lowercase[:12])  # a-l
    if kind == "upper":
        return random.choice(string.ascii_uppercase[:12])  # A-L
    if kind == "roman":
        romans = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X"]
        return random.choice(romans)
    # number
    return str(random.randint(1, 10))


def gen_shared_title() -> Tuple[str, str]:
    """Return (title_text, strategy) with either free text or enumerated token."""
    strat = random.choices(
        list(cfg.TITLE_CONTENT_STRATEGY.keys()),
        weights=list(cfg.TITLE_CONTENT_STRATEGY.values()),
        k=1
    )[0]

    if strat == "text":
        return gen_text_title(), "text"

    template = random.choice(cfg.ENUM_TITLE_TEMPLATES)
    token_kind = random.choice(cfg.ENUM_TOKENS)
    token = gen_enum_token(token_kind)
    return template.replace("X", token), "enum"


def make_panel_label(i: int) -> str:
    romans = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X"]
    if random.random() < 0.5:
        return romans[i % len(romans)]
    return f"{string.ascii_lowercase[i % 26]}."


def plot_into_ax(ax, ptype: str, seed: int, cmap: str):
    """Draw a simple synthetic plot of the requested type into the axis."""
    rng = np.random.default_rng(seed)
    x = np.linspace(0, 10, 60)

    if ptype == "line":
        for k in range(random.randint(1, 3)):
            y = np.sin(x + rng.uniform(0, 2 * np.pi)) + rng.normal(0, 0.15, size=x.shape)
            ax.plot(x, y, lw=1.4, label=f"Series {k+1}")
    elif ptype == "scatter":
        y = rng.uniform(0, 10, size=x.shape)
        ax.scatter(x, y, c=x, cmap=cmap, s=18, alpha=0.95, label="Samples")
    elif ptype == "bar":
        cats = np.arange(1, 9)
        vals = rng.uniform(2, 10, size=cats.shape)
        ax.bar(cats, vals, alpha=0.85, label="Bars")
    elif ptype == "hist":
        vals = rng.normal(5, 1.8, 200)
        ax.hist(vals, bins=12, alpha=0.85, label="Hist")
    elif ptype == "boxplot":
        data = [
            rng.normal(loc=rng.uniform(-0.5, 0.5), scale=rng.uniform(0.7, 1.3), size=80)
            for _ in range(3)
        ]
        ax.boxplot(data, patch_artist=True)
    else:
        ax.plot(x, x, label="Fallback")

    ax.grid(alpha=0.25)


def choose_x_mode() -> str:
    return random.choices(
        ["col_shared_ticks", "col_shared_title_only", "none"],
        weights=[
            cfg.PROBS["x_mode_col_shared_ticks"],
            cfg.PROBS["x_mode_col_shared_title_only"],
            cfg.PROBS["x_mode_none"],
        ],
        k=1,
    )[0]


def choose_y_mode() -> str:
    return random.choices(
        ["row_shared_ticks", "row_shared_title_only", "none"],
        weights=[
            cfg.PROBS["y_mode_row_shared_ticks"],
            cfg.PROBS["y_mode_row_shared_title_only"],
            cfg.PROBS["y_mode_none"],
        ],
        k=1,
    )[0]


def apply_x_labels(axes, nr: int, nc: int, x_mode: str):
    # X sharing only within columns
    if x_mode in ("col_shared_ticks", "col_shared_title_only"):
        for c in range(nc):
            axes[nr - 1, c].set_xlabel(f"X-Axis Col {c+1} [Unit]")
        for r in range(nr - 1):
            for c in range(nc):
                axes[r, c].set_xlabel("")

        if x_mode == "col_shared_ticks":
            for r in range(nr - 1):
                for c in range(nc):
                    axes[r, c].tick_params(axis="x", labelbottom=False)
            for c in range(nc):
                axes[nr - 1, c].tick_params(axis="x", labelbottom=True)
        else:
            # title-only: keep ticklabels everywhere
            for r in range(nr):
                for c in range(nc):
                    axes[r, c].tick_params(axis="x", labelbottom=True)
    else:
        for r in range(nr):
            for c in range(nc):
                axes[r, c].set_xlabel(f"X{r+1}-{c+1} [Unit]")
                axes[r, c].tick_params(axis="x", labelbottom=True)


def apply_y_labels(axes, nr: int, nc: int, y_mode: str):
    # Y sharing only within rows
    if y_mode in ("row_shared_ticks", "row_shared_title_only"):
        for r in range(nr):
            axes[r, 0].set_ylabel(f"Y-Axis Row {r+1} [Unit]")
        for r in range(nr):
            for c in range(1, nc):
                axes[r, c].set_ylabel("")

        if y_mode == "row_shared_ticks":
            for r in range(nr):
                for c in range(1, nc):
                    axes[r, c].tick_params(axis="y", labelleft=False)
            for r in range(nr):
                axes[r, 0].tick_params(axis="y", labelleft=True)
        else:
            # title-only: keep ticklabels everywhere
            for r in range(nr):
                for c in range(nc):
                    axes[r, c].tick_params(axis="y", labelleft=True)
    else:
        for r in range(nr):
            for c in range(nc):
                axes[r, c].set_ylabel(f"Y{r+1}-{c+1} [Unit]")
                axes[r, c].tick_params(axis="y", labelleft=True)


def yolo_shared_x(axes, nr: int, nc: int, x_mode: str, renderer, w_img: int, h_img: int, out_lines: list):
    if x_mode == "none":
        return

    if x_mode == "col_shared_ticks":
        for c in range(nc):
            bb = axes[nr - 1, c].xaxis.get_tightbbox(renderer)
            line = bbox_to_yolo(bb, w_img, h_img)
            if line:
                out_lines.append(f"{cfg.CLASS_MAP['Shared X-Axis']} {line}")
    else:
        for c in range(nc):
            label_artist = axes[nr - 1, c].xaxis.label
            if label_artist is not None and label_artist.get_text():
                bb = label_artist.get_window_extent(renderer)
                line = bbox_to_yolo(bb, w_img, h_img)
                if line:
                    out_lines.append(f"{cfg.CLASS_MAP['Shared X-Axis']} {line}")


def yolo_shared_y(axes, nr: int, nc: int, y_mode: str, renderer, w_img: int, h_img: int, out_lines: list):
    if y_mode == "none":
        return

    if y_mode == "row_shared_ticks":
        for r in range(nr):
            bb = axes[r, 0].yaxis.get_tightbbox(renderer)
            line = bbox_to_yolo(bb, w_img, h_img)
            if line:
                out_lines.append(f"{cfg.CLASS_MAP['Shared Y-Axis']} {line}")
    else:
        for r in range(nr):
            label_artist = axes[r, 0].yaxis.label
            if label_artist is not None and label_artist.get_text():
                bb = label_artist.get_window_extent(renderer)
                line = bbox_to_yolo(bb, w_img, h_img)
                if line:
                    out_lines.append(f"{cfg.CLASS_MAP['Shared Y-Axis']} {line}")


def generate_one(idx: int):
    nr, nc = pick_layout()
    x_mode = choose_x_mode()
    y_mode = choose_y_mode()

    want_title = random.random() < cfg.PROBS["share_title"]
    want_legend = random.random() < cfg.PROBS["share_legend"]

    # sharex/sharey only for *_shared_ticks modes
    sharex = "col" if x_mode == "col_shared_ticks" else False
    sharey = "row" if y_mode == "row_shared_ticks" else False

    fig_w, fig_h = compute_figsize(nr, nc)
    font_fam = random.choice(cfg.FONT_FAMILIES)

    with mpl.rc_context({**cfg.STYLE, "font.family": font_fam}):
        fig, axes = plt.subplots(
            nr, nc,
            figsize=(fig_w, fig_h),
            dpi=cfg.DPI,
            sharex=sharex,
            sharey=sharey,
            constrained_layout=True,
        )

        axes = np.array(axes).reshape(nr, nc)
        fig.set_constrained_layout_pads(w_pad=0.02, h_pad=0.02, wspace=0.02, hspace=0.02)

        title_obj = None
        title_strategy = None
        if want_title:
            title_text, title_strategy = gen_shared_title()

            # placement (still suptitle only, to keep layout stable)
            placement = random.choices(
                list(cfg.TITLE_PLACEMENT_WEIGHTS.keys()),
                weights=list(cfg.TITLE_PLACEMENT_WEIGHTS.values()),
                k=1
            )[0]

            # enum titles look better smaller
            fs = 11 if title_strategy == "enum" else cfg.STYLE.get("figure.titlesize", 14)

            if placement == "top_left":
                title_obj = fig.suptitle(title_text, x=0.02, ha="left", fontweight="bold", fontsize=fs)
            else:
                title_obj = fig.suptitle(title_text, x=0.5, ha="center", fontweight="bold", fontsize=fs)

        # plot + panel labels (always inside)
        for r in range(nr):
            for c in range(nc):
                ax = axes[r, c]
                plot_into_ax(
                    ax,
                    random.choice(cfg.PLOT_TYPES),
                    seed=10000 + idx * 97 + r * 11 + c,
                    cmap=random.choice(cfg.COLOR_MAPS),
                )
                lbl = make_panel_label(r * nc + c)
                ax.text(
                    0.02, 0.95, lbl,
                    transform=ax.transAxes,
                    ha="left", va="top",
                    fontweight="bold",
                    bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=1.0),
                )

        apply_x_labels(axes, nr, nc, x_mode)
        apply_y_labels(axes, nr, nc, y_mode)

        # Legend: simple (one subplot)
        leg_obj = None
        if want_legend:
            target = axes[0, nc - 1]
            h, l = target.get_legend_handles_labels()
            if not h:
                for rr in range(nr):
                    for cc in range(nc):
                        h, l = axes[rr, cc].get_legend_handles_labels()
                        if h:
                            target = axes[rr, cc]
                            break
                    if h:
                        break
            if h:
                leg_obj = target.legend(loc="upper right", frameon=True)

        # Render
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        w_img, h_img = fig.canvas.get_width_height()

        yolo = []
        yolo.append(f"{cfg.CLASS_MAP['Chart']} 0.500000 0.500000 1.000000 1.000000")

        # Shared Title bbox (works for text and enum)
        if title_obj is not None:
            bb = title_obj.get_window_extent(renderer)
            line = bbox_to_yolo(bb, w_img, h_img)
            if line:
                yolo.append(f"{cfg.CLASS_MAP['Shared Title']} {line}")

        # Shared Legend bbox
        if leg_obj is not None:
            bb = leg_obj.get_window_extent(renderer)
            line = bbox_to_yolo(bb, w_img, h_img)
            if line:
                yolo.append(f"{cfg.CLASS_MAP['Shared Legend']} {line}")

        # Shared axis objects
        yolo_shared_x(axes, nr, nc, x_mode, renderer, w_img, h_img, yolo)
        yolo_shared_y(axes, nr, nc, y_mode, renderer, w_img, h_img, yolo)

        # ONLY Subpanel
        for r in range(nr):
            for c in range(nc):
                ax = axes[r, c]
                bb_panel = ax.get_tightbbox(renderer)  # includes tick labels + axis labels
                line = bbox_to_yolo(bb_panel, w_img, h_img)
                if line:
                    yolo.append(f"{cfg.CLASS_MAP['Subpanel']} {line}")

        basename = f"synth_plot_{idx:05d}"
        with open(os.path.join(LBL_DIR, f"{basename}.txt"), "w") as f:
            f.write("\n".join(yolo))

        fig.savefig(os.path.join(IMG_DIR, f"{basename}.png"), dpi=cfg.DPI, facecolor="white")
        plt.close(fig)


if __name__ == "__main__":
    print(f"Generating {cfg.NUM_IMAGES_TO_GENERATE} images...")
    for i in tqdm(range(cfg.NUM_IMAGES_TO_GENERATE)):
        try:
            generate_one(i)
        except Exception:
            continue
    print("Done.")
