import json
import cv2
import numpy as np
import random
import os
from pathlib import Path
from tqdm import tqdm
from datetime import datetime

# --- KONFIGURATION ---
NUM_IMAGES_TO_GENERATE = 10000

# Paths (make sure these match your folder structure)
ASSET_DIR = Path("../../dataset/02_assets/SCI-3000-Singles")
JSON_INPUT_PATH = Path("../../dataset/02_assets/SCI-3000-Singles/single_labels.json")

OUT_ROOT = Path("../../dataset/03_intermediate/SCI-3000_synthetic-generated")
OUT_IMG_DIR = OUT_ROOT / "images"
OUT_LBL_DIR = OUT_ROOT / "yolo-labels"
OUT_JSON_FILE = OUT_ROOT / "synthetic_labels.json"
OUT_CLASSES_FILE = Path("../../dataset/classes.json")

# TARGET WIDTH (typical figure width in high-res)
PAGE_WIDTH = 1600 

OVERSAMPLE_RULES = {
    "Table": 20, "Image": 10, "Chart": 1, "Subplot": 1, "Illustration": 2
}

LABEL_STUDIO_MAPPING = [
    {"id": 0, "name": "Chart"},
    {"id": 1, "name": "Illustration"},
    {"id": 2, "name": "Image"},
    {"id": 3, "name": "Other"},
    {"id": 4, "name": "Shared Legend"},
    {"id": 5, "name": "Shared Title"},
    {"id": 6, "name": "Shared X-Axis"},
    {"id": 7, "name": "Shared Y-Axis"},
    {"id": 8, "name": "Subpanel"},
    {"id": 9, "name": "Table"},
    {"id": 10, "name": "Subplot"}
]

# Grid Wahrscheinlichkeiten (Row, Col)
GRID_CHOICES = [
    ((1, 1), 5), 
    ((1, 2), 20), ((2, 1), 20),
    ((2, 2), 30),
    ((1, 3), 10), ((3, 1), 5),
    ((2, 3), 5),  ((3, 2), 5),
    ((2, 4), 5),  ((4, 2), 2),
    # Make extreme layouts less frequent to avoid mostly stripe-like figures
    ((1, 4), 5), 
    ((1, 5), 2),
    ((2, 1), 5)
]

def setup_directories():
    OUT_IMG_DIR.mkdir(parents=True, exist_ok=True)
    OUT_LBL_DIR.mkdir(parents=True, exist_ok=True)
    mapping_dict = {"categories": LABEL_STUDIO_MAPPING, "info": {"version": "1.0"}}
    with open(OUT_CLASSES_FILE, "w") as f:
        json.dump(mapping_dict, f, indent=2)
    return {item['name']: item['id'] for item in LABEL_STUDIO_MAPPING}

def load_and_oversample_assets(name_to_id):
    print("Loading assets...")
    # Error handling if the JSON file is missing
    if not JSON_INPUT_PATH.exists():
        print(f"ERROR: {JSON_INPUT_PATH} not found!")
        return []

    with open(JSON_INPUT_PATH, 'r') as f:
        data = json.load(f)
    pool = []
    
    # Counter for debugging
    counts = {k:0 for k in OVERSAMPLE_RULES.keys()}
    
    for filename, label in data.items():
        if label not in name_to_id: continue
        
        # Path check (sometimes paths in the JSON differ from the filesystem)
        full_path = ASSET_DIR / filename
        if not full_path.exists(): 
            continue
            
        factor = OVERSAMPLE_RULES.get(label, 1)
        item = {"path": str(full_path), "label": label, "class_id": name_to_id[label]}
        
        for _ in range(factor): 
            pool.append(item)
            if label in counts: counts[label] += 1
            
    print(f"Pool size: {len(pool)} assets.")
    return pool

def get_random_grid():
    grids, weights = zip(*GRID_CHOICES)
    return random.choices(grids, weights=weights, k=1)[0]

def create_compound(idx, asset_pool):
    rows, cols = get_random_grid()
    num_slots = rows * cols
    
    if len(asset_pool) < num_slots:
        return None

    selection = random.sample(asset_pool, num_slots)
    
    # --- LAYOUT LOGIC: PAGE FLOW ---
    # Build the figure from top to bottom.
    
    bg_gray = random.randint(245, 255) # Very light gray/white
    
    # Define margins
    outer_pad = random.randint(20, 50)
    gap_x = random.randint(10, 30)
    gap_y = random.randint(20, 50)
    
    # Available width for content
    content_width = PAGE_WIDTH - (2 * outer_pad)
    
    # Breite pro Spalte (Zelle)
    # (Content Width - (Alle Gaps)) / Anzahl Spalten
    col_width = int((content_width - ((cols - 1) * gap_x)) / cols)
    
    # First compute the final canvas height.
    # We do this by simulating placement row by row.
    
    row_buffers = [] # Speichert (Image, LabelInfo) pro Zeile
    
    current_asset_idx = 0
    total_canvas_height = outer_pad
    
    for r in range(rows):
        row_items = []
        max_row_h = 0
        
        for c in range(cols):
            asset = selection[current_asset_idx]
            current_asset_idx += 1
            
            img = cv2.imread(asset["path"])
            if img is None: 
                # Fallback: create an empty white image
                img = np.ones((100, 100, 3), dtype=np.uint8) * 255
            
            # Resize to column width (keep aspect ratio)
            h_img, w_img = img.shape[:2]
            scale = col_width / w_img
            new_w = col_width # Exakt Spaltenbreite
            new_h = int(h_img * scale)
            
            resized = cv2.resize(img, (new_w, new_h))
            
            row_items.append({
                "img": resized,
                "label": asset["label"],
                "class_id": asset["class_id"],
                "h": new_h,
                "w": new_w
            })
            
            if new_h > max_row_h:
                max_row_h = new_h
        
        # Row prepared; store it.
        row_buffers.append({
            "items": row_items,
            "height": max_row_h
        })
        
        total_canvas_height += max_row_h + gap_y

    # Remove last gap + add bottom padding
    total_canvas_height = total_canvas_height - gap_y + outer_pad
    
    # --- CANVAS ERSTELLEN ---
    canvas = np.ones((total_canvas_height, PAGE_WIDTH, 3), dtype=np.uint8) * bg_gray
    
    yolo_labels = []
    json_labels = []
    
    # --- RENDERN ---
    current_y = outer_pad
    
    for row_data in row_buffers:
        row_h = row_data["height"]
        items = row_data["items"]
        
        for c, item in enumerate(items):
            img = item["img"]
            h, w = item["h"], item["w"]
            
            # X Position
            x_pos = outer_pad + c * (col_width + gap_x)
            
            # Y position (center within row for better alignment with varying heights)
            y_pos = current_y + (row_h - h) // 2
            
            # Paste
            canvas[y_pos:y_pos+h, x_pos:x_pos+w] = img
            
            # --- LABELS ---
            # YOLO (Normalized)
            cx = (x_pos + w / 2) / PAGE_WIDTH
            cy = (y_pos + h / 2) / total_canvas_height
            bw = w / PAGE_WIDTH
            bh = h / total_canvas_height
            
            yolo_labels.append(f"{item['class_id']} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
            
            # JSON (Pixel / Percentage)
            json_labels.append({
                "x": (x_pos / PAGE_WIDTH) * 100,
                "y": (y_pos / total_canvas_height) * 100,
                "width": (w / PAGE_WIDTH) * 100,
                "height": (h / total_canvas_height) * 100,
                "rotation": 0,
                "rectanglelabels": [item["label"]],
                "original_width": PAGE_WIDTH,
                "original_height": total_canvas_height
            })
            
        current_y += row_h + gap_y

    # Save
    filename = f"synth_{idx:06d}.jpg"
    out_img_path = OUT_IMG_DIR / filename
    cv2.imwrite(str(out_img_path), canvas)
    
    with open(OUT_LBL_DIR / f"synth_{idx:06d}.txt", "w") as f:
        f.write("\n".join(yolo_labels))
        
    return {
        "image": f"/data/local-files/?d={out_img_path.absolute()}",
        "id": 100000 + idx,
        "label": json_labels,
        "annotator": 0,
        "created_at": datetime.now().isoformat(),
        "meta": {"layout": f"{rows}x{cols}", "size": f"{PAGE_WIDTH}x{total_canvas_height}"}
    }

def main():
    name_to_id = setup_directories()
    pool = load_and_oversample_assets(name_to_id)
    if not pool: 
        print("Abort: no assets in the pool.")
        return

    all_tasks = []
    print(f"--- START GENERATION: {NUM_IMAGES_TO_GENERATE} IMAGES ---")
    
    for i in tqdm(range(NUM_IMAGES_TO_GENERATE)):
        task = create_compound(i, pool)
        if task: all_tasks.append(task)
            
    with open(OUT_JSON_FILE, "w") as f:
        json.dump(all_tasks, f, indent=2)
    print("Done!")

if __name__ == "__main__":
    main()