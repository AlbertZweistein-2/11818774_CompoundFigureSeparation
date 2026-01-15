"""Dataset integrity checks for YOLO label/image pairs."""

import os
from pathlib import Path
import pytest
from PIL import Image

# Allow overriding dataset root via env for portability
DATASET_ROOT = Path(os.getenv("DATASET_ROOT", Path(__file__).parent.parent / "dataset"))

TARGET_FOLDERS = [
    DATASET_ROOT / "04_all_classes/labels/train",
    DATASET_ROOT / "04_all_classes/labels/val",
    DATASET_ROOT / "04_all_classes/labels/test",
    DATASET_ROOT / "05_selected_classes/labels/train",
    DATASET_ROOT / "05_selected_classes/labels/val",
    DATASET_ROOT / "05_selected_classes/labels/test",
    DATASET_ROOT / "06_compound_chart_splitter/labels/train",
    DATASET_ROOT / "06_compound_chart_splitter/labels/val",
    DATASET_ROOT / "06_compound_chart_splitter/labels/test",
]

def check_label_file(label_path):
    """
    Validates a single label file.
    Returns None if valid, else returns an error message string.
    """
    # 1. Image Check
    image_folder = label_path.parent.parent.parent / "images" / label_path.parent.name
    
    found_image = False
    image_path = None
    for ext in [".png", ".jpg", ".jpeg"]:
        potential_img = image_folder / label_path.with_suffix(ext).name
        if potential_img.exists():
            found_image = True
            image_path = potential_img
            break
            
    if not found_image:
        return f"No image found for label: {label_path}"

    # 2. YOLO coordinates check
    try:
        with open(label_path, "r") as f:
            lines = f.readlines()
            
        for line_idx, line in enumerate(lines):
            parts = line.strip().split()
            if not parts: continue
                
            if len(parts) != 5:
                return f"Incorrect format in {label_path.name} line {line_idx+1}"
            
            try:
                class_id, x, y, w, h = map(float, parts)
            except ValueError:
                return f"Non-numeric values in {label_path.name} line {line_idx+1}"

            EPSILON = 1e-6
            if not (0 <= x <= 1 + EPSILON): return f"x_center {x} out of bounds in {label_path.name}"
            if not (0 <= y <= 1 + EPSILON): return f"y_center {y} out of bounds in {label_path.name}"
            if not (0 < w <= 1 + EPSILON):  return f"width {w} invalid in {label_path.name}"
            if not (0 < h <= 1 + EPSILON):  return f"height {h} invalid in {label_path.name}"
            
            if x + w/2 > 1.01: return f"Box right edge out of bounds in {label_path.name}"
            if x - w/2 < -0.01: return f"Box left edge out of bounds in {label_path.name}"
            if y + h/2 > 1.01: return f"Box bottom edge out of bounds in {label_path.name}"
            if y - h/2 < -0.01: return f"Box top edge out of bounds in {label_path.name}"

    except Exception as e:
        return f"Error reading {label_path}: {e}"
        
    return None

@pytest.mark.parametrize("folder", TARGET_FOLDERS)
def test_folder_integrity(folder):
    """
    Checks integrity of all labels in a specific folder.
    Fails if any label in the folder is invalid.
    """
    if not folder.exists():
        pytest.skip(f"Folder not found: {folder}")
        
    label_files = list(folder.glob("*.txt"))
    if not label_files:
        pytest.skip(f"No labels in {folder}")

    errors = []
    # limit checking to first 1000 files to speed up CI if needed, 
    # or check all. For 12k files, python loop is fast enough (seconds), 
    # it was the pytest parameterization overhead that killed it.
    for label_file in label_files:
        error = check_label_file(label_file)
        if error:
            errors.append(error)
            # Stop after first few errors to avoid log spam
            if len(errors) >= 10:
                errors.append("... and more (stopped counting)")
                break
    
    if errors:
        pytest.fail(f"Found {len(errors)} errors in {folder.name}:\n" + "\n".join(errors))