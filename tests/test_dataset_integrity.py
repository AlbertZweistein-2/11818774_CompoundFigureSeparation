import os
import pytest
from pathlib import Path
from PIL import Image

DATASET_ROOT = Path(__file__).parent.parent / "dataset"

TARGET_FOLDERS = [
    DATASET_ROOT / "04_model_ready/labels/train",
    DATASET_ROOT / "04_model_ready/labels/val",
    DATASET_ROOT / "04_model_ready/labels/test"
]

def get_label_files():
    files = []
    for folder in TARGET_FOLDERS:
        if folder.exists():
            files.extend(list(folder.glob("*.txt")))
    return files

@pytest.mark.parametrize("label_path", get_label_files())
def test_label_validity(label_path):
    """
    Checks for all label files in the dataset:
    1. Does the corresponding image exist?
    2. Are the YOLO coordinates valid (0-1)?
    """
    
    # 1. Image Check
    # We assume images are in the parallel "images" folder instead of "labels"
    # Path manipulation: .../labels/train/image.txt -> .../images/train/image.png
    image_folder = label_path.parent.parent.parent / "images" / label_path.parent.name
    print(image_folder)
    
    # Check for extensions (png, jpg, jpeg)
    found_image = False
    image_path = None
    for ext in [".png", ".jpg", ".jpeg"]:
        potential_img = image_folder / label_path.with_suffix(ext).name
        if potential_img.exists():
            found_image = True
            image_path = potential_img
            break
            
    assert found_image, f"No image found for label: {label_path}"

    # (Optional) Can the image be opened? (Corrupt file check)
    try:
        with Image.open(image_path) as img:
            img.verify() 
    except Exception:
        pytest.fail(f"Image file is corrupt: {image_path}")

    # 2. YOLO coordinates check
    with open(label_path, "r") as f:
        lines = f.readlines()
        
    for line_idx, line in enumerate(lines):
        parts = line.strip().split()
        
        # Ignore empty lines
        if not parts:
            continue
            
        # Format: class x y w h
        assert len(parts) == 5, f"Incorrect format in {label_path.name} line {line_idx+1}"
        
        try:
            class_id, x, y, w, h = map(float, parts)
        except ValueError:
            pytest.fail(f"Non-numeric values in {label_path.name} line {line_idx+1}")

        # The sacred YOLO rules
        EPSILON = 1e-6
        assert 0 <= x <= 1 + EPSILON, f"x_center {x} out of [0,1] in {label_path.name}"
        assert 0 <= y <= 1 + EPSILON, f"y_center {y} out of [0,1] in {label_path.name}"
        assert 0 < w <= 1 + EPSILON,  f"width {w} invalid in {label_path.name}"
        assert 0 < h <= 1 + EPSILON,  f"height {h} invalid in {label_path.name}"
        
        # Logic check: Box must not go out of bounds
        assert x + w/2 <= 1.01, f"Box extends beyond right edge in {label_path.name}"
        assert x - w/2 >= -0.01, f"Box extends beyond left edge in {label_path.name}"
        assert y + h/2 <= 1.01, f"Box extends beyond bottom edge in {label_path.name}"
        assert y - h/2 >= -0.01, f"Box extends beyond top edge in {label_path.name}"