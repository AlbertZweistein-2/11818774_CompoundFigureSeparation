"""
Inverse of relativizeSplitPaths.py.
Rewrite YOLO train/val/test split lists AND data.yaml to absolute paths.

Usage:
    python src/utils/absolutizeSplitPaths.py

What it does:
    - Reads the train/val/test .txt files and prepends the absolute path.
    - Updates data.yaml 'path' entry to the absolute root of the dataset.
    - Useful if YOLO complains about relative paths during training/validation.
"""
from pathlib import Path


def absolutize_file(list_path: Path) -> int:
    """Converts entries in a .txt split file to absolute paths."""
    # Get the absolute path of the folder containing the txt file
    root_abs = list_path.parent.resolve()
    
    lines_out = []
    with list_path.open() as fh:
        for raw in fh:
            raw = raw.strip()
            if not raw:
                continue
            
            p = Path(raw)
            
            # If it's already absolute, keep it
            if p.is_absolute():
                lines_out.append(p.as_posix())
            else:
                # Join the absolute root with the relative path
                abs_path = (root_abs / p).resolve()
                lines_out.append(abs_path.as_posix())

    list_path.write_text("\n".join(lines_out) + "\n")
    return len(lines_out)


def update_yaml_path(yaml_path: Path):
    """Updates the 'path: ...' line in data.yaml to the absolute path."""
    if not yaml_path.exists():
        return

    root_abs = yaml_path.parent.resolve().as_posix()
    lines = yaml_path.read_text().splitlines()
    new_lines = []
    
    updated = False
    for line in lines:
        if line.strip().startswith("path:"):
            # Replace the path line with the absolute path
            new_lines.append(f"path: {root_abs}")
            updated = True
        else:
            new_lines.append(line)
    
    if updated:
        yaml_path.write_text("\n".join(new_lines) + "\n")
        print(f"Updated 'path' in {yaml_path}")


def main():
    # Define the datasets relative to repo root
    datasets = [
        Path("dataset/04_all_classes"),
        Path("dataset/05_selected_classes"),
        Path("dataset/06_compound_chart_splitter"),
    ]
    splits = ["train", "val", "test"]

    for ds in datasets:
        if not ds.exists():
            print(f"Skipping missing dataset folder: {ds}")
            continue

        # 1. Update YAML
        update_yaml_path(ds / "data.yaml")

        # 2. Update split text files
        for split in splits:
            f = ds / f"{split}.txt"
            if not f.exists():
                continue
            
            n = absolutize_file(f)
            print(f"Restored absolute paths in {f} -> {n} lines")


if __name__ == "__main__":
    main()