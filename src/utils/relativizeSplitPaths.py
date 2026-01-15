"""Rewrite YOLO train/val/test split lists to repo-relative paths for portability.

Usage:
    python src/utils/make_splits_relative.py

What it does:
    - Overwrites train/val/test .txt files for the YOLO datasets so they contain
      paths relative to their dataset root instead of absolute machine paths.
    - Targets the datasets:
        * dataset/04_all_classes
        * dataset/05_selected_classes
        * dataset/06_compound_chart_splitter
"""
from pathlib import Path


def relativize_file(list_path: Path) -> int:
    root = list_path.parent  # e.g., dataset/04_all_classes
    lines_out = []
    with list_path.open() as fh:
        for raw in fh:
            raw = raw.strip()
            if not raw:
                continue
            p = Path(raw)
            try:
                rel = p.relative_to(root)
            except ValueError:
                # If the entry is outside the root, keep the last 3 parts (images/split/file)
                rel = Path(*p.parts[-3:]) if len(p.parts) >= 3 else Path(p.name)
            lines_out.append(rel.as_posix())
    list_path.write_text("\n".join(lines_out) + "\n")
    return len(lines_out)


def main():
    datasets = [
        Path("dataset/04_all_classes"),
        Path("dataset/05_selected_classes"),
        Path("dataset/06_compound_chart_splitter"),
    ]
    splits = ["train", "val", "test"]

    for ds in datasets:
        # 1. Update .txt splits
        for split in splits:
            f = ds / f"{split}.txt"
            if not f.exists():
                continue
            n = relativize_file(f)
            print(f"Rewrote {f} -> {n} lines")
            
        # 2. Update data.yaml
        yaml_path = ds / "data.yaml"
        if yaml_path.exists():
            update_yaml(yaml_path)

def update_yaml(yaml_path: Path):
    """
    Ensures data.yaml uses portable relative paths.
    Sets:
      path: .
      train: train.txt
      val: val.txt
      test: test.txt
    Preserves 'names' and 'nc'.
    """
    lines = []
    names_block = []
    in_names = False
    
    with yaml_path.open() as f:
        for line in f:
            stripped = line.strip()
            # Capture names block
            if stripped.startswith('names:'):
                in_names = True
                names_block.append(line)
                continue
            
            if in_names:
                # Naive check for indentation to capture the block
                if line.startswith(' ') or line.startswith('\t') or stripped.startswith('-'):
                    names_block.append(line)
                else:
                    in_names = False
                    # Fallthrough if we hit something else (though usually names is last)
            
            # Capture nc if present
            if stripped.startswith('nc:'):
                lines.append(line)

    # Reconstruct
    new_content = [
        "path: .",
        "train: train.txt",
        "val: val.txt",
        "test: test.txt",
    ]
    
    # Add nc if we found it (optional, YOLO infers from names often but good to keep)
    # However, names block is most important. Simple parsing might lose 'nc' if it was mixed.
    # Let's actually Just use basic string replacement for the top keys to be safer against structure variations,
    # OR explicitly write the standard structure + the names block we extracted.
    
    # Safest: Read strictly the names map and rewrite everything cleanly.
    # But without PyYAML, valid parsing is annoying. 
    # Let's assume the user just needs the paths fixed.
    
    # Let's try to be robust: read all lines, replace path/train/val/test lines, keep others.
    
    output_lines = []
    # explicit overrides
    overrides = {
        'path:': 'path: .',
        'train:': 'train: train.txt',
        'val:': 'val: val.txt',
        'test:': 'test: test.txt'
    }
    
    with yaml_path.open() as f:
        content = f.readlines()
        
    # Check if we need to add missing keys or just update
    # Strategy: Filter out existing path/train/val/test lines, prepend new ones, then append the rest.
    
    # Helper to check if line is one of our keys
    def is_key(line, key):
        return line.strip().startswith(key)
        
    kept_lines = []
    for line in content:
        if any(is_key(line, k) for k in overrides.keys()):
            continue
        kept_lines.append(line)
        
    # Write new file
    with yaml_path.open('w') as f:
        # Write our standard header
        f.write("path: .\n")
        f.write("train: train.txt\n")
        f.write("val: val.txt\n")
        f.write("test: test.txt\n")
        
        # Write the rest (names, nc, etc.)
        for line in kept_lines:
            f.write(line)
            
    print(f"Updated {yaml_path} to use relative paths.")

if __name__ == "__main__":
    main()
