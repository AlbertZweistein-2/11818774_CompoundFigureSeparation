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
        for split in splits:
            f = ds / f"{split}.txt"
            if not f.exists():
                continue
            n = relativize_file(f)
            print(f"Rewrote {f} -> {n} lines")


if __name__ == "__main__":
    main()
