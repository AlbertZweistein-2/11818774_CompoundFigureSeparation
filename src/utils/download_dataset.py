import argparse
import os
from pathlib import Path
try:
    from huggingface_hub import snapshot_download
except ImportError:
    print("Error: huggingface_hub not installed. Please run: pip install huggingface_hub")
    exit(1)

# Folders to download by default (Training Ready)
DEFAULT_FOLDERS = [
    "04_all_classes", 
    "05_selected_classes", 
    "06_compound_chart_splitter"
]

# All known folders (for validation/fuzzy matching)
ALL_AVAILABLE_FOLDERS = [
    "01_raw",
    "02_assets",
    "03_intermediate",
    "04_all_classes",
    "05_selected_classes",
    "06_compound_chart_splitter"
]

def download_dataset(repo_id, destination, selection, force=False):
    """
    Downloads the dataset (supports both zipped and unzipped structures).
    """
    print(f"Dataset download initiated from {repo_id}...")
    
    allow_patterns = None
    
    if selection == 'all':
        print(" -> Mode: ALL files. Downloading full repository content.")
        allow_patterns = None # Download everything
    elif selection == 'default':
        print(f" -> Mode: STANDARD (Training Ready). Downloading: {DEFAULT_FOLDERS}")
        # Match both the folder content (legacy) and the zip file (new strategy)
        allow_patterns = []
        for folder in DEFAULT_FOLDERS:
            allow_patterns.append(f"{folder}/**")
            allow_patterns.append(f"{folder}.zip")
        # Metadata
        allow_patterns.append("README.md") 
        allow_patterns.append("data.yaml")
    else:
        # User specified list
        print(f" -> Mode: CUSTOM. Downloading: {selection}")
        allow_patterns = []
        for folder in selection:
            allow_patterns.append(f"{folder}/**")
            allow_patterns.append(f"{folder}.zip")

    dest_path = Path(destination)
    print(f"Destination: {dest_path}")
    
    snapshot_download(
        repo_id=repo_id, 
        repo_type="dataset", 
        local_dir=dest_path,
        allow_patterns=allow_patterns,
        resume_download=True,
        max_workers=8
    )
    
    # Post-processing: Unzip any downloaded zip files
    print("Checking for archives to unzip...")
    import shutil
    for zip_file in dest_path.glob("*.zip"):
        print(f"Unzipping {zip_file.name}...")
        try:
            shutil.unpack_archive(zip_file, dest_path)
            print(f"  Extracted {zip_file.name}")
            # Optional: Delete zip to save space? 
            # zip_file.unlink() 
        except Exception as e:
            print(f"  Failed to unzip {zip_file.name}: {e}")

    print("Dataset download complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Dataset (Full, Training subset, or Specific folders)")
    
    # Arguments
    parser.add_argument("--repo_id", type=str, default="TobiPoni/CompoundFigureSeparation", help="HuggingFace Repo ID")
    parser.add_argument("--dest", type=str, default="dataset", help="Destination folder (default: dataset)")
    parser.add_argument("--force", action="store_true", help="Force download even if destination exists")
    
    # Simplified Selection Argument
    # Usage: 
    #   python download_dataset.py                   -> Downloads default (04,05,06)
    #   python download_dataset.py --select all      -> Downloads everything
    #   python download_dataset.py --select 01 02    -> Downloads specific folders
    parser.add_argument(
        "--select", 
        nargs="+", 
        default=["default"],
        help="Specify 'all', 'default' (04-06), or list specific folders like '01_raw'"
    )
    
    args = parser.parse_args()
    
    # Handle the 'select' logic
    selection = args.select
    
    # If user passed multiple args, it's a list. If they passed ['all'], it's a keyword.
    if len(selection) == 1 and selection[0] == 'all':
        selection_mode = 'all'
    elif len(selection) == 1 and selection[0] == 'default':
        selection_mode = 'default'
    else:
        # It's a custom list of folders. 
        # If user typed 'python src/download.py 01 02', selection is ['01', '02']
        # We assume partial matching or full matching. Let's act smart and help them if they just type "01"
        final_list = []
        for item in selection:
            # Try to fuzzy match to known folders if exact match fails
            match = next((f for f in ALL_AVAILABLE_FOLDERS if f.startswith(item)), item)
            final_list.append(match)
        selection_mode = final_list
        
    download_dataset(args.repo_id, args.dest, selection_mode, args.force)
