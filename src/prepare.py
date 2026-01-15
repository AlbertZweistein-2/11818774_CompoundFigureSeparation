import subprocess
import sys
import os
from pathlib import Path

def install_dependencies():
    """Installs dependencies from requirements.txt."""
    req_path = Path(__file__).parent.parent / "requirements.txt"
    if not req_path.exists():
        print(f"Error: {req_path} not found.")
        sys.exit(1)
        
    print(f"Installing dependencies from {req_path}...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", str(req_path)])
        print("Dependencies installed successfully.\n")
    except subprocess.CalledProcessError as e:
        print(f"Error installing dependencies: {e}")
        sys.exit(1)

def download_default_dataset():
    """Runs the download script to fetch default dataset."""
    download_script = Path(__file__).parent / "utils" / "download.py"
    if not download_script.exists():
        print(f"Error: {download_script} not found.")
        sys.exit(1)
        
    print("\n[Step 2] Downloading Dataset (Default: 04, 05, 06)...")
    try:
        subprocess.check_call([sys.executable, "src/utils/download_dataset.py"]) # Default args
        print("Dataset download completed.\n")
    except subprocess.CalledProcessError as e:
        print(f"Error downloading dataset: {e}")
        sys.exit(1)

def main():
    print("=== Training Preparation Setup ===\n")
    
    # 1. Install Dependencies
    install_dependencies()
    
    # 2. Download Dataset
    download_default_dataset()
    
    print("=== Setup Complete ===")
    print("You can now start training with:")
    print("  python src/train.py --data 05_selected_classes --name my_run")

if __name__ == "__main__":
    main()
