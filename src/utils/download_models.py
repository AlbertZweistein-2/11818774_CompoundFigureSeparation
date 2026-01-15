
import argparse
import sys
from pathlib import Path

try:
    from huggingface_hub import hf_hub_download
except ImportError:
    print("Error: huggingface_hub not installed. Please run: pip install huggingface_hub")
    exit(1)

# Configuration
HF_MODEL_REPO = "TobiPoni/BaseCompoundFigureSeparator"
REPO_URL = f"https://huggingface.co/{HF_MODEL_REPO}"

# Standard Models List (Should match the ones in demo_app.py and on HuggingFace)
STANDARD_MODELS = [
    "04_all_classes_Ymedium_1280_baseline.pt",
    "04_all_classes_Ysmall_1280_baseline.pt", 
    "04_all_classes_Ysmall_960_baseline.pt",
    "05_selected_classes_Ynano_1280_baseline.pt",
    "06_compound_chart_splitter_Ynano_1280_baseline.pt"
]

def download_model(model_name, dest_dir):
    """Downloads a single model and its config if available."""
    dest_path = Path(dest_dir)
    dest_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Downloading {model_name}...")
    try:
        # Download .pt
        hf_hub_download(
            repo_id=HF_MODEL_REPO,
            filename=model_name,
            local_dir=dest_path,
            local_dir_use_symlinks=False
        )
        
        # Try to download config .yaml too
        config_name = str(Path(model_name).with_suffix('.yaml'))
        try:
            hf_hub_download(
                repo_id=HF_MODEL_REPO,
                filename=config_name,
                local_dir=dest_path,
                local_dir_use_symlinks=False
            )
            print(f"  + Config {config_name} downloaded.")
        except Exception:
            print(f"  (Config {config_name} not found, skipping)")
            
        print(f"Successfully downloaded {model_name}")
        return True
    except Exception as e:
        print(f"Error downloading {model_name}: {e}")
        return False

def print_help():
    print("Usage: python src/utils/download_models.py [MODELS...]")
    print("\nArguments:")
    print("  all             Download ALL standard models.")
    print("  <model_name>    Download a specific model by filename (e.g., '04_all_classes_Ymedium_1280_baseline.pt').")
    print("\nAvailable Standard Models:")
    for m in STANDARD_MODELS:
        print(f"  - {m}")
    print(f"\nFor more models and details, visit the repository:\n  {REPO_URL}")

def main():
    if len(sys.argv) < 2:
        print("Error: No arguments provided.\n")
        print_help()
        sys.exit(1)

    # Skip the script name
    args = sys.argv[1:]
    
    # Check for help flags
    if '-h' in args or '--help' in args:
        print_help()
        sys.exit(0)

    # Determine what to download
    models_to_download = []
    
    if 'all' in args:
        models_to_download = STANDARD_MODELS
    else:
        # Assume args are model names
        # You could add fuzzy matching here if you want
        models_to_download = args

    # Destination is fixed to 'models/' relative to repo root usually
    # Assuming this script is in src/utils/, so models is ../../models
    dest_dir = Path(__file__).parent.parent.parent / "models"
    
    print(f"Target Directory: {dest_dir.resolve()}")
    print("-" * 40)

    success_count = 0
    for model in models_to_download:
        if download_model(model, dest_dir):
            success_count += 1
            
    print("-" * 40)
    print(f"Download complete. {success_count}/{len(models_to_download)} models downloaded.")

if __name__ == "__main__":
    main()
