import os
import shutil
from pathlib import Path
from huggingface_hub import HfApi, login

def main():
    print("Starting Upload Process...")
    login()
    api = HfApi()
    
    # Configuration
    # You can hardcode this if you prefer, or keep it interactive
    default_repo = "TobiPoni/CompoundFigureSeparation"
    repo_id = input(f"Enter HuggingFace Repo ID (default: {default_repo}): ").strip() or default_repo
    
    dataset_root = Path("dataset").resolve()
    
    # Folders to zip and upload
    folders_to_zip = [
        "01_raw",
        "02_assets",
        "03_intermediate",
        "04_all_classes",
        "05_selected_classes", 
        "06_compound_chart_splitter"
    ]
    
    print(f"Target Repository: {repo_id}")
    
    # Optional: Cleanup
    if input("Delete existing folders in repo before uploading? (y/N): ").lower().strip() == 'y':
        print("Deleting old folders on HuggingFace...")
        for folder in folders_to_zip:
            try:
                print(f"  Deleting remote folder {folder}...")
                api.delete_folder(path_in_repo=folder, repo_id=repo_id, repo_type="dataset")
            except Exception as e:
                print(f"  Skipping {folder} (probably doesn't exist): {str(e).splitlines()[0]}")

    print(f"Looking for datasets in: {dataset_root}")

    for folder_name in folders_to_zip:
        folder_path = dataset_root / folder_name
        zip_path = dataset_root / f"{folder_name}.zip"
        
        if folder_path.exists():
            print(f"\nProcessing {folder_name}...")
            
            # 1. Zip the folder if the zip doesn't exist
            if not zip_path.exists():
                print(f"  Zipping {folder_name} (this may take a while for large datasets)...")
                # root_dir=dataset_root means we run relative to dataset/
                # base_dir=folder_name means inside the zip, we start at 04_all_classes/
                shutil.make_archive(
                    base_name=str(dataset_root / folder_name), # no extension, shutil adds .zip
                    format='zip', 
                    root_dir=dataset_root, 
                    base_dir=folder_name
                )
                print(f"  Created {zip_path.name}")
            else:
                print(f"  {zip_path.name} already exists. Skipping zip step.")

            # 2. Upload the zip file
            print(f"  Uploading {zip_path.name} to HuggingFace...")
            try:
                api.upload_file(
                    path_or_fileobj=zip_path,
                    path_in_repo=f"{folder_name}.zip",
                    repo_id=repo_id,
                    repo_type="dataset",
                )
                print(f"  Successfully uploaded {zip_path.name}!")
            except Exception as e:
                print(f"  Error uploading {folder_name}: {e}")
                
        else:
            print(f"Skipping {folder_name} (Not found in {dataset_root})")

    print("\nAll tasks completed.")

if __name__ == "__main__":
    main()
