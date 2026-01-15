import argparse
from ultralytics import YOLO
from pathlib import Path
import os

def train(args):
    print(f"Loading Model: {args.model}...")
    model = YOLO(args.model)

    # Convert relative path to absolute to avoid YOLO confusion if cwd changes
    data_path = Path(args.data).resolve()
    if not data_path.exists():
        # Try finding it relative to project root dataset folder if simple name given
        repo_root = Path(__file__).parent.parent
        potential_path = repo_root / 'dataset' / args.data / 'data.yaml'
        if potential_path.exists():
            data_path = potential_path
        else:
            # Try direct yaml path
            potential_path = repo_root / 'dataset' / args.data
            if potential_path.exists() and potential_path.suffix in ['.yaml', '.yml']:
                data_path = potential_path

    print(f"Using Dataset Config: {data_path}")
    
    # Ensure runs dir is in project root
    project_root = Path(__file__).parent.parent
    runs_dir = project_root / 'runs'

    print(f"Starting Training Run: {args.name}")
    results = model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        name=args.name,
        device=args.device,
        workers=args.workers,
        exist_ok=args.exist_ok,
        project=str(runs_dir / 'detect'), # Explicitly set project dir to centralize runs
        close_mosaic=args.close_mosaic # Disable mosaic augmentation for final epochs
    )
    print(f"Training completed. Results saved to {runs_dir / 'detect' / args.name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLO11 Model for Compound Figure Separation")
    
    # Required/Important args
    parser.add_argument("--data", type=str, required=True, help="Path to data.yaml or dataset folder name (e.g. '04_all_classes')")
    parser.add_argument("--name", type=str, required=True, help="Name of the training run")
    
    # Model config
    parser.add_argument("--model", type=str, default="yolo11n.pt", help="YOLO model variant (yolo11n.pt, yolo11s.pt, etc)")
    
    # Training Hyperparams
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--close_mosaic", type=int, default=10, help="Epochs to disable mosaic augmentation before end")
    
    # Hardware/System
    parser.add_argument("--device", default="0", help="CUDA device or 'cpu'")
    parser.add_argument("--workers", type=int, default=8, help="Number of dataloader workers")
    parser.add_argument("--exist_ok", action="store_true", help="Overwrite existing run directory")

    args = parser.parse_args()
    
    train(args)
