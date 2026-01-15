import streamlit as st
import cv2
from PIL import Image
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import sys
import yaml
try:
    from huggingface_hub import hf_hub_download
except ImportError:
    st.error("huggingface_hub not installed. Please run `pip install huggingface_hub`")

# Add src to path to allow importing utils if needed
sys.path.append(str(Path(__file__).parent))

# Configuration
PAGE_TITLE = "Compound Figure Separation Demo"
PAGE_ICON = "🧩"
LAYOUT = "wide"
HF_MODEL_REPO = "TobiPoni/BaseCompoundFigureSeparator"

# Paths
REPO_ROOT = Path(__file__).parent.parent
MODELS_DIR = REPO_ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True) # Ensure models dir exists

# Standard Models List (Available on HuggingFace)
STANDARD_MODELS = [
    "04_all_classes_Ymedium_1280_baseline.pt",
    "04_all_classes_Ysmall_1280_baseline.pt", 
    "04_all_classes_Ysmall_960_baseline.pt",
    "05_selected_classes_Ynano_1280_baseline.pt",
    "06_compound_chart_splitter_Ynano_1280_baseline.pt"
]

st.set_page_config(page_title=PAGE_TITLE, page_icon=PAGE_ICON, layout=LAYOUT)

def load_model(model_path):
    """Loads a YOLO model."""
    return YOLO(model_path)

def ensure_model_available(filename):
    """Checks if model exists locally, else downloads from HF."""
    local_path = MODELS_DIR / filename
    
    # Check if it's a standard model or a local custom one
    is_standard = filename in STANDARD_MODELS
    
    if not local_path.exists():
        if is_standard:
            with st.spinner(f"Downloading {filename} from HuggingFace ({HF_MODEL_REPO})..."):
                try:
                    # Download .pt
                    downloaded_pt = hf_hub_download(
                        repo_id=HF_MODEL_REPO,
                        filename=filename,
                        local_dir=MODELS_DIR,
                        local_dir_use_symlinks=False
                    )
                    
                    # Try to download config .yaml too (if it exists)
                    config_name = str(Path(filename).with_suffix('.yaml'))
                    try:
                        hf_hub_download(
                            repo_id=HF_MODEL_REPO,
                            filename=config_name,
                            local_dir=MODELS_DIR,
                            local_dir_use_symlinks=False
                        )
                    except Exception:
                        pass # Config might not exist, that's fine
                        
                    st.success(f"Downloaded {filename}!")
                except Exception as e:
                    st.error(f"Failed to download model: {e}")
                    return None
        else:
            st.error(f"Model {filename} not found locally and is not a standard model.")
            return None
            
    return local_path

def main():
    st.title(f"{PAGE_ICON} {PAGE_TITLE}")
    st.markdown("Upload a compound figure or select a sample to detect subplots.")

    # Sidebar: Model Selection
    st.sidebar.header("Configuration")
    
    # 1. Gather Local Models
    local_files = list(MODELS_DIR.glob("*.pt"))
    local_filenames = [p.name for p in local_files]
    
    # 2. Merge with Standard Models (Deduplicated)
    # Union of local found files and the known standard list
    all_model_names = sorted(list(set(local_filenames + STANDARD_MODELS)))
    
    if not all_model_names:
        st.sidebar.warning("No models found locally or in standard list.")
        selected_model_name = None
    else:
        # 3. Search for a model
        # Add visual indicator for (Local) vs (Download Required)
        display_names = []
        for name in all_model_names:
            if (MODELS_DIR / name).exists():
                display_names.append(name)
            else:
                display_names.append(f"{name} (Download from HF)")
        
        # Map display name back to filename
        display_map = dict(zip(display_names, all_model_names))
        
        selected_display = st.sidebar.selectbox("Select Model (Type to search)", display_names)
        selected_model_name = display_map[selected_display]
        
        # Ensure availability (Download if needed)
        # We only check/download when they actually pick it, but for UI responsiveness 
        # let's just resolve path. We check existence properly during inference or explicit load.
        
        # Display Configuration from YAML if available (Locally)
        # If not local, we might not have the YAML yet.
        yaml_path = (MODELS_DIR / selected_model_name).with_suffix(".yaml")
        
        if yaml_path.exists():
            with open(yaml_path, 'r') as f:
                config = yaml.safe_load(f)
            
            st.sidebar.subheader("Model Configuration")
            keys_to_show = {
                "model": "Model", 
                "epochs": "Epochs", 
                "data": "Dataset", 
                "imgsz": "Image Size", 
                "batch": "Batch Size", 
                "close_mosaic": "Close Mosaic"
            }
            for key, label in keys_to_show.items():
                if key in config:
                    st.sidebar.text(f"{label}: {config[key]}")
        elif selected_model_name in STANDARD_MODELS and not (MODELS_DIR / selected_model_name).exists():
             st.sidebar.info("Select to download model and view config.")

    # Sidebar: Confidence Threshold
    conf_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.25, 0.05)
    
    # Main Content: Input
    st.subheader("Input Image")
    
    input_source = st.radio("Select Input Source", ["Sample from Test Set", "Upload Image"])
    
    image = None
    original_image_name = "Upload"
    
    if input_source == "Upload Image":
        uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            original_image_name = uploaded_file.name
            
    elif input_source == "Sample from Test Set":
        # Dynamic Dataset Selection
        potential_datasets = {}
        
        # 1. Full Datasets (Only if downloaded)
        potential_datasets["Full: 04 (All Classes)"] = REPO_ROOT / "dataset/04_all_classes/images/test"
        potential_datasets["Full: 05 (Selected Classes)"] = REPO_ROOT / "dataset/05_selected_classes/images/test"
        potential_datasets["Full: 06 (Compound Splitter)"] = REPO_ROOT / "dataset/06_compound_chart_splitter/images/test"
        
        # 2. Demo Assets (Always available in repo)
        potential_datasets["Demo: 04 (All Classes)"] = REPO_ROOT / "docs/assets/04_all_classes_demo/images/test"
        potential_datasets["Demo: 05 (Selected Classes)"] = REPO_ROOT / "docs/assets/05_selected_classes_demo/images/test"
        potential_datasets["Demo: 06 (Compound Splitter)"] = REPO_ROOT / "docs/assets/06_compound_chart_splitter_demo/images/test"
        
        # 3. Extra Examples (SCI-3000)
        potential_datasets["Example: SCI-3000 (Real)"] = REPO_ROOT / "docs/assets/SCI-3000_examples/real_compound"
        potential_datasets["Example: SCI-3000 (Synth)"] = REPO_ROOT / "docs/assets/SCI-3000_examples/synthetic_compound"

        # Filter: Only keep existing directories
        dataset_options = {k: v for k, v in potential_datasets.items() if v.exists()}
        
        if dataset_options:
            selected_ds_label = st.selectbox("Select Test Dataset", list(dataset_options.keys()), index=0)
            test_dir = dataset_options[selected_ds_label]
            
            # Support both jpg/png and nested searches if needed (though we keep it flat for now)
            # Glob for images
            sample_images = sorted(list(test_dir.glob("*.jpg")) + list(test_dir.glob("*.png")) + list(test_dir.glob("*.jpeg")))
            
            if sample_images:
                sample_names = [p.name for p in sample_images]
                selected_sample = st.selectbox("Choose a sample image", sample_names)
                image_path = test_dir / selected_sample
                image = Image.open(image_path)
                original_image_name = selected_sample
            else:
                st.warning(f"No images found in {test_dir}.")
        else:
            st.error("No datasets found! Please download the dataset `python src/prepare.py` or ensure `docs/assets` are present.")

    # Inference & Visualization
    if image is not None and selected_model_name:
        # Display Input
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption=f"Original: {original_image_name}", use_container_width=True)

        # Run Inference
        if st.button("Separation Analysis", type="primary"):
            # Ensure model is ready (download if needed)
            model_path = ensure_model_available(selected_model_name)
            
            if model_path:
                with st.spinner("Analyzing..."):
                    model = load_model(model_path)
                    
                    # Inference
                    results = model.predict(image, conf=conf_threshold)
                    
                    # Visualize
                    res_plotted = results[0].plot()
                    res_image = Image.fromarray(res_plotted[..., ::-1]) 
                    
                    with col2:
                        st.image(res_image, caption="Detected Subplots", use_container_width=True)
                    
                    # Show detected boxes/classes
                    st.subheader("Detections")
                    boxes = results[0].boxes
                    if len(boxes) > 0:
                        data = []
                        for box in boxes:
                            data.append({
                                "Class": results[0].names[int(box.cls)],
                                "Confidence": float(box.conf),
                                "BBox (xyxy)": [round(x, 1) for x in box.xyxy[0].tolist()]
                            })
                        st.dataframe(data)
                    else:
                        st.info("No objects detected.")

    elif image is not None and not selected_model_name:
        st.image(image, caption=f"Original: {original_image_name}", use_container_width=True)
        st.warning("Please select a model to run inference.")

if __name__ == "__main__":
    main()
