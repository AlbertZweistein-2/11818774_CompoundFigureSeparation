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

@st.cache_resource
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
    
    selected_model_name = None

    if not all_model_names:
        st.sidebar.warning("No models found locally or in standard list.")
    else:
        # 3. Search for a model
        display_names = ["Select a model..."]
        
        for name in all_model_names:
            if (MODELS_DIR / name).exists():
                display_names.append(name)
            else:
                display_names.append(f"{name} (Download from HF)")
        
        # Map display name back to filename
        display_map = {"Select a model...": None}
        for name in all_model_names:
            if (MODELS_DIR / name).exists():
                display_map[name] = name
            else:
                display_map[f"{name} (Download from HF)"] = name
        
        selected_display = st.sidebar.selectbox("Select Model (Type to search)", display_names, index=0)
        selected_model_name = display_map[selected_display]
        
        # Immediate Download / Availability Check
        if selected_model_name:
             # Check if we need to download (doesn't exist locally)
             if not (MODELS_DIR / selected_model_name).exists():
                 # This downloads the model and config
                 if ensure_model_available(selected_model_name):
                     st.sidebar.success(f"Model downloaded!")
             
             # Now show config
             yaml_path = (MODELS_DIR / selected_model_name).with_suffix(".yaml")
             if yaml_path.exists():
                try:
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
                except Exception:
                    st.sidebar.warning("Could not read config file.")
             else:
                 st.sidebar.info("No configuration file found.")

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
        # Button
        run_analysis = st.button("Separation Analysis", type="primary")

        # Display Input
        col1, col2 = st.columns(2)
        with col1:
             img_placeholder = st.empty()
             img_placeholder.image(image, caption=f"Original: {original_image_name}", width='stretch')

        # Run Inference
        if run_analysis:

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
                        # Fix: use_column_width deprecated -> use_container_width
                        st.image(res_image, caption="Detected Subplots", width='stretch')
                    
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
                    
                    # Show Ground Truth (if available)
                    if input_source == "Sample from Test Set" and 'image_path' in locals():
                         # Derive label path: replace 'images' parent with 'labels' and extension with .txt
                         # Path structure: .../images/test/file.jpg -> .../labels/test/file.txt
                         try:
                             # We need to handle potential directory structures. 
                             # Safest way: go up 2 levels from image file (test -> images -> root of subset) 
                             # then go into labels -> test -> file.txt
                             # BUT: "same level as the images directory" might mean .../dataset/images and .../dataset/labels
                             
                             # The `image_path` is a Path object.
                             # Assumption: `image_path` = .../images/test/filename.jpg
                             # Target: .../labels/test/filename.txt
                             
                             # Check if 'images' is in the path parts
                             parts = list(image_path.parts)
                             if 'images' in parts:
                                 # Replace right-most 'images' with 'labels'
                                 # (handle cases where 'images' might appear elsewhere? usually not)
                                 # Let's find index of 'images' relative to end to be safe, or just replace last occurence
                                 
                                 # Using pathlib replacement relative to parent
                                 # parent of image_path is .../images/test
                                 # grand_parent is .../images
                                 # if standard structure:
                                 # .../images/test/img.jpg
                                 
                                 label_filename = image_path.stem + ".txt"
                                 
                                 # Strategy: iterate parents to find 'images' folder and swap to 'labels'
                                 # Actually, straightforward replacement in the string path might be easiest if structure is strict.
                                 # Let's try pathlib swap.
                                 
                                 # Reconstruct path replacing 'images' with 'labels'
                                 # We assume the directory name is exactly "images"
                                 
                                 label_path = Path(str(image_path).replace("/images/", "/labels/")).with_suffix(".txt")
                                 
                                 if label_path.exists():
                                     st.subheader("Ground Truth (True Labels)")
                                     
                                     gt_data = []
                                     
                                     # Prepare to draw GT boxes on a copy of the original image
                                     # Need original PIL image back to numpy or draw on PIL
                                     # Since res_image is already a result, let's reload or copy original 'image'
                                     gt_image = image.copy()
                                     if gt_image.mode != "RGB":
                                         gt_image = gt_image.convert("RGB")
                                     gt_draw_np = np.array(gt_image)
                                     
                                     # If it's RGB (PIL default), OpenCV expects BGR usually, but we can just use BGR colors and keep it as is if we display with st.image
                                     # However, st.image expects RGB. cv2.rectangle works on the array.
                                     # Let's ensure it's contiguous array
                                     gt_draw_np = np.ascontiguousarray(gt_draw_np)
                                     
                                     height, width, _ = gt_draw_np.shape
                                     
                                     with open(label_path, 'r') as f:
                                         lines = f.readlines()
                                         
                                     for line in lines:
                                         parts = line.strip().split()
                                         if len(parts) >= 5:
                                             cls_id = int(parts[0])
                                             # Use model names map if available, else ID
                                             cls_name = model.names.get(cls_id, str(cls_id)) if hasattr(model, 'names') else str(cls_id)
                                             
                                             # BBox is xywh normalized
                                             cx, cy, w, h = [float(x) for x in parts[1:5]]
                                             
                                             # Convert to absolute xyxy for drawing
                                             # x_c, y_c, w, h -> x1, y1, x2, y2
                                             x1 = int((cx - w/2) * width)
                                             y1 = int((cy - h/2) * height)
                                             x2 = int((cx + w/2) * width)
                                             y2 = int((cy + h/2) * height)
                                             
                                             gt_data.append({
                                                 "Class": cls_name,
                                                 "BBox (xywh norm)": [cx, cy, w, h]
                                             })
                                             
                                             # Draw box (Green for GT)
                                             # Note: PIL np array is RGB. Green is (0, 255, 0)
                                             cv2.rectangle(gt_draw_np, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                             
                                             # Draw Label
                                             label_text = f"{cls_name}"
                                             t_size = cv2.getTextSize(label_text, 0, fontScale=0.5, thickness=1)[0]
                                             c2 = x1 + t_size[0], y1 - t_size[1] - 3
                                             cv2.rectangle(gt_draw_np, (x1, y1), c2, (0, 255, 0), -1, cv2.LINE_AA)  # filled
                                             cv2.putText(gt_draw_np, label_text, (x1, y1 - 2), 0, 0.5, (0, 0, 0), thickness=1, lineType=cv2.LINE_AA)

                                             cv2.putText(gt_draw_np, label_text, (x1, y1 - 2), 0, 0.5, (0, 0, 0), thickness=1, lineType=cv2.LINE_AA)

                                     # Show GT Image in the LEFT column (Replacing Original)
                                     img_placeholder.image(gt_draw_np, caption="Ground Truth Annotation", width='stretch')
                                     
                                     if gt_data:
                                         st.dataframe(gt_data)
                                     else:
                                         st.info("Label file found but empty.")
                                 else:
                                      pass
                         except Exception as e:
                             st.error(f"Error loading labels: {e}")

    elif image is not None and not selected_model_name:
        st.image(image, caption=f"Original: {original_image_name}", width='stretch')
        st.warning("Please select a model to run inference.")

if __name__ == "__main__":
    main()
