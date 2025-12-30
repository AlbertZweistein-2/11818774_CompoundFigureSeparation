import streamlit as st
import os
import sys
import json
from PIL import Image, ImageDraw, ImageFont

# Set page config
st.set_page_config(layout="wide", page_title="MedICaT Grader (Fast)")

# --- 1. Load data ---

@st.cache_data
def load_medicat_metadata(jsonl_path):
    """
    Load the JSONL metadata and build a mapping.
    Key: {pdf_hash}_{fig_uri}
    """
    data_map = {}
    if not os.path.exists(jsonl_path):
        return None
    
    try:
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                entry = json.loads(line)
                pdf_hash = entry.get('pdf_hash', '')
                fig_uri = entry.get('fig_uri', '')
                
                if pdf_hash and fig_uri:
                    constructed_filename = f"{pdf_hash}_{fig_uri}"
                    data_map[constructed_filename] = entry
        return data_map
    except Exception as e:
        st.error(f"Error loading JSONL: {e}")
        return None

def load_grading_status(json_path):
    if os.path.exists(json_path):
        try:
            with open(json_path, 'r') as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_grading_status(json_path, data):
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=4)

# --- 2. High-Performance Visualisierung (PIL) ---

def draw_annotations(image, metadata):
    """
    Draw polygons directly onto a PIL Image (server-side).
    Much faster than Plotly.
    """
    # Copy to avoid mutating the original (avoids caching issues)
    img_copy = image.copy().convert("RGB") 
    draw = ImageDraw.Draw(img_copy)
    
    if metadata and "subfigures" in metadata:
        for subfig in metadata["subfigures"]:
            points = subfig.get("points", [])
            label = subfig.get("label", "?")
            color_name = subfig.get("color", "cyan")
            
            # Fallback for colors PIL may not recognize
            if not isinstance(color_name, str): 
                color_name = "cyan"
            # MedICaT color mapping could be added here if needed (PIL is usually tolerant)
            
            if points:
                # PIL Polygon erwartet Liste von Tupeln: [(x,y), (x,y), ...]
                poly_points = [tuple(p) for p in points]
                
                try:
                    # Polygon zeichnen (Outline)
                    draw.polygon(poly_points, outline=color_name, width=5)
                    
                    # Optional: Label Text zeichnen (ganz simpel oben links vom Polygon)
                    # Wir nutzen default font, ist klein aber schnell
                    start_x, start_y = poly_points[0]
                    # Small background for readability
                    draw.rectangle([start_x, start_y-15, start_x+30, start_y], fill=color_name)
                    draw.text((start_x+2, start_y-12), str(label), fill="black")
                except Exception:
                    # Fallback if the color is invalid
                    draw.polygon(poly_points, outline="red", width=5)

    return img_copy

# --- 3. Main App ---

def main():
    st.title("MedICaT Grader (Fast Mode)")

    # --- Setup paths ---
    default_base_dir = os.getcwd()
    if len(sys.argv) > 1:
        potential_path = sys.argv[-1]
        if os.path.exists(potential_path) and os.path.isdir(potential_path):
            default_base_dir = os.path.abspath(potential_path)
            
    base_dir = st.sidebar.text_input("MedICaT Root Directory", value=default_base_dir)
    
    jsonl_path = os.path.join(base_dir, "subcaptions_public.jsonl")
    figures_dir = os.path.join(base_dir, "figures")
    grading_file = os.path.join(base_dir, "medicat_grading.json")

    # Checks
    if not os.path.exists(jsonl_path):
        st.error(f"Metadata file not found: `{jsonl_path}`")
        return
    if not os.path.exists(figures_dir):
        st.error(f"Figures directory not found: `{figures_dir}`")
        return

    # --- Load data ---
    with st.spinner("Loading Metadata..."):
        metadata_map = load_medicat_metadata(jsonl_path)
    
    if not metadata_map:
        return

    all_files_in_dir = sorted(os.listdir(figures_dir))
    available_images = [f for f in all_files_in_dir if f in metadata_map]
    
    if not available_images:
        st.warning("No matching images found.")
        return

    st.sidebar.write(f"Images: **{len(available_images)}**")
    
    # Grading Status
    grading_status = load_grading_status(grading_file)
    labeled_count = len(grading_status)
    st.sidebar.progress(labeled_count / len(available_images))
    st.sidebar.caption(f"Progress: {labeled_count}/{len(available_images)}")

    # --- Session State ---
    if 'medicat_index' not in st.session_state:
        st.session_state.medicat_index = 0
    
    if st.session_state.medicat_index >= len(available_images):
        st.session_state.medicat_index = 0

    current_filename = available_images[st.session_state.medicat_index]
    current_meta = metadata_map[current_filename]
    
    current_status = grading_status.get(current_filename, {
        "accepted": False,      
        "label_quality": "ok", 
        "comment": ""
    })

    # --- UI Layout ---
    col_plot, col_ctrl = st.columns([3, 1])

    with col_ctrl:
        st.subheader("Grading")
        
        # Unique key generation (fixes Streamlit state update issues)
        idx = st.session_state.medicat_index
        
        is_accepted = st.checkbox(
            "Accept for Training",
            value=current_status.get("accepted", False),
            key=f"chk_acc_{idx}" 
        )
        
        st.write("**Label Quality:**")
        quality_opts = ["ok", "bad_alignment", "missing_subfigures", "false_positives", "complex_overlap"]
        curr_qual = current_status.get("label_quality", "ok")
        try:
            qual_idx = quality_opts.index(curr_qual)
        except:
            qual_idx = 0
            
        quality = st.selectbox(
            "Issue Type", 
            quality_opts, 
            index=qual_idx,
            key=f"sel_qual_{idx}"
        )
        
        comment = st.text_input(
            "Comment", 
            value=current_status.get("comment", ""),
            key=f"txt_comm_{idx}"
        )

        # Auto-Save
        grading_status[current_filename] = {
            "accepted": is_accepted,
            "label_quality": quality,
            "comment": comment
        }
        save_grading_status(grading_file, grading_status)
        
        st.divider()

        # Navigation
        c1, c2 = st.columns(2)
        if c1.button("Prev", use_container_width=True):
            st.session_state.medicat_index = (st.session_state.medicat_index - 1) % len(available_images)
            st.rerun()
            
        if c2.button("Next", use_container_width=True, type="primary"):
            st.session_state.medicat_index = (st.session_state.medicat_index + 1) % len(available_images)
            st.rerun()
            
        new_idx = st.number_input("Jump to Index", 0, len(available_images)-1, st.session_state.medicat_index)
        if new_idx != st.session_state.medicat_index:
            st.session_state.medicat_index = new_idx
            st.rerun()
            
        st.caption(f"File: `{current_filename}`")

    with col_plot:
        img_path = os.path.join(figures_dir, current_filename)
        try:
            # 1. Load image
            pil_image = Image.open(img_path)
            
            # 2. Draw (server-side)
            annotated_image = draw_annotations(pil_image, current_meta)
            
            # 3. Display (static, width='stretch' as requested)
            st.image(annotated_image, width="stretch")
            
        except Exception as e:
            st.error(f"Error loading image: {e}")

if __name__ == "__main__":
    main()