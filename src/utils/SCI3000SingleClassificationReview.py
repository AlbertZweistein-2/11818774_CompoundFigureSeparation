"""Streamlit app for single-image classification into SCI-3000 label set.

Usage:
- Loads images + existing labels from single_labels.json in the selected folder.
- Choose one of the predefined classes; saves immediately back to JSON.
- Navigation via Prev/Next or jump-to index.
"""

import json
import os

from PIL import Image
import streamlit as st

st.set_page_config(layout="wide", page_title="Single Label Reviewer (Advanced)")

# Available categories
CLASSES = ["Chart", "Illustration", "Image", "Table", "Other"]

def main():
    st.title("Advanced Figure Classifier")
    st.markdown("Classify images into precise categories. Labels are saved to single_labels.json in the selected folder.")

    # Path setup
    default_dir = "../../dataset/raw/SCI-3000-Singles"  # Adjust if needed
    
    # Sidebar options
    base_dir = st.sidebar.text_input("Directory", default_dir)
    json_path = os.path.join(base_dir, "single_labels.json")

    if not os.path.exists(json_path):
        st.error(f"JSON not found: {json_path}")
        return

    # Load data
    with open(json_path, 'r') as f:
        labels = json.load(f)
    
    # List of images
    all_files = sorted(list(labels.keys()))
    
    if not all_files:
        st.warning("No images found in the JSON.")
        return
    
    # Session state
    if 'idx' not in st.session_state:
        st.session_state.idx = 0
    # Bounds check if the directory changes
    if st.session_state.idx >= len(all_files):
        st.session_state.idx = 0
    
    current_file = all_files[st.session_state.idx]
    
    # --- LABEL LOGIC ---
    # Load existing label; keep older labels but allow overwrite in UI
    current_label = labels.get(current_file, "Other")
    
    # Fallback: if the JSON contains a label not in our list
    if current_label not in CLASSES:
        # Default to 'Other' (user can still change it in the UI)
        current_label_idx = CLASSES.index("Other")
    else:
        current_label_idx = CLASSES.index(current_label)

    # GUI Layout
    col_img, col_ctrl = st.columns([3, 1])
    
    with col_ctrl:
        st.subheader("Classification")
        st.caption(f"Current label in file: **{current_label}**")
        
        # --- RADIO BUTTONS ---
        # Core control: selection from 5 classes
        new_label = st.radio(
            "Choose a category:",
            CLASSES, 
            index=current_label_idx,
            key=f"rad_{st.session_state.idx}"
        )
        
        # Show definitions as help (expandable)
        with st.expander("Help on classes"):
            st.markdown("""
            * **Chart:** Plots with axes (bar, line, scatter, pie).
            * **Illustration:** Schematics, molecules, flowcharts, architecture.
            * **Image:** Photos, microscopy, MRI, X-ray, screenshots.
            * **Table:** Rows and columns with text/numbers.
            * **Other:** Text blocks, empty images, errors.
            """)

        # Save on change
        if new_label != current_label:
            labels[current_file] = new_label
            with open(json_path, 'w') as f:
                json.dump(labels, f, indent=4)
            st.toast(f"Saved as {new_label}.")

        st.divider()
        
        # --- NAVIGATION ---
        c1, c2 = st.columns(2)
        if c1.button("Prev", use_container_width=True):
            st.session_state.idx = (st.session_state.idx - 1) % len(all_files)
            st.rerun()
            
        if c2.button("Next", type="primary", use_container_width=True):
            st.session_state.idx = (st.session_state.idx + 1) % len(all_files)
            st.rerun()

        # --- GOTO FEATURE ---
        st.divider()
        new_idx = st.number_input(
            "Jump to image index",
            min_value=0, 
            max_value=len(all_files)-1, 
            value=st.session_state.idx,
            step=1
        )
        
        if new_idx != st.session_state.idx:
            st.session_state.idx = new_idx
            st.rerun()

        # Info boxes
        st.progress((st.session_state.idx + 1) / len(all_files))
        st.caption(f"Progress: {st.session_state.idx + 1}/{len(all_files)}")
        st.text(f"Filename:\n{current_file}")

    with col_img:
        try:
            img_path = os.path.join(base_dir, current_file)
            img = Image.open(img_path)
            st.image(img, caption=current_file, use_container_width=True) 
        except Exception as e:
            st.error(f"Unable to load image: {e}")

if __name__ == "__main__":
    main()