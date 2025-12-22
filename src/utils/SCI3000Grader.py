import streamlit as st
import os
import sys
import json
from PIL import Image

# Set page config
st.set_page_config(layout="wide", page_title="Figure Grader")

def load_images(directory):
    valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.gif')
    if not os.path.exists(directory):
        return []
    try:
        # Sortiere alphabetisch, damit die Reihenfolge konsistent bleibt
        return sorted([
            f for f in os.listdir(directory) 
            if f.lower().endswith(valid_extensions)
        ])
    except Exception as e:
        st.error(f"Error accessing directory: {e}")
        return []

def load_labels(json_path):
    if os.path.exists(json_path):
        try:
            with open(json_path, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            return {}
    return {}

def save_labels(json_path, labels):
    with open(json_path, 'w') as f:
        json.dump(labels, f, indent=4)

def main():
    st.title("Scientific Figure Grader")

    # --- 1. Path logic ---
    default_path = os.getcwd() # Fallback
    
    if len(sys.argv) > 1:
        potential_path = sys.argv[-1]
        if os.path.exists(potential_path) and os.path.isdir(potential_path):
            default_path = os.path.abspath(potential_path)

    folder_path = st.sidebar.text_input("Image Directory", value=default_path)

    if not os.path.exists(folder_path):
        st.error(f"Directory not found: {folder_path}")
        return

    # Define JSON path
    json_path = os.path.join(folder_path, "grading_labels.json")

    # --- 2. Load data ---
    images = load_images(folder_path)
    labels = load_labels(json_path)
    
    if not images:
        st.warning("No images found in the directory.")
        return

    st.sidebar.write(f"Found **{len(images)}** images.")
    
    # Progress indicator
    labeled_count = len([img for img in images if img in labels])
    progress = labeled_count / len(images) if len(images) > 0 else 0
    st.sidebar.progress(progress)
    st.sidebar.caption(f"Progress: {labeled_count}/{len(images)} labeled")

    # --- 3. Session state management ---
    if 'image_index' not in st.session_state:
        st.session_state.image_index = 0
        
    if st.session_state.image_index >= len(images):
        st.session_state.image_index = 0

    current_image_file = images[st.session_state.image_index]
    
    # --- 4. Navigation & logic ---
    
    # Load label for current image
    current_label_data = labels.get(current_image_file, {
        "accepted": True, 
        "is_compound": False,
        "is_questionable": False
    })

    # Input container
    with st.container():
        col_img, col_ctrl = st.columns([3, 1])

        with col_ctrl:
            st.subheader("Grading")
            
            # Helper variable for unique keys
            idx = st.session_state.image_index
            
            # --- Checkboxen ---
            # Important: include the index in the key (key=f"..._{idx}")
            # so Streamlit resets state when switching images.
            
            is_accepted = st.checkbox(
                "Accepted (Usable)",
                value=current_label_data.get("accepted", True),
                key=f"chk_accepted_{idx}" 
            )
            
            is_compound = st.checkbox(
                "Is Compound Figure",
                value=current_label_data.get("is_compound", False),
                key=f"chk_compound_{idx}"
            )

            is_questionable = st.checkbox(
                "Questionable (Revisit)",
                value=current_label_data.get("is_questionable", False),
                key=f"chk_questionable_{idx}"
            )

            # Save current selection
            # Update the dictionary immediately so nothing gets lost
            labels[current_image_file] = {
                "accepted": is_accepted,
                "is_compound": is_compound,
                "is_questionable": is_questionable
            }
            save_labels(json_path, labels)

            # Visual feedback
            if is_questionable:
                st.warning("Marked for review later.")

            st.divider()
            
            # Navigation buttons
            # Use use_container_width=True for buttons (width='stretch' often isn't supported there yet)
            # Use width='stretch' for images (as requested)
            
            c1, c2 = st.columns(2)
            if c1.button("Prev", use_container_width=True): 
                st.session_state.image_index = (st.session_state.image_index - 1) % len(images)
                st.rerun()
            
            if c2.button("Next", use_container_width=True, type="primary"):
                st.session_state.image_index = (st.session_state.image_index + 1) % len(images)
                st.rerun()

            # Info box
            st.info(f"Filename:\n`{current_image_file}`")
            
            # Jump to
            new_index = st.number_input("Jump to Index", 0, len(images)-1, st.session_state.image_index)
            if new_index != st.session_state.image_index:
                st.session_state.image_index = new_index
                st.rerun()

        with col_img:
            image_path = os.path.join(folder_path, current_image_file)
            try:
                image = Image.open(image_path)
                # Fix: width='stretch' statt use_container_width=True
                st.image(image, caption=f"Image {st.session_state.image_index + 1}/{len(images)}", width="stretch")
            except Exception as e:
                st.error(f"Error loading image: {e}")

if __name__ == "__main__":
    main()