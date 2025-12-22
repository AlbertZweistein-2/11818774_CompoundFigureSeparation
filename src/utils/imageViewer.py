import streamlit as st
import os
import sys
from PIL import Image

# Set page config
st.set_page_config(layout="wide", page_title="Image Viewer")

def load_images(directory):
    valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.gif')
    if not os.path.exists(directory):
        return []
    try:
        return sorted([
            f for f in os.listdir(directory) 
            if f.lower().endswith(valid_extensions)
        ])
    except Exception as e:
        st.error(f"Error accessing directory: {e}")
        return []

def main():
    st.title("Dataset Image Viewer")

    # Determine directory
    # Default path
    default_path = os.path.join(os.getcwd(), "SciCap_data", "scicap_data", "SciCap-No-Subfig-Img", "train")
    
    # Check for command line argument (passed after --)
    if len(sys.argv) > 1:
        potential_path = sys.argv[1]
        if os.path.exists(potential_path):
            default_path = os.path.abspath(potential_path)
    
    folder_path = st.sidebar.text_input("Image Directory", value=default_path)

    if not os.path.exists(folder_path):
        st.error(f"Directory not found: {folder_path}")
        return

    images = load_images(folder_path)
    
    if not images:
        st.warning("No images found in the directory.")
        return

    st.sidebar.write(f"Found {len(images)} images.")
    
    # Session state for index
    if 'image_index' not in st.session_state:
        st.session_state.image_index = 0
        
    # Ensure index is valid if image list changes
    if st.session_state.image_index >= len(images):
        st.session_state.image_index = 0

    # Navigation
    col1, col2 = st.sidebar.columns(2)
    
    if col1.button("Previous"):
        st.session_state.image_index = (st.session_state.image_index - 1) % len(images)
        st.rerun()
        
    if col2.button("Next"):
        st.session_state.image_index = (st.session_state.image_index + 1) % len(images)
        st.rerun()

    # Direct jump
    index = st.sidebar.number_input("Go to Index", min_value=0, max_value=len(images)-1, value=st.session_state.image_index)
    if index != st.session_state.image_index:
        st.session_state.image_index = index
        st.rerun()

    # Display current image
    current_image_file = images[st.session_state.image_index]
    image_path = os.path.join(folder_path, current_image_file)
    
    st.header(f"Image {st.session_state.image_index + 1}/{len(images)}: {current_image_file}")
    
    try:
        image = Image.open(image_path)
        st.image(image, use_container_width=False)
    except Exception as e:
        st.error(f"Error loading image: {e}")

if __name__ == "__main__":
    main()

