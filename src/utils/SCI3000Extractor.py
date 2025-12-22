import json
import fitz  # PyMuPDF
import os
from tqdm import tqdm

def extract_figures_and_captions(
    page_ids: list, 
    pdf_input_dir: str, 
    annotations_folder: str, 
    output_dir: str, 
    metadata_file: str = "extracted_figures_metadata.json"
):
    """
    Extracts figures and their corresponding captions from SCI-3000 PDFs based on JSON annotations.
    Includes a progress bar and skips pages that have already been processed.
    
    Args:
        page_ids (list): List of page identifiers (e.g., 'Draft-123-5').
        pdf_input_dir (str): Path to the folder containing raw PDFs.
        annotations_folder (str): Path to the folder containing JSON annotation files.
        output_dir (str): Path where images and metadata will be saved.
        metadata_file (str): Filename for the metadata registry.
        
    Returns:
        list: A list of dictionaries containing metadata for all extracted figures.
    """
    #if not list or pandas series
    if not isinstance(page_ids, list) and not hasattr(page_ids, "tolist"):
        print("[Warning] No page IDs provided for extraction.")
        return []
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    metadata_output_path = os.path.join(output_dir, metadata_file)
    
    # --- RESUME LOGIC ---
    extracted_metadata = []
    processed_page_ids = set()

    if os.path.exists(metadata_output_path):
        print(f"[Info] Loading existing metadata from {metadata_output_path}...")
        try:
            with open(metadata_output_path, 'r', encoding='utf-8') as f:
                extracted_metadata = json.load(f)
            # Create a set of already processed page IDs for fast lookup
            processed_page_ids = set(entry['page_id'] for entry in extracted_metadata)
            print(f"[Info] Found {len(processed_page_ids)} already processed pages. Skipping them.")
        except Exception as e:
            print(f"[Warning] Could not load existing metadata: {e}. Starting fresh.")

    # Filter out page_ids that are already done
    # Note: We keep the original list for structure, but we skip inside the loop or filter here.
    # Filtering here is more efficient for the loop, but we need to group by PDF ID efficiently.
    pages_to_process = [pid for pid in page_ids if pid not in processed_page_ids]
    
    if not pages_to_process:
        print("All pages have already been processed.")
        return extracted_metadata

    # Group pages by PDF ID again (only for the remaining pages)
    pdf_map = {}
    for pid in pages_to_process:
        # Assumes format: 'DocID-PageNr', e.g., 'Draft-2023-5' -> 'Draft-2023'
        pdf_id = '-'.join(pid.split('-')[:-1])
        if pdf_id not in pdf_map:
            pdf_map[pdf_id] = []
        pdf_map[pdf_id].append(pid)

    print(f"Starting extraction for {len(pages_to_process)} new pages from {len(pdf_map)} PDFs...")
    
    # --- PROCESSING LOOP WITH PROGRESS BAR ---
    # We iterate over PDFs to open files efficiently, but update the progress bar per page
    
    with tqdm(total=len(pages_to_process), desc="Extracting Pages", unit="page") as pbar:
        
        for pdf_id, current_pdf_pages in pdf_map.items():
            pdf_path = os.path.join(pdf_input_dir, f"{pdf_id}.pdf")
            
            # Check if PDF exists
            if not os.path.exists(pdf_path):
                # If PDF is missing, we must update the pbar for all its pages to keep sync
                # print(f"[Warning] PDF not found: {pdf_path}")
                pbar.update(len(current_pdf_pages))
                continue

            try:
                doc = fitz.open(pdf_path)
            except Exception as e:
                print(f"[Error] Could not open {pdf_path}: {e}")
                pbar.update(len(current_pdf_pages))
                continue

            for page_id in current_pdf_pages:
                
                # --- PROCESS PAGE ---
                json_path = os.path.join(annotations_folder, f"{page_id}.json")
                if not os.path.exists(json_path):
                    pbar.update(1)
                    continue
                    
                try:
                    with open(json_path, 'r') as f:
                        data = json.load(f)
                
                    # Load page (1-based index in ID -> 0-based index in PyMuPDF)
                    page_nr = int(page_id.split('-')[-1])
                    page = doc.load_page(page_nr - 1)

                    # Coordinate scaling
                    pdf_w = page.rect.width
                    pdf_h = page.rect.height
                    json_w = data.get("canvasWidth", pdf_w) 
                    json_h = data.get("canvasHeight", pdf_h)
                    scale_x = pdf_w / json_w if json_w else 1
                    scale_y = pdf_h / json_h if json_h else 1
                    
                    # --- PASS 1: Index Captions ---
                    caption_map = {} 
                    for anno in data.get("annotations", []):
                        body_list = anno.get("body", [])
                        if not isinstance(body_list, list): continue

                        parent_id = None
                        is_caption = False
                        for item in body_list:
                            if item.get("value") == "Caption": is_caption = True
                            if item.get("purpose") == "parent": parent_id = item.get("value")
                        
                        if is_caption and parent_id:
                            selector = anno.get("target", {}).get("selector", {}).get("value", "")
                            if "pixel:" in selector:
                                coords_str = selector.split("pixel:")[1]
                                cx, cy, cw, ch = map(float, coords_str.split(","))
                                caption_map[parent_id] = fitz.Rect(
                                    cx * scale_x, cy * scale_y, (cx + cw) * scale_x, (cy + ch) * scale_y
                                )

                    # --- PASS 2: Extract Figures ---
                    figure_counter = 0 
                    
                    for anno in data.get("annotations", []):
                        body_list = anno.get("body", [])
                        if not isinstance(body_list, list): continue
                        
                        if any(item.get("value") == "Figure" for item in body_list):
                            fig_anno_id = anno.get("id")
                            selector = anno.get("target", {}).get("selector", {}).get("value", "")
                            if "pixel:" not in selector: continue
                            
                            coords_str = selector.split("pixel:")[1]
                            x, y, w, h = map(float, coords_str.split(","))
                            rect_points = fitz.Rect(x * scale_x, y * scale_y, (x + w) * scale_x, (y + h) * scale_y)

                            # Extract Image (High Res)
                            zoom = 300 / 72
                            mat = fitz.Matrix(zoom, zoom)
                            try:
                                pix = page.get_pixmap(matrix=mat, clip=rect_points)
                                out_filename = f"{page_id}-fig-{figure_counter}.png"
                                pix.save(os.path.join(output_dir, out_filename))
                            except Exception as e:
                                print(f"[Error] Save failed for {out_filename}: {e}")
                                continue
                            
                            # Extract Caption
                            caption_text = ""
                            if fig_anno_id in caption_map:
                                caption_text = page.get_text("text", clip=caption_map[fig_anno_id]).strip()
                                caption_text = caption_text.replace('\n', ' ').replace('\r', '')

                            meta_entry = {
                                "pdf_id": pdf_id,
                                "page_id": page_id,
                                "figure_id": figure_counter,
                                "original_annotation_id": fig_anno_id,
                                "image_filename": out_filename,
                                "caption": caption_text,
                                "bbox_pdf_coords": [rect_points.x0, rect_points.y0, rect_points.x1, rect_points.y1]
                            }
                            extracted_metadata.append(meta_entry)
                            figure_counter += 1
                
                except Exception as e:
                    #Catch-all for page processing errors
                    print(f"[Error] Processing failed for {page_id}: {e}")
                
                # Update progress bar
                pbar.update(1)
            
            doc.close()
            
            # OPTIONAL: Save metadata incrementally after each PDF (safer for large jobs)
            # Only do this if speed is not critical, otherwise save at the end
            with open(metadata_output_path, 'w', encoding='utf-8') as f:
                json.dump(extracted_metadata, f, indent=4, ensure_ascii=False)

    print(f"Extraction complete. Metadata saved to {metadata_output_path}")
    return extracted_metadata