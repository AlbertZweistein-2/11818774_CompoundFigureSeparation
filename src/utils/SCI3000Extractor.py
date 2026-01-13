"""PDF figure extraction utilities for SCI-3000 dataset."""

import json
import os

import fitz  # PyMuPDF
from tqdm import tqdm


def extract_figures_and_captions(
    page_ids: list,
    pdf_input_dir: str,
    annotations_folder: str,
    output_dir: str,
    metadata_file: str = "extracted_figures_metadata.json",
):
    """Extract figures + captions from PDFs using JSON annotations.

    Args:
        page_ids: List of page identifiers, e.g., "Draft-123-5" (DocID-PageNr).
        pdf_input_dir: Folder containing the raw PDFs.
        annotations_folder: Folder with JSON annotation files (Label Studio export).
        output_dir: Where extracted images and metadata will be saved.
        metadata_file: Filename for the metadata registry (for resume).

    Returns:
        List of metadata dicts for all extracted figures.
    """

    # Accept list-like (e.g., pandas Series via tolist)
    if not isinstance(page_ids, list) and not hasattr(page_ids, "tolist"):
        print("[Warning] No page IDs provided for extraction.")
        return []

    os.makedirs(output_dir, exist_ok=True)
    metadata_output_path = os.path.join(output_dir, metadata_file)

    # --- Resume logic: load already processed pages ---
    extracted_metadata = []
    processed_page_ids = set()

    if os.path.exists(metadata_output_path):
        print(f"[Info] Loading existing metadata from {metadata_output_path}...")
        try:
            with open(metadata_output_path, 'r', encoding='utf-8') as f:
                extracted_metadata = json.load(f)
            processed_page_ids = set(entry['page_id'] for entry in extracted_metadata)
            print(f"[Info] Found {len(processed_page_ids)} already processed pages. Skipping them.")
        except Exception as e:
            print(f"[Warning] Could not load existing metadata: {e}. Starting fresh.")

    pages_to_process = [pid for pid in page_ids if pid not in processed_page_ids]

    if not pages_to_process:
        print("All pages have already been processed.")
        return extracted_metadata

    # Group pages by PDF ID for efficient loading
    pdf_map = {}
    for pid in pages_to_process:
        pdf_id = '-'.join(pid.split('-')[:-1])  # e.g., Draft-2023-5 -> Draft-2023
        if pdf_id not in pdf_map:
            pdf_map[pdf_id] = []
        pdf_map[pdf_id].append(pid)

    print(f"Starting extraction for {len(pages_to_process)} new pages from {len(pdf_map)} PDFs...")

    # Process PDFs with progress bar (update per page)
    with tqdm(total=len(pages_to_process), desc="Extracting Pages", unit="page") as pbar:

        for pdf_id, current_pdf_pages in pdf_map.items():
            pdf_path = os.path.join(pdf_input_dir, f"{pdf_id}.pdf")

            if not os.path.exists(pdf_path):
                pbar.update(len(current_pdf_pages))
                continue

            try:
                doc = fitz.open(pdf_path)
            except Exception as e:
                print(f"[Error] Could not open {pdf_path}: {e}")
                pbar.update(len(current_pdf_pages))
                continue

            for page_id in current_pdf_pages:

                json_path = os.path.join(annotations_folder, f"{page_id}.json")
                if not os.path.exists(json_path):
                    pbar.update(1)
                    continue

                try:
                    with open(json_path, 'r') as f:
                        data = json.load(f)

                    # Convert 1-based page id to 0-based page index
                    page_nr = int(page_id.split('-')[-1])
                    page = doc.load_page(page_nr - 1)

                    # Coordinate scaling from annotation canvas to PDF pixels
                    pdf_w = page.rect.width
                    pdf_h = page.rect.height
                    json_w = data.get("canvasWidth", pdf_w)
                    json_h = data.get("canvasHeight", pdf_h)
                    scale_x = pdf_w / json_w if json_w else 1
                    scale_y = pdf_h / json_h if json_h else 1

                    # Pass 1: index captions by parent id
                    caption_map = {}
                    for anno in data.get("annotations", []):
                        body_list = anno.get("body", [])
                        if not isinstance(body_list, list):
                            continue

                        parent_id = None
                        is_caption = False
                        for item in body_list:
                            if item.get("value") == "Caption":
                                is_caption = True
                            if item.get("purpose") == "parent":
                                parent_id = item.get("value")

                        if is_caption and parent_id:
                            selector = anno.get("target", {}).get("selector", {}).get("value", "")
                            if "pixel:" in selector:
                                coords_str = selector.split("pixel:")[1]
                                cx, cy, cw, ch = map(float, coords_str.split(","))
                                caption_map[parent_id] = fitz.Rect(
                                    cx * scale_x,
                                    cy * scale_y,
                                    (cx + cw) * scale_x,
                                    (cy + ch) * scale_y,
                                )

                    # Pass 2: extract figure crops and captions
                    figure_counter = 0

                    for anno in data.get("annotations", []):
                        body_list = anno.get("body", [])
                        if not isinstance(body_list, list):
                            continue

                        if any(item.get("value") == "Figure" for item in body_list):
                            fig_anno_id = anno.get("id")
                            selector = anno.get("target", {}).get("selector", {}).get("value", "")
                            if "pixel:" not in selector:
                                continue

                            coords_str = selector.split("pixel:")[1]
                            x, y, w, h = map(float, coords_str.split(","))
                            rect_points = fitz.Rect(x * scale_x, y * scale_y, (x + w) * scale_x, (y + h) * scale_y)

                            # High-res extract
                            zoom = 300 / 72
                            mat = fitz.Matrix(zoom, zoom)
                            try:
                                pix = page.get_pixmap(matrix=mat, clip=rect_points)
                                out_filename = f"{page_id}-fig-{figure_counter}.png"
                                pix.save(os.path.join(output_dir, out_filename))
                            except Exception as e:
                                print(f"[Error] Save failed for {out_filename}: {e}")
                                continue

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
                                "bbox_pdf_coords": [rect_points.x0, rect_points.y0, rect_points.x1, rect_points.y1],
                            }
                            extracted_metadata.append(meta_entry)
                            figure_counter += 1

                except Exception as e:
                    # Catch-all for page processing errors
                    print(f"[Error] Processing failed for {page_id}: {e}")

                pbar.update(1)

            doc.close()

            # Optionally persist incrementally (safer for long runs)
            with open(metadata_output_path, 'w', encoding='utf-8') as f:
                json.dump(extracted_metadata, f, indent=4, ensure_ascii=False)

    print(f"Extraction complete. Metadata saved to {metadata_output_path}")
    return extracted_metadata