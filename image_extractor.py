import logging
import json
import re
from pathlib import Path
from docling_core.types.doc import PictureItem, DocItemLabel
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

logging.basicConfig(level=logging.INFO)

def find_figure_number(text):
    """Parses text to extract the numeric identifier following 'Figure' or 'Fig'."""
    if not text:
        return None
    # Optimized to catch Fig 1, Figure 1, Fig. 1, FIG 1
    match = re.search(r'(?i)(?:Figure|Fig\.?)\s*(\d+)', text)
    return match.group(1) if match else None

def clean_caption(text):
    """Removes the 'Figure X' prefix from the beginning of a caption string."""
    if not text:
        return ""
    pattern = r'(?i)^(?:Figure|Fig\.?)\s*\d+[:\.\s-]*\s*'
    text = re.sub(pattern, '', text)
    return text.strip()

def find_nearby_caption(target_element, all_items, threshold=250):
    """Searches for the nearest caption element on the same page based on vertical distance (default threshold: 250)."""
    if not target_element.prov:
        return None
    img_bbox = target_element.prov[0].bbox
    img_page = target_element.prov[0].page_no
    best_text = None
    min_dist = float('inf')
    for item, _ in all_items:
        if item.label == DocItemLabel.CAPTION and item.prov and item.prov[0].page_no == img_page:
            cap_bbox = item.prov[0].bbox
            # Calculate vertical distance (Top-to-Bottom OR Bottom-to-Top)
            # This handles captions both above and below the image
            dist_below = abs(img_bbox.b - cap_bbox.t)
            dist_above = abs(img_bbox.t - cap_bbox.b)
            current_min = min(dist_below, dist_above)
            if current_min < threshold and current_min < min_dist:
                min_dist = current_min
                best_text = item.text
    return best_text

def extract_figures_from_pdf(pdf_path_str, output_dir_str):
    """Converts a PDF to extract images and metadata, matching figures to captions via direct links or spatial proximity."""
    pdf_path = Path(pdf_path_str)
    output_dir = Path(output_dir_str)
    output_dir.mkdir(exist_ok=True, parents=True)
    paper_id = pdf_path.stem 
    metadata_file = output_dir / f"{paper_id}_metadata.json"
    pdf_opts = PdfPipelineOptions()
    pdf_opts.generate_picture_images = True
    pdf_opts.images_scale = 2.0
    
    converter = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_opts)}
    )
    
    print(f"--- Starting Extraction: {pdf_path.name} ---")
    result = converter.convert(pdf_path)
    doc = result.document
    metadata_list = []
    all_items = list(doc.iterate_items())
    # Process EVERY element in the document
    for element, _ in all_items:
        if isinstance(element, PictureItem):
            raw_caption = ""
            # 1. Direct Caption Link (Standard Docling behavior)
            if hasattr(element, 'captions') and element.captions:
                raw_caption = " ".join([cap.text for cap in element.captions if hasattr(cap, 'text')]).strip()
            # 2. Advanced Spatial Search (If direct link is missing)
            if not raw_caption:
                spatial_cap = find_nearby_caption(element, all_items)
                if spatial_cap:
                    raw_caption = spatial_cap.strip()
            # 3. Extract the number from whatever text we found
            fig_num = find_figure_number(raw_caption)
            # If we found a number, we save the image
            if fig_num:
                img = element.get_image(doc)
                if img:
                    figure_name = f"{paper_id}_{fig_num}.png"
                    save_path = output_dir / figure_name
                    img.save(save_path, format="PNG")
                    page_no = element.prov[0].page_no if element.prov else "Unknown"
                    metadata_list.append({
                        "figure_name": figure_name,
                        "figure_number": fig_num,
                        "caption": clean_caption(raw_caption),
                        "page_number": page_no
                    })
                    print(f"  [Saved] {figure_name} from Page {page_no}")

    # Remove duplicate metadata entries if multiple images were saved for the same Fig
    unique_metadata = {m['figure_name']: m for m in metadata_list}.values()
    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(list(unique_metadata), f, indent=4, ensure_ascii=False)
    print(f"--- Finished! Check {output_dir} ---")
    return metadata_file