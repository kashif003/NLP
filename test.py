import logging
import json
import re
from pathlib import Path
from docling_core.types.doc import PictureItem, DocItemLabel
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
logging.basicConfig(level=logging.INFO)
def clean_caption(text):
    if not text:
        return ""
    text = re.sub(r'^\d+\s+', '', text)
    pattern = r'^(?:Figure|Fig\.?)\s+\d+[:\.\s-]*\s*'
    text = re.sub(pattern, '', text, flags=re.IGNORECASE)
    return text.strip()

def find_nearby_caption(target_element, all_items, threshold=150):
    if not target_element.prov:
        return None
    img_bbox = target_element.prov[0].bbox
    img_page = target_element.prov[0].page_no
    
    nearby_texts = []
    for item, _ in all_items:
        if item.label == DocItemLabel.CAPTION and item.prov and item.prov[0].page_no == img_page:
            cap_bbox = item.prov[0].bbox
            distance = img_bbox.b - cap_bbox.t 
            if abs(distance) < threshold:
                nearby_texts.append(item.text)
    return " ".join(nearby_texts) if nearby_texts else None

def extract_figures_from_pdf(pdf_path_str, output_dir_str):
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
    saved_count = 0
    all_items = list(doc.iterate_items())
    for element, _ in all_items:
        if isinstance(element, PictureItem):
            raw_caption = ""

            if hasattr(element, 'captions') and element.captions:
                parts = []
                for cap in element.captions:
                    if hasattr(cap, 'text'): parts.append(cap.text)
                    elif hasattr(cap, 'target'):
                        ref = doc.get_item(cap.target)
                        if ref and hasattr(ref, 'text'): parts.append(ref.text)
                raw_caption = " ".join(parts).strip()
            if not raw_caption:
                spatial_cap = find_nearby_caption(element, all_items)
                if spatial_cap:
                    raw_caption = spatial_cap.strip()
            final_caption = clean_caption(raw_caption)
            if final_caption:
                saved_count += 1
                img = element.get_image(doc)
                if img:
                    figure_name = f"{paper_id}_{saved_count}.png"
                    img.save(output_dir / figure_name, format="PNG")

                    page_no = element.prov[0].page_no if element.prov else "Unknown"
                    metadata_list.append({
                        "figure_index": saved_count,
                        "figure_name": figure_name,
                        "caption": final_caption,
                        "page_number": page_no
                    })
    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(metadata_list, f, indent=4, ensure_ascii=False)

    print(f"--- Finished! Saved {saved_count} figures to {output_dir} ---\n")
    return metadata_file

if __name__ == "__main__":
    my_pdf = "paper_pdf/2402.11205v3.pdf"
    my_output = "processed_papers"
    
    extract_figures_from_pdf(my_pdf, my_output)