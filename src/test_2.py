"""
Equation extraction pipeline:
  PDF → PyMuPDF block detection → crop image → pix2tex → LaTeX

Install dependencies:
    pip install PyMuPDF pix2tex Pillow
"""

import os
import re
import fitz                        # PyMuPDF
from PIL import Image
from pix2tex.cli import LatexOCR

# Initialize pix2tex model once (slow to load, reuse across papers)
latex_ocr = LatexOCR()


class PdfEquationExtractor:
    def __init__(self, pdf_path: str, output_dir: str = "./equation_crops"):
        self.pdf_path = pdf_path
        self.paper_ID = os.path.splitext(os.path.basename(pdf_path))[0]
        self.output_dir = os.path.join(output_dir, self.paper_ID)
        os.makedirs(self.output_dir, exist_ok=True)
        self._appendix_page = self._find_appendix_page()

    # ------------------------------------------------------------------
    # Appendix detection
    # ------------------------------------------------------------------

    def _find_appendix_page(self) -> int:
        """Return the first page index (0-based) of the appendix, or inf."""
        doc = fitz.open(self.pdf_path)
        for i, page in enumerate(doc):
            text = page.get_text("text")
            if re.search(r"(?:^|\n)\s*(?:Appendix|APPENDIX)\b", text):
                doc.close()
                return i
        doc.close()
        return float("inf")

    # ------------------------------------------------------------------
    # Equation block detection
    # ------------------------------------------------------------------

    def _get_block_text(self, block: dict) -> str:
        return " ".join(
            span["text"]
            for line in block.get("lines", [])
            for span in line.get("spans", [])
        ).strip()

    def _has_math(self, text: str) -> bool:
        math_chars = set("=+−-×·÷/<>∑∏∫∂∇∞αβγδεζηθλμνξπρστυφχψωΩΓΔΛΦΨ∈∉⊂⊃∪∩")
        return (
            any(c in math_chars for c in text)
            or bool(re.search(r"[a-zA-Z][\^_]|[\^_][{0-9]", text))
        )

    def _is_prose(self, text: str) -> bool:
        # Too many plain lowercase words → it's a sentence, not an equation
        prose_words = re.findall(r"\b[a-z]{4,}\b", text)
        return len(prose_words) > 4

    def _is_table_block(self, block: dict) -> bool:
        # Table blocks in PyMuPDF have type 1 (image) for ruled tables,
        # but text tables show up as type 0 with many short lines.
        # Detect by counting lines — equations rarely have more than 4 lines.
        lines = block.get("lines", [])
        return len(lines) > 4

    def _is_figure_caption(self, text: str) -> bool:
        return bool(re.match(r"^\s*(FIG\.|Fig\.|Figure|TABLE|Table)\s*\d*", text))

    def _is_footnote(self, block: dict, page_height: float) -> bool:
        # Footnotes appear in the bottom 15% of the page
        _, y0, _, _ = block["bbox"]
        return y0 > page_height * 0.85

    def _is_author_header(self, block: dict, page_idx: int) -> bool:
        # Author blocks only appear on page 0
        return page_idx == 0

    def _is_equation_block(self, block, page_width: float, page_height: float,
                            blocks: list, block_idx: int, page_idx: int) -> bool:
        if block.get("type") != 0:
            return False

        bbox = block["bbox"]
        x0, y0, x1, y1 = bbox

        # --- Reject page 0 non-equation blocks (title, authors, abstract) ---
        # Only apply loose filtering on page 0; still allow equations
        if page_idx == 0:
            block_text = self._get_block_text(block)
            if not self._has_math(block_text):
                return False

        # --- Gap above: display equations have whitespace above them ---
        gap_above = y0 - blocks[block_idx - 1]["bbox"][3] if block_idx > 0 else y0
        if gap_above < 5:
            return False

        # --- Gap below: display equations also have whitespace below ---
        gap_below = blocks[block_idx + 1]["bbox"][1] - y1 if block_idx < len(blocks) - 1 else y1
        if gap_below < -30:  # deeply negative = tightly packed prose
            return False

        block_text = self._get_block_text(block)

        if len(block_text) < 2 or len(block_text) > 400:
            return False

        # --- Reject non-equation block types ---
        if self._is_figure_caption(block_text):
            return False
        if self._is_prose(block_text):
            return False
        if self._is_table_block(block):
            return False
        if self._is_footnote(block, page_height):
            return False

        # --- Must contain math ---
        if not self._has_math(block_text):
            return False

        return True

    # ------------------------------------------------------------------
    # Crop equation image from page
    # ------------------------------------------------------------------

    def _get_column_bounds(self, bbox, page_width: float):
        """
        Detect whether the equation is in left or right column of a two-column
        PDF, and return the x bounds of that column. Falls back to full width
        for single-column layouts.
        """
        x0, _, x1, _ = bbox
        eq_center = (x0 + x1) / 2
        page_center = page_width / 2

        # Two-column: equation center is clearly left or right of page center
        col_margin = page_width * 0.05  # 5% gutter
        if eq_center < page_center - col_margin:
            # Left column
            return 0, page_center - col_margin
        elif eq_center > page_center + col_margin:
            # Right column
            return page_center + col_margin, page_width
        else:
            # Single column or centered
            return 0, page_width

    def _crop_equation(self, page, bbox, eq_idx: int, page_idx: int, dpi: int = 300) -> str:
        """
        Render the equation region at high DPI:
        - Horizontal: restricted to the equation's column (fixes two-column bleed)
        - Vertical: tight around the bbox with small padding
        """
        padding_v = 8   # vertical padding in PDF points (tight)
        padding_h = 10  # horizontal padding within the column

        page_width = page.rect.width
        col_x0, col_x1 = self._get_column_bounds(bbox, page_width)

        x0_pt = max(0,          col_x0 - padding_h)
        x1_pt = min(page_width, col_x1 + padding_h)
        y0_pt = max(0,               bbox[1] - padding_v)
        y1_pt = min(page.rect.height, bbox[3] + padding_v)

        clip = fitz.Rect(x0_pt, y0_pt, x1_pt, y1_pt)
        mat = fitz.Matrix(dpi / 72, dpi / 72)
        pix = page.get_pixmap(matrix=mat, clip=clip, colorspace=fitz.csGRAY)
        img = Image.frombytes("L", [pix.width, pix.height], pix.samples)

        # White background (pix2tex expects white bg)
        bg = Image.new("L", img.size, 255)
        bg.paste(img)

        img_path = os.path.join(self.output_dir, f"eq_p{page_idx+1}_{eq_idx}.png")
        bg.save(img_path, dpi=(dpi, dpi))
        return img_path

    # ------------------------------------------------------------------
    # Main pipeline
    # ------------------------------------------------------------------

    def get_equations(self) -> dict:
        """
        Full pipeline: detect blocks → crop images → pix2tex → LaTeX.

        Returns:
            {
                eq_number: {
                    "latex": "...",
                    "image_path": "...",
                    "page": int,
                    "bbox": (x0, y0, x1, y1)
                }
            }
        """
        doc = fitz.open(self.pdf_path)
        equations = {}
        eq_counter = 1

        for page_idx, page in enumerate(doc):
            if page_idx >= self._appendix_page:
                break

            page_width = page.rect.width
            page_height = page.rect.height
            blocks = page.get_text("dict")["blocks"]

            for block_idx, block in enumerate(blocks):
                if not self._is_equation_block(
                    block, page_width, page_height, blocks, block_idx, page_idx
                ):
                    continue

                bbox = block["bbox"]

                # Step 1: crop equation region to image
                img_path = self._crop_equation(page, bbox, eq_counter, page_idx)

                # Step 2: run pix2tex LaTeX OCR on the cropped image
                try:
                    img = Image.open(img_path)
                    latex = latex_ocr(img)
                except Exception as e:
                    latex = f"[OCR ERROR: {e}]"

                equations[eq_counter] = {
                    "latex": latex,
                    "image_path": img_path,
                    "page": page_idx + 1,
                    "bbox": bbox,
                }

                print(f"  [{eq_counter}] page {page_idx+1} → {latex}")
                eq_counter += 1

        doc.close()
        return equations


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

paper_ids = [
"2403.07417"
]

PDF_DIR = "./data/pdf_source"
CROP_DIR = "./data/equation_crops"

for paper_id in paper_ids:
    pdf_path = os.path.join(PDF_DIR, f"{paper_id}.pdf")

    if not os.path.exists(pdf_path):
        print(f"[SKIP] {pdf_path} not found")
        continue

    print(f"\n=== Processing {paper_id} ===")
    extractor = PdfEquationExtractor(pdf_path, output_dir=CROP_DIR)
    equations = extractor.get_equations()

    print(f"  → {len(equations)} equations extracted")
    print("done")