import os
import re
import fitz  # PyMuPDF


class PdfReader:
    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
        self.paper_ID = os.path.splitext(os.path.basename(pdf_path))[0]
        self.file_content = self._get_file_content()

    def _get_file_content(self) -> dict:
        """Extract text from PDF, splitting at appendix."""
        doc = fitz.open(self.pdf_path)
        pages_text = []
        for page in doc:
            pages_text.append(page.get_text("text"))
        doc.close()

        full_text = "\n".join(pages_text)

        appendix_match = re.search(
            r"(?:^|\n)\s*(?:Appendix|APPENDIX)\b",
            full_text,
        )

        if appendix_match:
            main_content = full_text[: appendix_match.start()]
            appendix_content = full_text[appendix_match.start() :]
        else:
            main_content = full_text
            appendix_content = ""

        return {"main": main_content, "appendix": appendix_content}

    def get_equations(self) -> dict:
        """
        Extract equations from the PDF using PyMuPDF's math-block detection.
        Returns a dict: {equation_number: equation_string}
        """
        doc = fitz.open(self.pdf_path)
        equations = {}
        eq_counter = 1

        # Determine which pages belong to the main content (before appendix)
        full_pages_text = []
        page_boundaries = []
        char_count = 0
        for page in doc:
            text = page.get_text("text")
            page_boundaries.append((char_count, char_count + len(text)))
            full_pages_text.append(text)
            char_count += len(text) + 1  # +1 for the "\n" join

        appendix_start_char = len(self.file_content["main"])

        for page_idx, page in enumerate(doc):
            page_start, page_end = page_boundaries[page_idx]

            # Skip pages that are entirely in the appendix
            if page_start >= appendix_start_char:
                break

            # Use dict mode to get block positions and types
            blocks = page.get_text("dict")["blocks"]

            for block in blocks:
                if block.get("type") != 0:  # 0 = text block
                    continue

                block_text = " ".join(
                    span["text"]
                    for line in block.get("lines", [])
                    for span in line.get("spans", [])
                ).strip()

                if not block_text:
                    continue

                # Heuristic: equation blocks are short, centered, and contain
                # math-like characters. Adjust thresholds as needed.
                if self._looks_like_equation(block_text):
                    equations[eq_counter] = block_text
                    eq_counter += 1

        doc.close()
        return equations

    @staticmethod
    def _looks_like_equation(text: str) -> bool:
        """
        Heuristic filter to identify equation blocks.
        Tuned for academic PDFs (short lines with math symbols).
        """
        if len(text) > 300:   # equations are rarely very long
            return False
        if len(text) < 2:
            return False

        math_chars = set("=+−-×·÷/<>∑∏∫∂∇∞αβγδεζηθλμνξπρστυφχψωΩΓΔΛΦΨ∈∉⊂⊃∪∩")
        math_hit = sum(1 for c in text if c in math_chars)

        # Must contain at least one math character or look like an inline formula
        has_math = math_hit > 0 or bool(re.search(r"[a-zA-Z]_|[a-zA-Z]\^|\^[{0-9]|_[{0-9]", text))

        # Should not look like plain prose (many lowercase words in a row)
        prose_words = re.findall(r"\b[a-z]{4,}\b", text)
        is_prose = len(prose_words) > 5

        return has_math and not is_prose


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

paper_ids = [
    "2307.08384"
]

PDF_DIR = "./data/pdf_source"   # <-- change this to your actual PDF directory

for paper_id in paper_ids:
    pdf_path = os.path.join(PDF_DIR, f"{paper_id}.pdf")

    if not os.path.exists(pdf_path):
        print(f"[SKIP] {pdf_path} not found")
        continue

    reader = PdfReader(pdf_path)
    content = reader.file_content
    print(content)
    print("done")