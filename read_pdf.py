import fitz  # PyMuPDF


class PdfReader:
    def __init__(self, pdf_path: str):
        self.path = pdf_path
        self._page_text: list[str] = []
        self._num_pages=None

    def _load_pages(self) -> None:
        # Only load once (lazy)
        if self._page_text:
            return

        # Use context manager so the file is always closed
        with fitz.open(self.path) as doc:
            for page in doc:
                self._page_text.append(page.get_text())
        self._num_pages=len(self._page_text)

    @property
    def pages(self) -> list[str]:
        """Return list of page texts (one string per page)."""
        self._load_pages()
        return self._page_text

    def get_page(self, index: int) -> str:
        """Return text of a single page by zero-based index."""
        self._load_pages()
        return self._page_text[index]
    
    def get_num_pages(self):
        self._load_pages()
        return self._num_pages
    




