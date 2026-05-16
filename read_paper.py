# this file is used to read the paper page by page and extract the number of figures in it.
from langchain_docling import  DoclingLoader
class Paper_reader():
    def __init__(self,latex_path = None, pdf_path = None):
        self.latex_path = latex_path
        self.pdf_path = pdf_path

    def _find_figures(self):
        """
        this fucntion reads the paper and checks each page if it has a figure in it.
        """
        print("[INFO] Processing the pdf:", self.pdf_path)
        loader = DoclingLoader(self.pdf_path)
        docs = loader.load()
        for i, page in enumerate(docs):
            print(page.page_content)
            if i==2:
                break






def main():
    paper_path = "./data/pdf_source/2404.12603.pdf"
    reader = Paper_reader(pdf_path=paper_path)
    reader._find_figures()
if __name__ == "__main__":
    main()
















