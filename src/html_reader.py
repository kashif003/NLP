import os
import re
from bs4 import BeautifulSoup

class HTMLReader:
    def __init__(self, paper_ID):
        self.paper_ID = paper_ID
        self.html_path = f"{os.path.join("data/html_source", paper_ID)}.html"
        self.file_content = self._get_file_content()



    def _get_file_content(self):
        with open(self.html_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()

    def findfing_all_equations(self):
        soup = BeautifulSoup(self.file_content, "html.parser")
        for span in soup.find_all("span", class_="ltx_tag_equation"):
            eq_num = span.get_text(strip=True)  # e.g. "(139)"
            parent = span.find_parent(id=re.compile(r"E\d+"))
            eq_id = parent.get("id") if parent else None  # e.g. "S4.E139"
            math = parent.find("math") if parent else None
            latex = math.get("alttext") if math else None     #TODO not getting the full equation check "2402.07100" equation S3.E31, A1.E56(A2, A5 in paper)
            print(eq_num, eq_id, latex)



paper_ids = [
"2402.07100",
]

for paper_id in paper_ids:
    reader = HTMLReader(paper_id)

    content = reader.file_content
    reader.findfing_all_equations()


    print("done")