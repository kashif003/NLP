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


    def finding_all_equations(self):
        soup = BeautifulSoup(self.file_content, "html.parser")
        equations = {}
        current_prefix_letter = None
        section_counter = 1
        global_counter = 1
        for span in soup.find_all("span", class_="ltx_tag_equation"):
            parent = span.find_parent(id=re.compile(r"E\d+"))
            eq_id = parent.get("id") if parent else None
            if eq_id:
                prefix = re.match(r"^(.*?)\.E\d+$", eq_id).group(1)
                prefix_letter = prefix[0]
                if current_prefix_letter and prefix_letter != current_prefix_letter:
                    section_counter = 1
                current_prefix_letter = prefix_letter
                mapped_id = f"{prefix}:E{section_counter}"
                section_counter += 1
            else:
                mapped_id = None
            if parent:
                math_tags = parent.find_all("math")
                latex = " ".join(m.get("alttext") for m in math_tags if m.get("alttext"))
            else:
                latex = None
            equations[global_counter] = {"eq_id": mapped_id, "latex": str(latex)}
            global_counter += 1
        return equations



paper_ids = [
"2401.02303",
]

for paper_id in paper_ids:
    reader = HTMLReader(paper_id)

    content = reader.file_content
    equattions= reader.finding_all_equations()
    for k,y in equattions.items():
         print(k)
         print(y["eq_id"])
         print(y["latex"])

        
    #print(content)

    print("done")