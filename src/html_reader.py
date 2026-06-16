import os
import re
from bs4 import BeautifulSoup
class HTMLReader:
    def __init__(self, paper_ID):
        self.paper_ID = paper_ID
        self.html_path = f"{os.path.join("data/html_source", paper_ID)}.html"
        self.file_content = self._get_file_content()
        self.equations = self.finding_all_equations()
        self.all_symbols = self.get_all_paragraph_symbols()
        self.equation_symbols = self.match_symbols_in_equation()



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
                mapped_id = f"{prefix}.E{section_counter}"
                section_counter += 1
            else:
                mapped_id = None
            if parent:
                math_tags = parent.find_all("math")
                latex = " ".join(m.get("alttext") for m in math_tags if m.get("alttext"))
            else:
                latex = None
            equations[global_counter] = {"eq_id": mapped_id, "real_id": eq_id, "latex": str(latex)}
            global_counter += 1
        return equations

    def find_symbol_in_paragraphs(self, symbol):
        soup = BeautifulSoup(self.file_content, "html.parser")
        results = {}
        for m in soup.find_all("math"):
            if m.get("alttext") == symbol:
                p = m.find_parent("p")
                if p:
                    for tag in p.find_all("math"):
                        tag.replace_with(tag.get("alttext", ""))
                    results[m.get("id")] = p.get_text(" ", strip=True)
        return results
    def get_all_paragraph_symbols(self):
        soup = BeautifulSoup(self.file_content, "html.parser")
        paragraph_symbols = {}

        for p in soup.find_all("p"):
            for m in p.find_all("math"):
                alt = m.get("alttext")
                math_id = m.get("id")
                if not alt or not math_id:
                    continue
                
                # skip single plain letters like "f", "z", "a"
                if re.match(r'^[a-zA-Z]$', alt):
                    continue
                
                # skip plain numbers
                if re.match(r'^\d+$', alt):
                    continue
                
                # skip operators and punctuation
                if re.match(r'^[=<>+\-*/,.\(\)]+$', alt):
                    continue

                paragraph_symbols[alt] = math_id

        return paragraph_symbols
   
    def match_symbols_in_equation(self):
        matched = {}  # symbol → {"eq_id": ..., "unique_id": ...}
        
        for k, eq in self.equations.items():
            eq_latex = eq["latex"]
            eq_id = eq["eq_id"]
            
            for symbol, math_id in self.all_symbols.items():
                if symbol in eq_latex:
                    if symbol not in matched:
                        matched[symbol] = []
                    matched[symbol].append({
                        "eq_id": eq_id,
                        "unique_id": math_id
                    })
        
        return matched


paper_ids = [
"2401.13724",
]

for paper_id in paper_ids:
    reader = HTMLReader(paper_id)

    content = reader.file_content
    equattions= reader.finding_all_equations()
    #symbols = reader.get_all_paragraph_symbols()
    symbols = reader.equation_symbols
    for k,v in symbols.items():
        print(k)
        print(v)

    #symbols_text = reader.find_symbol_in_paragraphs("\eta_{t}")
    
    # for k,y in symbols.items():
    #     print(k)
    #     print(y)
    #     print()

    # for k,y in equattions.items():
    #      print(k)
    #      print(y["eq_id"])
    #      print(y["real_id"])
    #      print(y["latex"])

        
    #print(content)

    print("done")