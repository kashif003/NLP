import os
import re
from bs4 import BeautifulSoup
from pylatexenc.latexwalker import LatexWalker, LatexMacroNode, LatexCharsNode, LatexGroupNode, LatexEnvironmentNode

class HTMLReader:
    def __init__(self, paper_ID):
        self.paper_ID = paper_ID
        self.html_path = f"{os.path.join("data/html_source", paper_ID)}.html"
        self.file_content = self._get_file_content()
        self.equations = self.finding_all_equations()



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

    def get_symbols_from_latex(self, latex):
        ignore = {'frac', 'left', 'right', 'cdot', 'times', 'sqrt', 'exp',
                'sec', 'cos', 'sin', 'tan', 'log', 'ln', 'begin', 'end',
                'displaystyle', 'text', 'mathbf', 'mathrm', 'infty', 'sum',
                'int', 'lim', 'over', 'bar', 'hat', 'dot', 'tilde', 'leq',
                'geq', 'neq', 'pm', 'mp', 'to', 'rightarrow', 'leftarrow', 'and'}
        symbols = set()

        # Step 1: simple symbols like G_{t} or A or \eta_{t}
        step1 = re.findall(r'\\[a-zA-Z]+(?:_\{[^}]+\}|\^\{[^}]+\})?|[A-Za-z](?:_\{[^}]+\}|\^\{[^}]+\})?', latex)
        for s in step1:
            name = s.lstrip('\\').split('_')[0].split('^')[0]
            if name not in ignore:
                symbols.add(s)

        # Step 2: symbols inside {} not preceded by a command
        step2 = re.findall(r'(?<!\\[a-zA-Z]{0,20})\{([^}]+)\}', latex)
        for group in step2:
            for s in re.findall(r'\\[a-zA-Z]+|[A-Za-z]', group):
                name = s.lstrip('\\')
                if name not in ignore:
                    symbols.add(s)

        # Step 3: remove anything that came from inside ignored commands like \frac{...}
        for cmd in ignore:
            pattern = re.compile(r'\\' + cmd + r'\{[^}]*\}')
            for m in pattern.finditer(latex):
                inner = re.findall(r'\\[a-zA-Z]+|[A-Za-z]', m.group())
                for s in inner:
                    symbols.discard(s)

        return list(symbols)



paper_ids = [
"2401.02303",
]

for paper_id in paper_ids:
    reader = HTMLReader(paper_id)

    content = reader.file_content
    equattions= reader.finding_all_equations()
    symbols = reader.get_symbols_from_latex("G_{t}=\frac{8}{\Theta^{2}_{B}}\,,\ G_{r}=\frac{4\pi A_{r}}{\lambda^{2}}\ \text{and}\ L_{r}=(\frac{\lambda}{4\pi L})^{2}\,")
    symbols_text = reader.find_symbol_in_paragraphs("\eta_{t}")
    
    print(symbols)
    print(symbols_text)

    # for k,y in equattions.items():
    #      print(k)
    #      print(y["eq_id"])
    #      print(y["real_id"])
    #      print(y["latex"])

        
    #print(content)

    print("done")