import os
import re
import tarfile
from collections import defaultdict

class LatexReader:
    def __init__(self, paper_ID):
        self.paper_ID = paper_ID
        self.latex_dir = os.path.join("./data/latex_source", paper_ID)
        self.tex_files = self._get_tex_files_path()
        self.file_content = self._get_file_content()

    
    def _get_tex_files_path(self):
        """Get all .tex files from the latex directory."""
        tex_files = []
        for root, _, files in os.walk(self.latex_dir):
            for file in files:
                if file.endswith(".tex"):
                    tex_files.append(os.path.join(root, file))
        return tex_files
    
    def _get_file_content(self):
        file_with_content = {}
        for path in self.tex_files:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()

            # split at appendix
            appendix_match = re.search(
                r"\\appendix\b|\\begin\{appendices?\}|\\section\*?\{[^}]*appendix[^}]*\}",
                content,
                re.IGNORECASE
            )

            if appendix_match:
                main_content = content[:appendix_match.start()]
                appendix_content = content[appendix_match.start():]
            else:
                main_content = content
                appendix_content = ""

            file_with_content[path] = {
                "main": main_content,
                "appendix": appendix_content
            }

        return file_with_content
    
    def get_equations(self):
        """Extract all numbered equations from the tex files."""
        equations = {}
        for tex_file in self.tex_files:
            with open(tex_file, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
            equations.update(self._extract_equations(content))
        return equations
    

    def _extract_equations(self):
        """Extract numbered equations from latex content."""
        equations = {}
        pattern = re.compile(
            r"\\begin\{equation\}(.*?)\\end\{equation\}",
            re.DOTALL
        )
        eq_counter = 1
        for file_data in self.file_content.values():
            for section in file_data["main"]:
                for match in pattern.finditer(section):
                    eq = match.group(1).strip()
                    equations[eq_counter] = eq
                    eq_counter += 1
        return equations


paper_ids = [
    "2307.08384",
    "2401.02303",
    "2401.13724",
    "2401.14764",
    "2402.03500",
    "2402.07100",
    "2402.09752"
]
for id in paper_ids:
    file = LatexReader(id)
    tex_files = file.tex_files
    content = file.file_content
    equations = file._extract_equations()
    print(equations)
    print("done")


'''

'''