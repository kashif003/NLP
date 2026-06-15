import os
import re
from bs4 import BeautifulSoup


class HTMLReader:
    def __init__(self, paper_ID):
        self.paper_ID = paper_ID
        self.html_path = os.path.join("./data/html_source", f"{paper_ID}.html")
        self.soup = self._load_html()
        self.main_content, self.appendix_content = self._split_content()

    def _load_html(self) -> BeautifulSoup:
        """Load and parse the HTML file."""
        with open(self.html_path, "r", encoding="utf-8") as f:
            return BeautifulSoup(f.read(), "html.parser")

    def _split_content(self):
        """Split raw HTML into main and appendix at the appendix section."""
        html = str(self.soup)

        # find appendix header
        appendix_match = re.search(
            r'<[^>]*(h1|h2|h3|h4)[^>]*>.*?appendix.*?<',
            html,
            re.IGNORECASE
        )

        if appendix_match:
            main_html = html[:appendix_match.start()]
            appendix_html = html[appendix_match.start():]
        else:
            main_html = html
            appendix_html = ""

        return main_html, appendix_html

    def get_equations(self) -> dict:
        """
        Extract all numbered equations from the main HTML content.

        Returns
        -------
        dict
            Dictionary with equation number as key and LaTeX string as value.
        """
        equations = {}

        # arxiv HTML uses <math> tags with alttext containing LaTeX
        all_math = self.soup.find_all("math")

        for math_tag in all_math:
            # check if it has an equation number nearby
            parent = math_tag.find_parent()
            if not parent:
                continue

            # look for equation number pattern (1), (2), etc. in surrounding text
            parent_text = parent.get_text()
            eq_num_match = re.search(r"\((\d+)\)", parent_text)
            if not eq_num_match:
                continue

            eq_number = int(eq_num_match.group(1))
            # get LaTeX from alttext attribute
            latex = math_tag.get("alttext", math_tag.get_text().strip())

            if latex:
                equations[eq_number] = latex

        return equations

from bs4 import BeautifulSoup

def count_equations(html_content) -> int:
    if isinstance(html_content, list):
        html_content = " ".join(str(tag) for tag in html_content)
    
    soup = BeautifulSoup(html_content, "html.parser")
    equation_tables = soup.find_all("table", class_="ltx_equation")
    
    for t in equation_tables[:5]:  # print first 5 to inspect
        span = t.find("span", class_="ltx_tag_equation")
        print(f"span: {span}")
    
    numbered = [t for t in equation_tables if t.find("span", class_="ltx_tag_equation")]
    return len(numbered)

if __name__ == "__main__":
    paper_ids = [
        "2401.14764",
        "2406.04217"
    ]
    with open("./data/html_source/2401.14764.html", "r", encoding="utf-8") as f:
        html_content = f.read()

    count = count_equations(html_content)  # pass raw string, not main/appendix split
    print(count)
    for paper_id in paper_ids:
        print(f"\nProcessing {paper_id}...")
        reader = HTMLReader(paper_id)
        
        main=reader.main_content
        count = count_equations(main)
        print("count:", count)
        appendix=reader.appendix_content
        print("done")
        '''equations = reader.get_equations()
        print(f"Found {len(equations)} equations")
        for num, eq in equations.items():
            print(f"  ({num}): {eq}")
        print("done")'''