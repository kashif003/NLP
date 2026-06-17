import os
import re
from bs4 import BeautifulSoup

DEFINITION_PATTERNS = re.compile(
    r'\b(where|denotes|denote|represents|represent|let|is defined as|is given by|is the|are the|refers to|correspond to|stands for)\b',
    re.IGNORECASE
)

SKIP_PATTERNS = [
    re.compile(r'^[a-zA-Z]$'),
    re.compile(r'^\d+$'),
    re.compile(r'^[=<>+\-*/,.\(\)]+$'),
]

MAX_EQUATIONS = 7


class HTMLReader:
    def __init__(self, paper_ID):
        self.paper_ID = paper_ID
        self.html_path = os.path.join("data/html_source", f"{paper_ID}.html")
        self.soup = self._parse_html()
        self.equations = self._find_equations()
        self.all_symbols = self._get_paragraph_symbols()
        self.equation_symbols = self._match_symbols_to_equations()

    def _parse_html(self):
        with open(self.html_path, "r", encoding="utf-8", errors="ignore") as f:
            return BeautifulSoup(f.read(), "html.parser")

    def _find_equations(self):
        equations = {}
        current_prefix_letter = None
        section_counter = 1
        global_counter = 1

        for span in self.soup.find_all("span", class_="ltx_tag_equation"):
            if len(equations) == MAX_EQUATIONS:
                break

            parent = span.find_parent(id=re.compile(r"E\d+"))
            eq_id = parent.get("id") if parent else None
            mapped_id = None

            if eq_id:
                match = re.match(r"^(.*?)\.E\d+$", eq_id)
                if not match:
                    global_counter += 1
                    continue
                prefix = match.group(1)
                prefix_letter = prefix[0]
                if current_prefix_letter and prefix_letter != current_prefix_letter:
                    section_counter = 1
                current_prefix_letter = prefix_letter
                mapped_id = f"{prefix}.E{section_counter}"
                section_counter += 1

            latex = None
            if parent:
                latex = " ".join(
                    m.get("alttext") for m in parent.find_all("math") if m.get("alttext")
                )

            equations[global_counter] = {
                "eq_id": mapped_id,
                "real_id": eq_id,
                "latex": str(latex)
            }
            global_counter += 1

        return equations

    def _should_skip_symbol(self, alt):
        return any(p.match(alt) for p in SKIP_PATTERNS)

    def _get_paragraph_symbols(self):
        paragraph_symbols = {}
        for p in self.soup.find_all("p"):
            for m in p.find_all("math"):
                alt = m.get("alttext")
                math_id = m.get("id")
                if not alt or not math_id or self._should_skip_symbol(alt):
                    continue
                paragraph_symbols.setdefault(alt, []).append(math_id)
        return paragraph_symbols

    def _match_symbols_to_equations(self):
        matched = {}
        for eq in self.equations.values():
            eq_latex, eq_id = eq["latex"], eq["eq_id"]
            for symbol, math_ids in self.all_symbols.items():
                if symbol in eq_latex:
                    if symbol not in matched:
                        matched[symbol] = {"equations": [], "unique_ids": math_ids}
                    if eq_id not in matched[symbol]["equations"]:
                        matched[symbol]["equations"].append(eq_id)
        return matched

    def _extract_text_from_container(self, container):
        """Copy container, replace math tags with alttext, return plain text."""
        container_copy = BeautifulSoup(str(container), "html.parser")
        for tag in container_copy.find_all("math"):
            tag.replace_with(tag.get("alttext", ""))
        return container_copy.get_text(" ", strip=True)

    def _find_container(self, tag):
        """Walk up the tree to find the nearest meaningful container."""
        for parent in tag.parents:
            if parent.name in ["p", "td", "li", "div", "section",
                                "span", "dd", "dt", "blockquote", "figcaption"]:
                return parent
        return None

    def _best_sentence(self, symbol, uid):
        """Return the best defining sentence for a symbol from a given math tag id."""
        m = self.soup.find(attrs={"id": uid})
        if not m:
            return None

        container = self._find_container(m)
        if not container:
            return None

        full_text = self._extract_text_from_container(container)
        sentences = re.split(r'(?<=[.!?])\s+', full_text)

        # 1. Prefer sentences with definition patterns
        for sentence in sentences:
            if symbol in sentence and DEFINITION_PATTERNS.search(sentence):
                return sentence

        # 2. Fallback: first sentence containing the symbol
        for sentence in sentences:
            if symbol in sentence:
                return sentence

        return None

    def get_symbol_full_context(self):
        result = {}
        for symbol, data in self.equation_symbols.items():
            best_sentence = None
            for uid in data["unique_ids"]:
                best_sentence = self._best_sentence(symbol, uid)
                if best_sentence:
                    break
            result[symbol] = {
                "equations": data["equations"],
                "context": best_sentence
            }
        return result

    def get_data(self):
        full_context = self.get_symbol_full_context()
        result = {}

        # build a lookup: eq_id -> latex
        eq_id_to_latex = {eq["eq_id"]: eq["latex"] for eq in self.equations.values()}

        for symbol, data in full_context.items():
            ctx = data["context"]
            if ctx is None:
                continue
            for eq_id in data["equations"]:
                eq_latex = eq_id_to_latex.get(eq_id)
                if eq_latex not in result:
                    result[eq_latex] = []
                result[eq_latex].append((symbol, ctx))

        return result


if __name__ == "__main__":
    paper_ids = ["2410.08937"]

    for paper_id in paper_ids:
        reader = HTMLReader(paper_id)
        data = reader.get_data()

        for equation, symbols in data.items():
            print("[EQUATION:]")
            print(equation)
            print("Symbols:")
            for symbol, ctx in symbols:
                print(f"  {symbol}: {ctx}")
            print()

    print("done")

import json
output_filename = "equations_data.json"
with open(output_filename, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=4, ensure_ascii=False)

print(f"Data successfully saved to {output_filename}")
print("done")