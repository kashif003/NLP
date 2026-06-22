import os
import re
from bs4 import BeautifulSoup, NavigableString

EQUATION_BLOCK_CLASSES = {
    "ltx_equationgroup",
    "ltx_equation",
    "ltx_eqn_row",
    "ltx_eqn_table",
}

MAX_EQUATIONS = 7


class HTML_Reader:
    """Extracts clean text from arXiv HTML papers, replacing math with placeholders."""

    def __init__(self, paper_id):
        """Initialize the extractor with file paths, counters, mappings, and audit trails."""
        self.paper_id = paper_id
        self.html_path = os.path.join("data/html_source", f"{paper_id}.html")
        self.soup = self._parse_html()

        # counters and mappings
        self._sym_counter = 1
        self.eqn_mapping = {}   # "EQN1" -> {"latex": ..., "real_id": ...}
        self.sym_mapping = {}   # "SYM1"  -> latex string
        self._sym_seen = {}     # latex string -> placeholder (dedup)
        self._eqn_seen = {}     # real_id -> placeholder (dedup)

        # audit trail: method_name -> list of short messages.
        self.audit = {}

        # authoritative equation set, mirrors HTMLReader._find_equations
        self.equations = self._find_equations()   # real_id -> {number, latex}

    def _log(self, method, message):
        """Append a short diagnostic message to the audit log."""
        self.audit.setdefault(method, []).append(message)

    def _parse_html(self):
        """Parse the input HTML file into a BeautifulSoup object."""
        with open(self.html_path, "r", encoding="utf-8", errors="ignore") as f:
            return BeautifulSoup(f.read(), "html.parser")

    def _get_block_root(self, tag):
        """Find the topmost equation block container by walking up parents."""
        current = tag
        while current.parent:
            parent_classes = set(current.parent.get("class") or [])
            if parent_classes & EQUATION_BLOCK_CLASSES:
                current = current.parent
            else:
                break
        return current

    def _find_equations(self):
        """Locate and track up to MAX_EQUATIONS enumerated equations from the HTML."""
        equations = {}
        current_prefix = None
        section_counter = 1
        seen_roots = set()

        for span in self.soup.find_all("span", class_="ltx_tag_equation"):
            if len(equations) == MAX_EQUATIONS:
                break

            parent = span.find_parent(id=re.compile(r"E\d+"))
            eq_id = parent.get("id") if parent else None

            if eq_id:
                match = re.match(r"^(.*?)\.E\d+$", eq_id)
                if not match:
                    continue
                prefix = match.group(1)
                if current_prefix and prefix != current_prefix:
                    section_counter = 1
                current_prefix = prefix
                section_counter += 1

            # walk to block root to deduplicate multi-line equations
            block_root = self._get_block_root(parent) if parent else parent
            root_id = id(block_root)
            if root_id in seen_roots:
                continue
            seen_roots.add(root_id)

            if not eq_id:
                continue

            # displayed paper number from span text e.g. "(1)" -> "1"
            number = re.sub(r'[\(\)]', '', span.get_text(strip=True)).strip()
            # full latex from ALL math tags in the entire block
            latex = self._get_equation_latex(block_root)

            equations[eq_id] = {"number": number, "latex": latex}

            # audit: record that this equation was found
            snippet = (latex[:60] + "...") if len(latex) > 60 else latex
            self._log(
                "find_equations",
                f"Found equation {number}: {snippet}",
            )

        return equations

    def _get_equation_latex(self, block_root):
        """Extract and concatenate full LaTeX string from all math tags in a block."""
        raw_latex = " ".join(
            m.get("alttext", "")
            for m in block_root.find_all("math")
            if m.get("alttext")
        )
        
        # --- NEW CLEANING STEP ---
        # Strip \text{} formatting from the equations as well so they match the symbols
        if raw_latex:
            raw_latex = re.sub(r'\\text{([^}]+)}', r'\1', raw_latex)
        # -------------------------
        
        return raw_latex

    def _get_eqn_placeholder(self, real_id):
        """Get or create a unique placeholder (e.g., EQN1) for a valid equation ID."""
        # not in authoritative enumerated set (non-enumerated / beyond limit)
        if real_id not in self.equations:
            return "TEMPEQN"

        # already assigned
        if real_id in self._eqn_seen:
            return self._eqn_seen[real_id]

        # new enumerated equation — paper number + precomputed latex
        number = self.equations[real_id]["number"]
        placeholder = f"EQN{number}"
        self.eqn_mapping[placeholder] = {
            "latex": self.equations[real_id]["latex"],
            "real_id": real_id,
        }
        self._eqn_seen[real_id] = placeholder
        return placeholder

    def _get_sym_placeholder(self, alttext):
        """Get or create a unique placeholder (e.g., SYM1) for an inline math token."""
        # --- NEW CLEANING STEP ---
        # This removes \text{ or \text and the enclosing brackets, leaving just the content
        if alttext:
            alttext = re.sub(r'\\text{([^}]+)}', r'\1', alttext)
        # -------------------------

        if alttext in self._sym_seen:
            return self._sym_seen[alttext]

        placeholder = f"SYM{self._sym_counter}"
        self._sym_counter += 1
        self.sym_mapping[placeholder] = alttext
        self._sym_seen[alttext] = placeholder
        return placeholder

    def _is_inside_equation(self, tag):
        """Check if a given HTML tag is structurally inside an equation block."""
        for parent in tag.parents:
            parent_classes = set(parent.get("class") or [])
            if parent_classes & EQUATION_BLOCK_CLASSES:
                return True
        return False

    def _process_node(self, node):
        """Recursively convert HTML text and nodes into placeholder-replaced strings."""
        # plain text node
        if isinstance(node, NavigableString):
            return str(node)

        # equation block — replace with placeholder
        node_classes = set(node.get("class") or [])
        if node_classes & EQUATION_BLOCK_CLASSES:
            real_id = None
            own_id = node.get("id")
            if own_id in self.equations:
                real_id = own_id
            else:
                for tagged in node.find_all(id=re.compile(r"E\d+")):
                    if tagged.get("id") in self.equations:
                        real_id = tagged.get("id")
                        break
            if real_id:
                return " " + self._get_eqn_placeholder(real_id) + " "
            return " TEMPEQN "

        # equation mention
        if node.name == "a":
            href = node.get("href", "")
            if "#" in href:
                ref_id = href.rsplit("#", 1)[1]
                if re.search(r'\.E\d+', ref_id):
                    if ref_id in self.equations:
                        number = self.equations[ref_id]["number"]
                    else:
                        number = re.sub(r'[()]', '', node.get_text()).strip()
                    if number:
                        return " MEQN" + number + " "

        # inline math tag — replace with symbol placeholder
        if node.name == "math":
            if self._is_inside_equation(node):
                return ""
            alttext = node.get("alttext", "")
            if alttext:
                return " " + self._get_sym_placeholder(alttext) + " "
            return ""

        # recurse into children
        parts = []
        for child in node.children:
            parts.append(self._process_node(child))
        return "".join(parts)

    def extract(self):
        """Process the document to extract cleaned text along with its math mappings."""
        body = self.soup.find("body") or self.soup

        raw_text = self._process_node(body)

        # collapse whitespace
        clean_text = re.sub(r'\n{3,}', '\n\n', raw_text)
        clean_text = re.sub(r' {2,}', ' ', clean_text)

        # tidy equation mentions
        mention = r'MEQN[A-Za-z0-9]+(?:\.[A-Za-z0-9]+)*'
        clean_text = re.sub(
            r'(?:(?:Eqs?|Eqns?|Equations?)\.?\s*)?\(\s*(' + mention + r')\s*\)',
            r'\1',
            clean_text,
        )
        clean_text = re.sub(
            r'(?:Eqs?|Eqns?|Equations?)\.?\s*(' + mention + r')',
            r'\1',
            clean_text,
        )

        clean_text = clean_text.strip()

        return clean_text, self.eqn_mapping, self.sym_mapping


def map_symbols_to_equations(eqn_mapping, sym_mapping, audit=None):
    """Map which inline symbols appear inside each equation using boundary-aware regex."""
    matchers = {}
    for sym_ph, sym_latex in sym_mapping.items():
        if not sym_latex:
            continue
        if len(sym_latex) == 1 and sym_latex.isalpha():
            pattern = r"(?<![A-Za-z\\])" + re.escape(sym_latex) + r"(?![A-Za-z])"
        else:
            pattern = re.escape(sym_latex)
        matchers[sym_ph] = re.compile(pattern)

    result = {}
    for eq_ph, eq_data in eqn_mapping.items():
        eq_latex = eq_data.get("latex", "")
        found = [
            sym_ph for sym_ph, rx in matchers.items() if rx.search(eq_latex)
        ]
        result[eq_ph] = found

        if audit is not None:
            audit.setdefault("map_symbols_to_equations", []).append(
                f"{eq_ph}: matched symbols {found}"
            )

    return result


# if __name__ == "__main__":
#     from pathlib import Path

#     html_dir = Path("./data/html_source")
#     out_dir = Path("./logs/parsed_papers")
#     out_dir.mkdir(parents=True, exist_ok=True)

#     paper_ids = [f.stem for f in html_dir.iterdir()
#                  if f.is_file() and f.suffix == ".html"]

#     for paper_id in paper_ids:
#         try:
#             extractor = PaperTextExtractor(paper_id)
#             clean_text, eqn_mapping, sym_mapping = extractor.extract()
#             eq_to_syms = map_symbols_to_equations(
#                 eqn_mapping, sym_mapping, audit=extractor.audit
#             )
#         except Exception as e:
#             print(f"[SKIP] {paper_id}: {e}")
#             continue

#         lines = [
#             "[CLEAN TEXT]",
#             clean_text,
#             "#" * 100,
#             "[SYMBOL MAPPING]"
#         ]
#         for k, v in sym_mapping.items():
#             lines.append(f"{k} -> {v}")
#         lines.append("#" * 100)

#         lines.append("[EQUATION MAPPING]")
#         for k, v in eqn_mapping.items():
#             lines.append(f"{k} -> {v}")
#         lines.append("#" * 100)

#         lines.append("[SYMBOLS IN EQUATIONS]")
#         for eq_ph, syms in eq_to_syms.items():
#             lines.append(f"{eq_ph} -> {syms}")
#             for s in syms:
#                 lines.append(f"    {s} : {sym_mapping[s]}")
#         lines.append("#" * 100)

#         lines.append("[AUDIT TRAIL]")
#         for method, messages in extractor.audit.items():
#             lines.append(f"{method}:")
#             for msg in messages:
#                 lines.append(f"    {msg}")

#         out_path = out_dir / f"{paper_id}.txt"
#         out_path.write_text("\n".join(lines), encoding="utf-8")
#         print(f"[OK] {paper_id} -> {out_path}")