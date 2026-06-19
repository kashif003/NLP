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


class PaperTextExtractor:
    """
    Extracts clean text from arxiv HTML papers replacing enumerated equations
    with EQN1,EQN2... non-enumerated equations with TEMPEQN and inline
    math symbols with SYM1,SYM2... Returns clean text and a mapping of
    placeholders to their original latex.
    """

    def __init__(self, paper_id):
        self.paper_id = paper_id
        self.html_path = os.path.join("data/html_source", f"{paper_id}.html")
        self.soup = self._parse_html()

        # counters and mappings
        self._sym_counter = 1
        self.eqn_mapping = {}   # "EQN1" -> {"latex": ..., "real_id": ...}
        self.sym_mapping = {}   # "SYM1"  -> latex string
        self._sym_seen = {}     # latex string -> placeholder (dedup)
        self._eqn_seen = {}     # real_id -> placeholder (dedup)

        # authoritative equation set, mirrors HTMLReader._find_equations
        self.equations = self._find_equations()   # real_id -> {number, latex}

    def _parse_html(self):
        """
        Parse HTML file into BeautifulSoup object.

        Returns
        -------
        bs4.BeautifulSoup
            Parsed HTML document.
        """
        with open(self.html_path, "r", encoding="utf-8", errors="ignore") as f:
            return BeautifulSoup(f.read(), "html.parser")

    def _get_block_root(self, tag):
        """
        Walk up from tag to the root of a multi-line equation block.
        Stops when the parent is no longer an equation-type container.

        Parameters
        ----------
        tag : bs4.element.Tag
            Starting tag.

        Returns
        -------
        bs4.element.Tag
            Topmost equation block container, or tag itself if not
            part of a multi-line block.
        """
        current = tag
        while current.parent:
            parent_classes = set(current.parent.get("class") or [])
            if parent_classes & EQUATION_BLOCK_CLASSES:
                current = current.parent
            else:
                break
        return current

    def _find_equations(self):
        """
        Find enumerated equations using the exact same traversal, prefix
        validation, block-root deduplication and MAX_EQUATIONS limit as
        HTMLReader._find_equations. For each equation it stores the displayed
        paper number (from the ltx_tag_equation span) and the full latex
        collected from every math tag in the block root. This guarantees
        PaperTextExtractor finds exactly the same equations — and the same
        latex — as HTMLReader.

        Returns
        -------
        dict
            Keys are real HTML id strings, values are dicts with:
            'number' : displayed equation number e.g. "1", "A.1"
            'latex'  : full latex string of the equation block
        """
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

        return equations

    def _get_equation_latex(self, block_root):
        """
        Extract full latex string from all math tags in an equation block.

        Parameters
        ----------
        block_root : bs4.element.Tag
            Root of the equation block.

        Returns
        -------
        str
            Latex string of the full equation.
        """
        return " ".join(
            m.get("alttext", "")
            for m in block_root.find_all("math")
            if m.get("alttext")
        )

    def _get_eqn_placeholder(self, real_id):
        """
        Get or create a placeholder for an equation. Equations found by
        _find_equations get EQN<N> using the paper's displayed number;
        all others get TEMPEQN. Latex is taken from the precomputed
        equation set so it always matches HTMLReader. Deduplicates so the
        same equation always gets the same placeholder.

        Parameters
        ----------
        real_id : str
            Real HTML id of the equation.

        Returns
        -------
        str
            Placeholder string e.g. "EQN1" or "TEMPEQN".
        """
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
        """
        Get or create a placeholder for an inline math symbol.
        Deduplicates so same symbol always gets same placeholder.

        Parameters
        ----------
        alttext : str
            Latex alttext of the math tag.

        Returns
        -------
        str
            Placeholder string e.g. "SYM1".
        """
        if alttext in self._sym_seen:
            return self._sym_seen[alttext]

        placeholder = f"SYM{self._sym_counter}"
        self._sym_counter += 1
        self.sym_mapping[placeholder] = alttext
        self._sym_seen[alttext] = placeholder
        return placeholder

    def _is_inside_equation(self, tag):
        """
        Check if a tag is inside an equation block and should not be
        processed as inline math.

        Parameters
        ----------
        tag : bs4.element.Tag
            Tag to check.

        Returns
        -------
        bool
            True if tag is inside an equation block.
        """
        for parent in tag.parents:
            parent_classes = set(parent.get("class") or [])
            if parent_classes & EQUATION_BLOCK_CLASSES:
                return True
        return False

    def _process_node(self, node):
        """
        Recursively process an HTML node and return its text with
        math and equations replaced by placeholders.

        Parameters
        ----------
        node : bs4.element.Tag or NavigableString
            Node to process.

        Returns
        -------
        str
            Processed text with placeholders.
        """
        # plain text node
        if isinstance(node, NavigableString):
            return str(node)

        # equation block — replace with placeholder
        node_classes = set(node.get("class") or [])
        if node_classes & EQUATION_BLOCK_CLASSES:
            # the equation id may be on the block node itself (single-line
            # equation, e.g. <table class="ltx_equation" id="S2.E1">) or on
            # an inner row (multi-line / grouped equation). check the node's
            # own id first, then its descendants, matching against the stored
            # equation set so we always pick the numbered line.
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

        # equation mention — an in-text reference link, where the href is
        # either a bare fragment (#S2.E1) or a full URL ending in the
        # fragment (https://arxiv.org/.../2506.19219v3#S5.E52). this applies
        # to ANY referenced equation, not only the tracked ones: the
        # displayed number lives in the link text, so we read it from there
        # (preferring the tracked number when available). non-equation links
        # fall through and are processed normally.
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
            # skip if inside equation block
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
        """
        Extract clean text from the paper with placeholders for equations
        and symbols. Also returns mappings of placeholders to latex.

        Returns
        -------
        tuple
            (clean_text, eqn_mapping, sym_mapping)

            clean_text : str
                Full paper text with EQN1,EQN2,TEMPEQN,SYM1...
            eqn_mapping : dict
                Keys are placeholder strings like "EQN1",
                values are dicts with 'latex' and 'real_id'.
            sym_mapping : dict
                Keys are placeholder strings like "SYM1",
                values are latex alttext strings.
        """
        body = self.soup.find("body") or self.soup

        raw_text = self._process_node(body)

        # collapse whitespace
        clean_text = re.sub(r'\n{3,}', '\n\n', raw_text)
        clean_text = re.sub(r' {2,}', ' ', clean_text)

        # tidy equation mentions: the "(", ")" and "Eq."/"Equation" wrapper
        # sit outside the link, so a raw reference renders as
        # "Eq. ( MEQN1 )". collapse it to just "MEQN1" — the
        # placeholder already reads as "equation 1".
        mention = r'MEQN[A-Za-z0-9]+(?:\.[A-Za-z0-9]+)*'
        # case 1: optional Eq word + parentheses around the mention
        clean_text = re.sub(
            r'(?:(?:Eqs?|Eqns?|Equations?)\.?\s*)?\(\s*(' + mention + r')\s*\)',
            r'\1',
            clean_text,
        )
        # case 2: leftover Eq word sitting directly before the mention
        clean_text = re.sub(
            r'(?:Eqs?|Eqns?|Equations?)\.?\s*(' + mention + r')',
            r'\1',
            clean_text,
        )

        clean_text = clean_text.strip()

        return clean_text, self.eqn_mapping, self.sym_mapping


def map_symbols_to_equations(eqn_mapping, sym_mapping):
    """
    Find which inline symbols appear in each equation, using boundary-aware
    matching so a single-letter symbol (e.g. 'e') does not falsely match
    inside a longer token (e.g. '\\eta', '\\int', or '\\mathrm{ATI}').

    Parameters
    ----------
    eqn_mapping : dict
        "EQN1" -> {"latex": str, "real_id": str}
    sym_mapping : dict
        "SYM1" -> latex string

    Returns
    -------
    dict
        "EQN1" -> list of symbol placeholders found in that equation.
    """
    # precompile a matcher per symbol
    matchers = {}
    for sym_ph, sym_latex in sym_mapping.items():
        if not sym_latex:
            continue
        if len(sym_latex) == 1 and sym_latex.isalpha():
            # single letter: must stand alone — not part of a longer
            # identifier or a \command (so 'e' won't match '\eta',
            # 'T'/'I' won't match '\mathrm{ATI}', 'i' won't match '\int')
            pattern = r"(?<![A-Za-z\\])" + re.escape(sym_latex) + r"(?![A-Za-z])"
        else:
            pattern = re.escape(sym_latex)
        matchers[sym_ph] = re.compile(pattern)

    result = {}
    for eq_ph, eq_data in eqn_mapping.items():
        eq_latex = eq_data.get("latex", "")
        result[eq_ph] = [
            sym_ph for sym_ph, rx in matchers.items() if rx.search(eq_latex)
        ]
    return result

if __name__ == "__main__":
    paper_id = "2404.04958"
    extractor = PaperTextExtractor(paper_id)
    clean_text, eqn_mapping, sym_mapping = extractor.extract()
    
    print("[CLEAN TEXT]")
    print(clean_text)
    print("#"*100)
    
    print("[SYMBOL MAPPING]")
    for k,v in sym_mapping.items():
        print(k, "->", v)
    
    print("#"*100)
    print("[EQUATION MAPPING]")
    for k,v in eqn_mapping.items():
        print(k, "->", v)

    print("#"*100)
    print("[SYMBOLS IN EQUATIONS]")
    eq_to_syms = map_symbols_to_equations(eqn_mapping, sym_mapping)
    for eq_ph, syms in eq_to_syms.items():
        print(eq_ph, "->", syms)
        for s in syms:
            print("   ", s, ":", sym_mapping[s])