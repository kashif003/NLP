import os
import re
from bs4 import BeautifulSoup

SKIP_PATTERNS = [
    re.compile(r'^[a-zA-Z]$'),
    re.compile(r'^\d+$'),
    re.compile(r'^[=<>+\-*/,.\(\)]+$'),
    re.compile(r'^[A-Z]{1,4}$'),   # bare subscript labels like AB, XY
]

MIN_DEFINITION_SCORE = 2

MAX_EQUATIONS = 7


class HTMLReader:
    def __init__(self, paper_ID):
        self.paper_ID = paper_ID
        self.html_path = os.path.join("data/html_source", f"{paper_ID}.html")
        self.soup = self._parse_html()
        self._all_paragraphs = self.soup.find_all("p")
        self.equations = self._find_equations()
        self.all_symbols = self._get_paragraph_symbols()
        self.equation_symbols = self._match_symbols_to_equations()

    def _parse_html(self):
        """Parse HTML file into BeautifulSoup object."""
        with open(self.html_path, "r", encoding="utf-8", errors="ignore") as f:
            return BeautifulSoup(f.read(), "html.parser")

    def _find_equations(self):
        """
        Extract enumerated equations from the HTML.

        Returns
        -------
        dict
            Keys are global counters, values are dicts with eq_id, real_id, latex.
        """
        equations = {}
        current_prefix = None
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
                    continue
                prefix = match.group(1)
                if current_prefix and prefix != current_prefix:
                    section_counter = 1
                current_prefix = prefix
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
        """Check if a symbol should be skipped based on skip patterns or complex expression heuristics."""
        if any(p.match(alt) for p in SKIP_PATTERNS):
            return True
        if alt.startswith('(') or alt.startswith('['):
            return True
        if '=' in alt:
            return True
        if re.search(r'\\in(?![a-zA-Z])|\\rightarrow|\\to(?![a-zA-Z])|\\gets', alt):
            return True
        return False

    def _get_paragraph_symbols(self):
        """
        Extract all math symbols appearing in paragraphs.

        Returns
        -------
        dict
            Keys are symbol alttext, values are lists of math tag ids.
        """
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
        """
        Match paragraph symbols to equations they appear in.

        Returns
        -------
        dict
            Keys are symbols, values are dicts with equations and unique_ids.
        """
        matched = {}
        for eq in self.equations.values():
            eq_latex, eq_id = eq["latex"], eq["eq_id"]
            for symbol, math_ids in self.all_symbols.items():
                # use negative lookahead to avoid matching \rho inside \rho_{AB}
                pattern = re.escape(symbol) + r'(?![_\^{}\[\]a-zA-Z])'
                if re.search(pattern, eq_latex):
                    if symbol not in matched:
                        matched[symbol] = {"equations": [], "unique_ids": math_ids}
                    if eq_id not in matched[symbol]["equations"]:
                        matched[symbol]["equations"].append(eq_id)
        return matched

    def _extract_text_from_container(self, container):
        """
        Copy container, replace math tags with alttext, return plain text.

        Parameters
        ----------
        container : bs4.element.Tag
            HTML container element.

        Returns
        -------
        str
            Plain text with math replaced by alttext.
        """
        container_copy = BeautifulSoup(str(container), "html.parser")
        for tag in container_copy.find_all("math"):
            tag.replace_with(tag.get("alttext", ""))
        return container_copy.get_text(" ", strip=True)

    def _find_container(self, tag):
        """
        Walk up the tree to find the nearest meaningful container.

        Parameters
        ----------
        tag : bs4.element.Tag
            Starting HTML tag.

        Returns
        -------
        bs4.element.Tag or None
            Nearest meaningful container or None.
        """
        for parent in tag.parents:
            if parent.name in ["p", "td", "li", "div", "section",
                                "span", "dd", "dt", "blockquote", "figcaption"]:
                return parent
        return None

    def _get_surrounding_paragraphs(self, uid, window=2):
        """
        Get surrounding paragraphs around the paragraph containing the math tag.
        Looks window paragraphs before and after.

        Parameters
        ----------
        uid : str
            Math tag id.
        window : int
            Number of paragraphs to look before and after.

        Returns
        -------
        list of str
            List of paragraph texts surrounding the symbol occurrence.
        """
        m = self.soup.find(attrs={"id": uid})
        if not m:
            return []

        # find the containing paragraph
        container = None
        for parent in m.parents:
            if parent.name == "p":
                container = parent
                break
        if not container:
            return []

        try:
            idx = self._all_paragraphs.index(container)
        except ValueError:
            return []

        # get window paragraphs before and after
        start = max(0, idx - window)
        end = min(len(self._all_paragraphs), idx + window + 1)
        surrounding = self._all_paragraphs[start:end]

        return [self._extract_text_from_container(p) for p in surrounding]

    def _score_sentence(self, sentence, symbol):
        """
        Score a sentence by how likely it defines the symbol.
        Higher score = more likely to be a definition sentence.

        Parameters
        ----------
        sentence : str
            Candidate sentence containing the symbol.
        symbol : str
            The latex symbol being defined.

        Returns
        -------
        float
            Score value.
        """
        score = 0
        tokens = sentence.split()

        # find symbol position in tokens
        sym_idx = next((i for i, t in enumerate(tokens) if symbol in t), None)
        if sym_idx is None:
            return float("-inf")

        # strong bonus: defining verb within 5 tokens of symbol
        window = tokens[max(0, sym_idx - 5): min(len(tokens), sym_idx + 6)]
        for i, w in enumerate(window):
            abs_idx = max(0, sym_idx - 5) + i
            wl = w.lower()
            if wl in {"denotes", "denote", "defined", "called", "termed"}:
                score += 6
            elif wl in {"represents", "represent", "let"}:
                score += 4
            elif wl in {"is", "are"} and abs_idx > sym_idx:
                # only reward "SYM is/are X" when followed by an article (noun phrase)
                next_w = tokens[abs_idx + 1].lower() if abs_idx + 1 < len(tokens) else ""
                if next_w in {"the", "a", "an", "called", "defined", "known"}:
                    score += 4
                elif next_w in {"infinite", "zero", "finite", "positive", "negative",
                                 "bounded", "orthogonal", "equal", "arbitrary", "fixed"}:
                    score -= 3

        # bonus: definition keywords anywhere in sentence
        if re.search(r'\bwhere\b', sentence, re.IGNORECASE):              score += 3
        if re.search(r'\bis\s+defined\s+as\b', sentence, re.IGNORECASE):  score += 5
        if re.search(r'\bstands?\s+for\b', sentence, re.IGNORECASE):      score += 4
        if re.search(r'\brefers?\s+to\b', sentence, re.IGNORECASE):       score += 4

        # penalize math-heavy sentences — symbol likely used not defined
        math_tokens = re.findall(r'\\[a-zA-Z]+', sentence)
        score -= len(math_tokens) * 0.5

        # penalize long sentences
        if len(tokens) > 40:
            score -= 3

        # bonus for short focused sentences
        if len(tokens) < 20:
            score += 2

        # penalize if symbol appears inside a larger expression
        # i.e. surrounded by other math tokens on both sides
        left = tokens[max(0, sym_idx - 1)] if sym_idx > 0 else ""
        right = tokens[min(len(tokens) - 1, sym_idx + 1)] if sym_idx < len(tokens) - 1 else ""
        if re.search(r'[\\{}_^,]', left) and re.search(r'[\\{}_^,]', right):
            score -= 4

        # penalize conditional / non-definitional patterns
        if re.search(r'\bwhenever\b', sentence, re.IGNORECASE):
            score -= 3
        sym_placeholder = sentence.replace(symbol, "SYM")
        if re.search(r'SYM\s+(?:is|are)\s+(?:infinite|zero|finite|positive|negative|'
                     r'bounded|orthogonal|equal|arbitrary|fixed)\b',
                     sym_placeholder, re.IGNORECASE):
            score -= 4

        return score

    def _collect_candidate_sentences(self, symbol, uid):
        """
        Collect all candidate sentences for a symbol from its paragraph
        and surrounding paragraphs.

        Parameters
        ----------
        symbol : str
            The latex symbol string.
        uid : str
            The math tag id.

        Returns
        -------
        list of str
            All sentences containing the symbol.
        """
        candidates = []

        # sentences from own paragraph
        m = self.soup.find(attrs={"id": uid})
        if m:
            container = self._find_container(m)
            if container:
                text = self._extract_text_from_container(container)
                for s in re.split(r'(?<=[.!?])\s+', text):
                    if symbol in s:
                        candidates.append(s)

        # sentences from surrounding paragraphs
        for para_text in self._get_surrounding_paragraphs(uid, window=2):
            for s in re.split(r'(?<=[.!?])\s+', para_text):
                if symbol in s and s not in candidates:
                    candidates.append(s)

        return candidates

    def get_symbol_full_context(self):
        """
        Get the best scored defining sentence for each symbol
        across all its occurrences and surrounding paragraphs.

        Returns
        -------
        dict
            Keys are symbols, values are dicts with equations and context sentence.
        """
        result = {}
        for symbol, data in self.equation_symbols.items():
            best_sentence = None
            best_score = float("-inf")

            for uid in data["unique_ids"]:
                candidates = self._collect_candidate_sentences(symbol, uid)
                for sentence in candidates:
                    score = self._score_sentence(sentence, symbol)
                    if score > best_score:
                        best_score = score
                        best_sentence = sentence

            if best_score < MIN_DEFINITION_SCORE:
                best_sentence = None
            result[symbol] = {
                "equations": data["equations"],
                "context": best_sentence
            }
        return result

    def get_data(self):
        """
        Build final result: equation latex → list of (symbol, context) tuples.

        Returns
        -------
        dict
            Keys are equation latex strings, values are lists of (symbol, context).
        """
        full_context = self.get_symbol_full_context()
        result = {}

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
    import json

    paper_ids = ["2410.08937"]

    all_data = {}
    for paper_id in paper_ids:
        reader = HTMLReader(paper_id)
        data = reader.get_data()
        all_data.update(data)

        for equation, symbols in data.items():
            print("[EQUATION:]")
            print(equation)
            print("Symbols:")
            for symbol, ctx in symbols:
                print(f"  {symbol}: {ctx}")
            print()

    output_filename = "equations_data.json"
    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(all_data, f, indent=4, ensure_ascii=False)

    print(f"Data saved to {output_filename}")
    print("done")