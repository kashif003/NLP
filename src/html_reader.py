import os
import re
from bs4 import BeautifulSoup, NavigableString

SKIP_PATTERNS = [
    re.compile(r'^[a-zA-Z]$'),
    re.compile(r'^\d+$'),
    re.compile(r'^[=<>+\-*/,.\(\)]+$'),
    re.compile(r'^[A-Z]{1,4}$'),
]

MIN_DEFINITION_SCORE = -1
MAX_EQUATIONS = 7

EQUATION_BLOCK_CLASSES = {
    "ltx_equationgroup",
    "ltx_equation",
    "ltx_eqn_row",
    "ltx_eqn_table",
}


class HTMLReader:
    def __init__(self, paper_ID):
        self.paper_ID = paper_ID
        self.html_path = os.path.join("data/html_source", f"{paper_ID}.html")
        self.soup = self._parse_html()
        self._all_paragraphs = self.soup.find_all("p")
        self._para_index = {p: i for i, p in enumerate(self._all_paragraphs)}
        self.equations = self._find_equations()
        self.all_symbols = self._get_paragraph_symbols()
        self.equation_symbols = self._match_symbols_to_equations()

    def _parse_html(self):
        """Parse HTML file into BeautifulSoup object."""
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
        Extract enumerated equations from the HTML. For multi-line equation
        blocks, walks up to the block root and collects latex from all lines
        not just the tagged line. Skips duplicate blocks already processed.

        Returns
        -------
        dict
            Keys are global counters, values are dicts with eq_id, real_id, latex.
        """
        equations = {}
        current_prefix = None
        section_counter = 1
        global_counter = 1
        seen_roots = set()

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

            # walk up to block root to collect ALL lines of multi-line equation
            block_root = self._get_block_root(parent) if parent else parent
            root_id = id(block_root)

            # skip if this block was already processed
            if root_id in seen_roots:
                continue
            seen_roots.add(root_id)

            # collect latex from ALL math tags in the entire block
            latex = " ".join(
                m.get("alttext")
                for m in block_root.find_all("math")
                if m.get("alttext")
            )

            equations[global_counter] = {
                "eq_id": mapped_id,
                "real_id": eq_id,
                "latex": str(latex)
            }
            global_counter += 1

        return equations

    def _should_skip_symbol(self, alt):
        """
        Check if a symbol should be skipped based on skip patterns
        or complex expression heuristics.

        Parameters
        ----------
        alt : str
            Symbol alttext from math tag.

        Returns
        -------
        bool
            True if symbol should be skipped.
        """
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

        container = None
        for parent in m.parents:
            if parent.name == "p":
                container = parent
                break
        if not container:
            return []

        idx = self._para_index.get(container, -1)
        if idx == -1:
            return []

        start = max(0, idx - window)
        end = min(len(self._all_paragraphs), idx + window + 1)
        surrounding = self._all_paragraphs[start:end]

        return [self._extract_text_from_container(p) for p in surrounding]

    def _score_sentence(self, sentence, symbol):
        """
        Score a sentence by how likely it defines the symbol.
        Higher score = more likely to be a definition sentence.
        Includes bonus for juxtaposition pattern where symbol appears
        immediately after a meaningful noun phrase.

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

        sym_idx = next((i for i, t in enumerate(tokens) if symbol in t), None)
        if sym_idx is None:
            return float("-inf")

        window = tokens[max(0, sym_idx - 5): min(len(tokens), sym_idx + 6)]
        for i, w in enumerate(window):
            abs_idx = max(0, sym_idx - 5) + i
            wl = w.lower()
            if wl in {"denotes", "denote", "defined", "called", "termed"}:
                score += 6
            elif wl in {"represents", "represent", "let"}:
                score += 4
            elif wl in {"is", "are"} and abs_idx > sym_idx:
                next_w = tokens[abs_idx + 1].lower() if abs_idx + 1 < len(tokens) else ""
                if next_w in {"the", "a", "an", "called", "defined", "known"}:
                    score += 4
                elif next_w in {"infinite", "zero", "finite", "positive", "negative",
                                 "bounded", "orthogonal", "equal", "arbitrary", "fixed"}:
                    score -= 3

        if re.search(r'\bwhere\b', sentence, re.IGNORECASE):              score += 3
        if re.search(r'\bis\s+defined\s+as\b', sentence, re.IGNORECASE):  score += 5
        if re.search(r'\bstands?\s+for\b', sentence, re.IGNORECASE):      score += 4
        if re.search(r'\brefers?\s+to\b', sentence, re.IGNORECASE):       score += 4

        math_tokens = re.findall(r'\\[a-zA-Z]+', sentence)
        score -= len(math_tokens) * 0.5

        if len(tokens) > 40:
            score -= 3
        if len(tokens) < 20:
            score += 2

        left = tokens[max(0, sym_idx - 1)] if sym_idx > 0 else ""
        right = tokens[min(len(tokens) - 1, sym_idx + 1)] if sym_idx < len(tokens) - 1 else ""
        if re.search(r'[\\{}_^,]', left) and re.search(r'[\\{}_^,]', right):
            score -= 4

        if re.search(r'\bwhenever\b', sentence, re.IGNORECASE):
            score -= 3
        sym_placeholder = sentence.replace(symbol, "SYM")
        if re.search(r'SYM\s+(?:is|are)\s+(?:infinite|zero|finite|positive|negative|'
                     r'bounded|orthogonal|equal|arbitrary|fixed)\b',
                     sym_placeholder, re.IGNORECASE):
            score -= 4

        # bonus: symbol immediately after meaningful noun (definition by juxtaposition)
        # e.g. "vector potential \mathbf{A}(t)" → clear definition pattern
        NON_DEFINITION_WORDS = {
            "where", "when", "thus", "hence", "then", "and", "or",
            "but", "the", "a", "an", "if", "as", "by", "with"
        }
        if sym_idx > 0:
            left_word = tokens[sym_idx - 1].lower()
            if (re.search(r'[a-zA-Z]{3,}', left_word) and
                    left_word not in NON_DEFINITION_WORDS):
                score += 3

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

        m = self.soup.find(attrs={"id": uid})
        if m:
            container = self._find_container(m)
            if container:
                text = self._extract_text_from_container(container)
                for s in re.split(r'(?<=[.!?])\s+', text):
                    if symbol in s:
                        candidates.append(s)

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
        Build final result: equation latex to list of (symbol, context) tuples.

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

    def _is_trivial_sentence(self, sentence):
        """
        Check if a sentence is trivial or transitional and not useful
        for equation meaning extraction.

        Parameters
        ----------
        sentence : str
            Sentence to check.

        Returns
        -------
        bool
            True if sentence is trivial and should be skipped.
        """
        s = sentence.strip()

        if len(s.split()) < 5:
            return True

        english_words = re.findall(r'\b[a-zA-Z]{2,}\b', s)
        if len(english_words) < 3:
            return True

        trivial_patterns = [
            r'^then[,\s]',
            r'^where[,\s]',
            r'^hence[,\s]',
            r'^thus[,\s]',
            r'^note that[,\s]',
            r'^observe that[,\s]',
            r'^we have[,\s]',
            r'^it follows[,\s]',
            r'^substituting',
            r'^combining',
            r'^plugging',
            r'^\u220e',
            r'^proof',
        ]
        for pattern in trivial_patterns:
            if re.match(pattern, s, re.IGNORECASE):
                return True

        return False

    def _is_real_text(self, sentence):
        """
        Check if sentence contains enough English words to be real text
        and not just latex or math content.

        Parameters
        ----------
        sentence : str
            Sentence to check.

        Returns
        -------
        bool
            True if sentence contains at least 3 real English words.
        """
        english_words = re.findall(r'\b[a-zA-Z]{3,}\b', sentence)
        return len(english_words) >= 3

    def _collect_text_before_eq(self, parent, eq_tag):
        """
        Walk all siblings before eq_tag inside parent, extracting plain text
        from every node type: NavigableString, math tags replaced by alttext,
        anchor tags, spans, and any other inline elements. Returns empty string
        if result contains no real English words indicating pure math content.

        Parameters
        ----------
        parent : bs4.element.Tag
            Parent element containing eq_tag.
        eq_tag : bs4.element.Tag
            The equation tag to stop at.

        Returns
        -------
        str
            Concatenated plain text of all content before eq_tag,
            or empty string if result is pure latex/math.
        """
        text_parts = []
        for sibling in parent.children:
            if sibling == eq_tag:
                break

            if isinstance(sibling, NavigableString):
                text_parts.append(str(sibling))
            elif sibling.name == "math":
                text_parts.append(sibling.get("alttext", ""))
            else:
                inner = BeautifulSoup(str(sibling), "html.parser")
                for m in inner.find_all("math"):
                    m.replace_with(m.get("alttext", ""))
                text_parts.append(inner.get_text(" ", strip=False))

        result = re.sub(r'\s+', ' ', " ".join(text_parts)).strip()

        # reject if no real English words — pure latex/math block
        if not self._is_real_text(result):
            return ""

        return result

    def _get_physical_location_sentence(self, real_id):
        """
        Collect the last meaningful sentence immediately before the equation
        block in the HTML. Walks up to the root of the full multi-line equation
        block first, then checks text before the block in the same parent.
        If the text before the block is a sentence fragment (does not start
        with a capital letter), prepends the last sentence of the previous
        paragraph to complete it. Falls back to walking previous paragraphs
        if no text found in same parent. Appends [EQUATION] to mark position.

        Parameters
        ----------
        real_id : str
            Real HTML id of the equation tag.

        Returns
        -------
        str or None
            Last sentence before equation with [EQUATION] appended,
            or None if no real text found before equation.
        """
        def extract_sentences(text):
            """Split text into sentences."""
            return [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]

        def best_sentence(sents):
            """
            Return last meaningful real-text sentence from list,
            falling back to last real-text sentence if all are trivial.

            Parameters
            ----------
            sents : list of str
                Sentences to filter.

            Returns
            -------
            str or None
                Best sentence or None if no real text found.
            """
            meaningful = [s for s in sents
                          if not self._is_trivial_sentence(s) and self._is_real_text(s)]
            if meaningful:
                return meaningful[-1]
            fallback = [s for s in sents if self._is_real_text(s)]
            return fallback[-1] if fallback else None

        def is_fragment(text):
            """
            Check if text is a sentence fragment by testing whether it
            starts with a lowercase letter, meaning the sentence began
            in the previous paragraph.

            Parameters
            ----------
            text : str
                Text to check.

            Returns
            -------
            bool
                True if text appears to be a fragment.
            """
            stripped = text.strip()
            return bool(stripped) and stripped[0].islower()

        eq_tag = self.soup.find(id=real_id)
        if not eq_tag:
            return None

        # walk up to root of full multi-line equation block
        block_root = self._get_block_root(eq_tag)

        # --- text before block root in same parent ---
        parent = block_root.parent
        if parent:
            before_text = self._collect_text_before_eq(parent, block_root)
            before_text = re.sub(r'\s+', ' ', before_text).strip()

            if before_text:
                # if fragment, prepend last sentence of previous paragraph
                if is_fragment(before_text):
                    prev_para = block_root.find_previous("p")
                    if prev_para:
                        prev_text = self._extract_text_from_container(prev_para)
                        prev_sents = extract_sentences(prev_text)
                        real_prev = [s for s in prev_sents if self._is_real_text(s)]
                        if real_prev:
                            before_text = real_prev[-1] + " " + before_text

                sents = extract_sentences(before_text)
                picked = best_sentence(sents)
                if picked:
                    return picked + " [EQUATION]"

        # --- fallback: walk previous paragraphs until real text found ---
        prev_para = block_root.find_previous("p")
        while prev_para:
            text = self._extract_text_from_container(prev_para)
            sents = extract_sentences(text)
            picked = best_sentence(sents)
            if picked:
                return picked + " [EQUATION]"
            prev_para = prev_para.find_previous("p")

        return None

    def get_equation_contexts(self):
        """
        Get context sentence for all equations using only the physical
        location of the equation in HTML. Extracts the last meaningful
        real-text sentence from immediately before the equation block,
        with [EQUATION] appended to mark the equation position.

        Returns
        -------
        dict
            Keys are eq_id (mapped), values are dicts with:
            - 'context': sentence string with [EQUATION] or None
            - 'referenced': False, reserved for future reference-based extraction
        """
        result = {}
        for counter, eq in self.equations.items():
            eq_id = eq["eq_id"]
            real_id = eq["real_id"]

            if not real_id:
                result[eq_id] = {"context": None, "referenced": False}
                continue

            context = self._get_physical_location_sentence(real_id)
            result[eq_id] = {"context": context, "referenced": False}

        return result


if __name__ == "__main__":
    paper_ids = ["2510.12545"]

    for paper_id in paper_ids:
        reader = HTMLReader(paper_id)
        eq_contexts = reader.equations
        for k, v in eq_contexts.items():
          print(k)
          print(v)
