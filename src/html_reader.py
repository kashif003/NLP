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
        self._para_index = {p: i for i, p in enumerate(self._all_paragraphs)}
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
                # negative lookahead to avoid matching \rho inside \rho_{AB}
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

    def _extract_text_with_mention(self, paragraph, real_id):
        """
        Extract text from paragraph replacing the equation anchor with [MENTION].

        Parameters
        ----------
        paragraph : bs4.element.Tag
            Paragraph HTML tag.
        real_id : str
            Real HTML id of the equation to replace with [MENTION].

        Returns
        -------
        str
            Plain text with equation reference replaced by [MENTION]
            and all math tags replaced by their alttext.
        """
        para_copy = BeautifulSoup(str(paragraph), "html.parser")

        # replace equation anchor with [MENTION]
        for a in para_copy.find_all("a", href=lambda h: h and (
            h == f"#{real_id}" or h.endswith(f"#{real_id}")
        )):
            a.replace_with("[MENTION]")

        # replace math tags with alttext
        for tag in para_copy.find_all("math"):
            tag.replace_with(tag.get("alttext", ""))

        return para_copy.get_text(" ", strip=True)

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

        # too short to be meaningful
        if len(s.split()) < 5:
            return True

        # too few english words — mostly math
        english_words = re.findall(r'\b[a-zA-Z]{2,}\b', s)
        if len(english_words) < 3:
            return True

        # common trivial/transitional patterns
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
            r'^\u220e',   # tombstone ∎
            r'^proof',
        ]
        for pattern in trivial_patterns:
            if re.match(pattern, s, re.IGNORECASE):
                return True

        return False

    def _get_referenced_sentences(self, real_id):
        """
        Find all sentences that reference an equation by its HTML id.
        For each unique reference location collect:
        - sentence before the reference  (key 1)
        - sentence containing reference  (key 2) with [MENTION] replacing the ref
        - sentence after the reference   (key 3)
        Duplicate paragraph locations are skipped.

        Handles both relative hrefs (#S1.E1) and absolute URLs
        (https://arxiv.org/html/...#S1.E1).

        Parameters
        ----------
        real_id : str
            Real HTML id of the equation (e.g. 'S1.E1')

        Returns
        -------
        list of dict
            Each dict is {1: sent_before, 2: sent_with_mention, 3: sent_after}
            Empty list if equation is not referenced.
        """
        all_groups = []
        seen_para_indices = set()

        def extract_sentences(paragraph):
            """Split paragraph plain text into sentences."""
            if not paragraph:
                return []
            text = self._extract_text_from_container(paragraph)
            return [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]

        anchors = self.soup.find_all(
            "a",
            href=lambda h: h and (
                h == f"#{real_id}" or
                h.endswith(f"#{real_id}")
            )
        )

        if not anchors:
            return []

        for anchor in anchors:
            para = None
            for parent in anchor.parents:
                if parent.name == "p":
                    para = parent
                    break
            if not para:
                continue

            idx = self._para_index.get(para, -1)
            if idx == -1:
                continue

            if idx in seen_para_indices:
                continue
            seen_para_indices.add(idx)

            group = {}

            # 1 — last sentence of previous paragraph
            if idx > 0:
                prev_sents = extract_sentences(self._all_paragraphs[idx - 1])
                if prev_sents:
                    group[1] = prev_sents[-1]

            # 2 — current paragraph with equation reference replaced by [MENTION]
            group[2] = self._extract_text_with_mention(para, real_id)

            # 3 — first sentence of next paragraph
            if idx + 1 < len(self._all_paragraphs):
                next_sents = extract_sentences(self._all_paragraphs[idx + 1])
                if next_sents:
                    group[3] = next_sents[0]

            if group:
                all_groups.append(group)

        return all_groups

    def _get_physical_location_sentences(self, real_id):
        """
        Collect sentences around the physical location of the equation in HTML.
        Used when equation is not referenced anywhere.
        Gets last 2 non-trivial sentences from paragraph before and first 2
        non-trivial sentences from paragraph after the equation tag.
        Falls back to unfiltered sentences if no meaningful ones are found.
        [MENTION] is placed after the before-sentences to mark equation location.

        Parameters
        ----------
        real_id : str
            Real HTML id of the equation tag.

        Returns
        -------
        list of dict
            Single group with numbered sentences and [MENTION] after before-sentences.
        """
        def extract_sentences(paragraph):
            """Split paragraph text into sentences."""
            if not paragraph:
                return []
            text = self._extract_text_from_container(paragraph)
            return [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]

        def get_best_sentences(sents, n, from_end=False):
            """
            Get up to n meaningful sentences, falling back to unfiltered if needed.

            Parameters
            ----------
            sents : list of str
                All sentences from paragraph.
            n : int
                Number of sentences to return.
            from_end : bool
                If True take from end, else from start.

            Returns
            -------
            list of str
                Up to n sentences.
            """
            meaningful = [s for s in sents if not self._is_trivial_sentence(s)]
            # use meaningful if available, else fall back to all sentences
            pool = meaningful if meaningful else sents
            return pool[-n:] if from_end else pool[:n]

        eq_tag = self.soup.find(id=real_id)
        if not eq_tag:
            return []

        group = {}
        key = 1

        # last 2 sentences from paragraph before equation
        prev_para = eq_tag.find_previous("p")
        if prev_para:
            prev_sents = extract_sentences(prev_para)
            for s in get_best_sentences(prev_sents, 2, from_end=True):
                group[key] = s
                key += 1

        # [MENTION] marks where the equation is
        group["[MENTION]"] = "[MENTION]"

        # first 2 sentences from paragraph after equation
        next_para = eq_tag.find_next("p")
        if next_para:
            next_sents = extract_sentences(next_para)
            for s in get_best_sentences(next_sents, 2, from_end=False):
                group[key] = s
                key += 1

        return [group] if group else []

    def get_equation_contexts(self):
        """
        Get context sentences for all equations.
        - If equation is referenced: use referenced sentences, one group per reference
        - If equation is not referenced: use physical location sentences with [MENTION]

        Returns
        -------
        dict
            Keys are eq_id (mapped), values are dicts with:
            - 'groups': list of dicts {1: sent, 2: sent, '[MENTION]': '[MENTION]', ...}
            - 'referenced': bool, True if equation was referenced in text
        """
        result = {}
        for counter, eq in self.equations.items():
            eq_id = eq["eq_id"]
            real_id = eq["real_id"]

            if not real_id:
                result[eq_id] = {"groups": [], "referenced": False}
                continue

            ref_groups = self._get_referenced_sentences(real_id)
            if ref_groups:
                result[eq_id] = {"groups": ref_groups, "referenced": True}
            else:
                phys_groups = self._get_physical_location_sentences(real_id)
                result[eq_id] = {"groups": phys_groups, "referenced": False}

        return result


if __name__ == "__main__":
    paper_ids = ["2510.12545"]

    for paper_id in paper_ids:
        reader = HTMLReader(paper_id)
        eq_contexts = reader.get_equation_contexts()

        for counter, eq in reader.equations.items():
            eq_id = eq["eq_id"]
            latex = eq["latex"]
            ctx = eq_contexts.get(eq_id, {})
            groups = ctx.get("groups", [])
            referenced = ctx.get("referenced", False)

            print(f"\n{'='*60}")
            print(f"EQUATION {counter}: {latex}")
            print(f"Referenced: {referenced}")
            print(f"{'='*60}")
            for g_idx, group in enumerate(groups, 1):
                print(f"  [Reference {g_idx}]" if referenced else "  [Physical location]")

                # print before-sentences (integer keys), then [MENTION], then after-sentences
                int_keys = sorted([k for k in group.keys() if isinstance(k, int)])
                before_keys = int_keys[:2]
                after_keys = int_keys[2:]

                for k in before_keys:
                    print(f"    {k}. {group[k]}")

                if "[MENTION]" in group:
                    print(f"    [MENTION]")

                for k in after_keys:
                    print(f"    {k}. {group[k]}")