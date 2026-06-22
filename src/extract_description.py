"""
Generic symbol / equation description extractor based on dependency parsing.

Reads the dependency tree and recovers the noun phrase the placeholder is
syntactically tied to. A few structural rules cover unlimited surface
phrasings, so it generalizes across papers far better than fixed frames.

Compliant with the no-prompting rule: spaCy's parser is discriminative (it
labels structure, it does not generate text). Run locally on DC1.07.

Setup (once):
    pip install spacy
    python -m spacy download en_core_web_sm      # or en_core_web_md (better)

Simple usage:
    from symbol_description_dep import get_meaning
    get_meaning(text, "EQN5")     # -> "the polarization-entangled two-photon state"

Detailed usage (head / relation / confidence):
    from symbol_description_dep import extract_symbol_description
    extract_symbol_description(text, "EQN5")
    # -> {'symbol','relation','description','head','rule','confidence'}
"""

import re

SYMBOL_RE = re.compile(r"^(?:SYM|EQN|MEQN)\d+$")
NOUN_POS = {"NOUN", "PROPN"}

DEF_VERB_LEMMAS = {
    "give", "define", "describe", "express", "write", "compute", "calculate",
    "obtain", "denote", "represent", "quantify", "yield", "capture", "read",
}
PREP_DEF = {"by", "as", "in", "from", "via", "through"}

# --- noise filters (Bucket B + C) -------------------------------------------
# pronouns that carry no standalone meaning ("it", "we", "they")
PRONOUNS = {"it", "we", "they", "i", "he", "she", "you", "one", "ones"}
# vague "shell" nouns: meaningless on their own ("output", "The factors")
VAGUE_HEADS = {
    "expression", "expressions", "equation", "relation", "relations",
    "form", "forms", "formula", "result", "results", "output", "outputs",
    "quantity", "quantities", "term", "terms", "value", "values",
    "function", "functions", "case", "cases", "thing", "things",
    "approach", "approaches", "factor", "factors",
}
# bare reference words with no number: "Fig", "Table", "Eq"
REF_WORDS = {
    "fig", "figs", "figure", "figures", "table", "tables", "tab",
    "eq", "eqn", "equation", "section", "sec", "panel", "panels",
    "appendix", "ref", "refs",
}
# leading enumeration label: "(b) non linearity" -> "non linearity"
_LABEL_RE = re.compile(r"^\(?[a-zA-Z0-9]{1,3}\)\s+")
# leading demonstrative: "this modified algebra" -> "modified algebra"
_DEMO_RE = re.compile(r"^(this|that|these|those)\s+", re.I)
_ART_RE = re.compile(r"^(the|a|an)\s+", re.I)
# orphaned ordinal artifact from symbol stripping: "The -th component"
_ORD_RE = re.compile(r"\s-(?:th|st|nd|rd)\b", re.I)
# dangling trailing preposition: "an increase in" -> "an increase"
_TAILPREP_RE = re.compile(
    r"\s+(in|of|to|on|for|with|by|from|as|at|into|over|between)$", re.I)
# float reference (matches plurals: "Figures 2b", "Tables 1", "Figs. 3")
_REF_RE = re.compile(
    r"^(figures?|figs?|tables?|tabs?|eqs?|eqns?|equations?|sections?|secs?|"
    r"appendix|app|panels?)\b\.?\s*\d", re.I)
# leaked citation macro / bib key: "autocitemcrae2020a", "mcrae2020a"
_CITE_RE = re.compile(r"^(?:auto)?cite\w+$|^[a-z]+\d{4}[a-z]?$", re.I)
# bare numeric / unit / measurement value: "1 10-9 mbar", "10 mK", "2.5 GHz"
_VALUE_RE = re.compile(r"^[\d.,\s×x*+\-/()]*\d[\d.,\s×x*+\-/()]*[a-zA-Z%]{0,4}$")
# short all-caps identifier with a digit: "LER1", "TLS2"
_ID_RE = re.compile(r"^[A-Z]{2,5}\d+[a-zA-Z]?$")

_NLP = None


def _get_nlp(model="en_core_web_sm"):
    global _NLP
    if _NLP is None:
        import spacy
        _NLP = spacy.load(model)
    return _NLP


def _name(ph, name_map):
    """
    Return the latex label of a placeholder for audit display.

    The token SEARCH still uses the placeholder (the clean text contains
    'SYM26'/'EQN1'); this only changes what is PRINTED in the audit. Falls back
    to the placeholder itself when no map is given or the key is missing.
    """
    if name_map is None:
        return ph
    return name_map.get(ph, ph)


def _is_symbol(tok):
    return bool(SYMBOL_RE.match(tok.text))


def _is_person(tok):
    """True if spaCy tagged this token as part of a PERSON entity (author name)."""
    return tok.ent_type_ == "PERSON"


def _is_pron(tok):
    """True if this token is a pronoun ('it', 'we', 'they')."""
    return tok.pos_ == "PRON"


def _is_verb(tok):
    """True if this token is a verb ('simplifies', 'reads')."""
    return tok.pos_ == "VERB"


def _bad_anchor(tok):
    """Unusable as a meaning anchor: a placeholder, a person, or a pronoun."""
    return _is_symbol(tok) or _is_person(tok) or _is_pron(tok)


def _clean_desc(desc):
    """
    Tidy a raw description string:
      - drop leading latex/paren junk: '\\autocite...' -> 'autocite...',
        '(material' -> 'material'
      - drop a leading enumeration label '(b) ' and a leading demonstrative
        'this ' ('this modified algebra' -> 'modified algebra')
      - remove an orphaned ordinal artifact ('The -th component' -> 'The component')
      - strip a dangling trailing preposition ('an increase in' -> 'an increase')
    """
    if not desc:
        return desc
    desc = desc.lstrip("\\([{ \t").strip()
    desc = _LABEL_RE.sub("", desc).strip()
    desc = _DEMO_RE.sub("", desc).strip()
    desc = _ART_RE.sub("", desc).strip()
    desc = _ORD_RE.sub("", desc).strip()
    desc = _TAILPREP_RE.sub("", desc).strip()
    return desc


def _is_meaningless(desc):
    """
    True if a (cleaned) description is not a real meaning: a bare placeholder,
    a pronoun, a vague shell noun, a bare/numbered figure reference, a citation
    key, a numeric/unit value, or a short all-caps identifier.
    """
    if not desc:
        return True
    low = desc.lower().strip()
    if SYMBOL_RE.match(desc):          # bare placeholder, e.g. "SYM24"
        return True
    if _REF_RE.match(low):             # "figures 2b", "table 1"
        return True
    if _CITE_RE.match(low):            # "autocitemcrae2020a", "mcrae2020a"
        return True
    if _VALUE_RE.match(desc):          # "1 10-9 mbar", "10 mK"
        return True
    if _ID_RE.match(desc):             # "LER1", "TLS2"
        return True
    # strip a leading article so "the output" is judged like "output"
    core = re.sub(r"^(the|a|an)\s+", "", low).strip()
    if core in PRONOUNS or core in VAGUE_HEADS or core in REF_WORDS:
        return True
    if low in PRONOUNS or low in REF_WORDS:
        return True
    return False


def _find_symbol_token(doc, symbol):
    for tok in doc:
        if tok.text == symbol:
            return tok
    return None


def _has_or_cc(tok):
    return any(ch.dep_ == "cc" and ch.lower_ == "or" for ch in tok.children)


def _find_or_alias(anchor):
    """
    Search the anchor's whole subtree for an author rename introduced by
    ', or <noun>' and return that noun. Robust to how the parser attaches the
    conjunct (we follow the 'or' token to its conjunct head). The comma
    requirement distinguishes a rename ('A, or B') from a true disjunction
    ('A or B').
    """
    doc = anchor.doc
    for t in anchor.subtree:
        if (t.dep_ == "cc" and t.lower_ == "or"
                and t.i - 1 >= 0 and doc[t.i - 1].text == ","):
            cand = t.head
            if cand.pos_ in NOUN_POS and not _is_symbol(cand):
                return cand
    return None


def _refine_anchor(anchor):
    """
    Prefer an author-introduced alias over a vague grammatical head:
    'the amount ... , or process fidelity' -> 'fidelity'.
    """
    alias = _find_or_alias(anchor)
    if alias is not None:
        return alias
    return anchor


def _anchor(tok):
    """
    Return (anchor_noun_token, relation, rule_name) for the noun phrase that
    describes the symbol. relation is 'denotes' (SYM *is* the quantity) or
    'computes' (EQN *yields* it). (None, None, None) if no structural anchor.
    """
    dep = tok.dep_
    head = tok.head

    # 1) passive defining clause: 'X is given/defined by SYM'
    if dep == "pobj" and head.lemma_.lower() in PREP_DEF:
        part = head.head
        if part.tag_ in {"VBN", "VBD"} or part.lemma_ in DEF_VERB_LEMMAS:
            for c in part.children:
                if c.dep_ in {"nsubjpass", "nsubj"} and c.pos_ in NOUN_POS:
                    return c, "computes", "passive_def"
            if part.dep_ in {"acl", "relcl"} and part.head.pos_ in NOUN_POS:
                return part.head, "computes", "passive_def_relcl"

    # 2) subject of copula / defining verb: 'SYM is the X' / 'SYM gives X'
    if dep in {"nsubj", "nsubjpass"}:
        verb = head
        for c in verb.children:
            if c.dep_ in {"attr", "oprd"} and c.pos_ in NOUN_POS:
                return c, "denotes", "copula"
        if verb.lemma_ in DEF_VERB_LEMMAS:
            for c in verb.children:
                if c.dep_ in {"dobj", "attr", "oprd"} and c.pos_ in NOUN_POS:
                    return c, "computes", "active_def"

    # 3) SYM has an appositive child: 'SYM, the X' / 'SYM (the X)'
    for c in tok.children:
        if c.dep_ == "appos" and c.pos_ in NOUN_POS:
            return c, "denotes", "appos_child"

    # 4) SYM attaches to a noun head: 'the X SYM' (trailing symbol)
    if dep in {"appos", "compound", "flat", "nmod", "nummod",
               "dep", "npadvmod", "conj"} and head.pos_ in NOUN_POS:
        return head, "denotes", "trailing_np"

    # 5) parser made the PROPN symbol the chunk head; grab its noun modifier
    noun_mods = [c for c in tok.children
                 if c.dep_ in {"compound", "amod", "nmod", "appos"}
                 and c.pos_ in NOUN_POS and not _is_symbol(c)]
    if noun_mods:
        return noun_mods[-1], "denotes", "head_flip"

    return None, None, None


def _short_chunk(chunk, symbol_i):
    """True if the chunk has at most one real (non-symbol) noun. Used to gate
    of-PP extension so we only extend thin, relational heads."""
    noun_ct = sum(1 for t in chunk
                  if t.pos_ in NOUN_POS and t.i != symbol_i and not _is_symbol(t))
    return noun_ct <= 1


def _extend_of_pp(doc, chunk, desc, chunks, symbol_i):
    """
    Append an immediately-following 'of'-PP: 'solutions' -> 'solutions of the
    time-dependent Schroedinger equation'. Gated to short heads so we don't
    drag tails onto already-descriptive NPs.
    """
    if not _short_chunk(chunk, symbol_i):
        return desc
    j = chunk.end
    if j < len(doc) and doc[j].lower_ == "of":
        for c2 in chunks:
            if c2.start == j + 1:
                extra = "".join(t.text_with_ws for t in c2
                                if t.i != symbol_i and not _is_symbol(t)).strip()
                if extra:
                    return (desc + " of " + extra).strip()
            if c2.start > j + 1:
                break
    return desc


def _prepend_of_governor(doc, chunk, desc, chunks, symbol_i):
    """
    If the chunk is the object of a preceding 'of', prepend the governing NP:
    'photons' in 'number of photons' -> 'number of photons'. Gated to short
    heads so we don't over-extend.
    """
    if not _short_chunk(chunk, symbol_i):
        return desc
    start = chunk.start
    if start - 1 >= 0 and doc[start - 1].lower_ == "of":
        for c2 in chunks:
            if c2.end == start - 1:          # chunk ending right before 'of'
                gov = "".join(t.text_with_ws for t in c2
                              if t.i != symbol_i and not _is_symbol(t)).strip()
                if gov:
                    return (gov + " of " + desc).strip()
    return desc


def _np_and_head(doc, anchor, symbol_i, token_to_chunk, chunks):
    """
    Resolve an anchor to (description_text, head_token). Uses the noun chunk
    that *contains* the anchor (falling back to the chunk containing the
    symbol), so trailing-symbol NPs are captured whole. Head selection is
    attachment-agnostic: if the chunk root is the symbol/number, use the
    rightmost real noun instead.

    Returns the head *token* (not text) so the caller can reject a non-noun
    head. Other placeholder tokens are dropped from the description text, and
    a neighbouring 'of'-phrase is stitched on in either direction.
    """
    chunk = token_to_chunk.get(anchor.i) or token_to_chunk.get(symbol_i)
    if chunk is None:
        # minimal build from the anchor and its left modifiers
        left = anchor
        while left.i - 1 >= 0:
            prev = doc[left.i - 1]
            if prev.dep_ in {"det", "amod", "compound", "nummod", "nmod"}:
                left = prev
            else:
                break
        toks = [doc[i] for i in range(left.i, anchor.i + 1)
                if i != symbol_i and not _is_symbol(doc[i])]
        return "".join(t.text_with_ws for t in toks).strip(), anchor

    head_tok = chunk.root
    if head_tok.i == symbol_i or head_tok.like_num or _is_symbol(head_tok):
        nouns = [t for t in chunk
                 if t.pos_ in NOUN_POS and t.i != symbol_i and not _is_symbol(t)]
        if nouns:
            head_tok = nouns[-1]
    elif anchor.pos_ in NOUN_POS and not _is_symbol(anchor):
        head_tok = anchor

    desc = "".join(t.text_with_ws for t in chunk
                   if t.i != symbol_i and not _is_symbol(t)).strip()
    desc = _extend_of_pp(doc, chunk, desc, chunks, symbol_i)
    desc = _prepend_of_governor(doc, chunk, desc, chunks, symbol_i)
    return desc, head_tok


def _fallback_nearest_np(doc, tok, symbol_i, chunks):
    """
    Last resort: nearest noun chunk to the left of the symbol (then right).
    Chunks rooted on a placeholder, a person name, a pronoun, or a verb are
    skipped. The chosen NP is extended with a neighbouring 'of'-phrase, so the
    fallback no longer returns bare truncations like 'solutions'. Returns the
    head *token*.
    """
    best = None
    for chunk in chunks:
        if (_is_symbol(chunk.root) or _is_person(chunk.root)
                or _is_pron(chunk.root) or _is_verb(chunk.root)):
            continue
        dist = tok.i - chunk.root.i
        key = (0 if dist > 0 else 1, abs(dist))
        if best is None or key < best[0]:
            best = (key, chunk)
    if best is None:
        return None, None
    chunk = best[1]
    desc = "".join(t.text_with_ws for t in chunk
                   if t.i != symbol_i and not _is_symbol(t)).strip()
    desc = _extend_of_pp(doc, chunk, desc, chunks, symbol_i)
    desc = _prepend_of_governor(doc, chunk, desc, chunks, symbol_i)
    return desc, chunk.root


def _latex_context(doc, name_map):
    """
    Build the audit KEY: the context text that was searched, with every
    placeholder swapped for its latex so the symbol code is visible.

    doc       : the parsed spaCy Doc; doc.text is the searched context, which
                still contains placeholders like 'SYM41'/'EQN2'.
    name_map  : placeholder -> latex dict, e.g. {'SYM41': '\\vec{\\lambda}'}.
    returns   : the context string with placeholders replaced by latex. If no
                name_map is given, doc.text is returned unchanged.

    Longer placeholders are replaced first so that 'SYM1' cannot corrupt
    'SYM12' (substring clash).
    """
    text = doc.text
    if not name_map:
        return text
    for ph in sorted(name_map, key=len, reverse=True):
        text = text.replace(ph, name_map[ph])
    return text


def extract_from_doc(doc, symbol, audit=None, name_map=None):
    """
    Core: run on an already-parsed spaCy Doc. Returns the full detail dict.

    Parameters
    ----------
    doc : spacy.tokens.Doc
        Parsed sentence/context containing the placeholder.
    symbol : str
        Placeholder to describe, e.g. "SYM77" or "EQN5". This is what we SEARCH
        for in the text (the text still contains placeholders).
    audit : dict, optional
        Flat audit dict (method_name -> list of messages). Records the chosen
        description, the rule, the confidence, and any rejected anchors/noise.
    name_map : dict, optional
        Placeholder -> latex map. When given, every placeholder PRINTED in the
        audit (the target symbol and any rejected anchor) is shown as its latex
        instead of "SYM77"/"EQN5". The search is unaffected.

    Returns
    -------
    dict
        Keys: symbol, relation, description, head, rule, confidence.
    """
    result = {"symbol": symbol, "relation": None, "description": None,
              "head": None, "rule": None, "confidence": None}
    tok = _find_symbol_token(doc, symbol)
    if tok is None:
        if audit is not None:
            audit.setdefault("extract_symbol_description", {})[
                _latex_context(doc, name_map)] = None
        return result

    # audit key: the ONE sentence containing the symbol, with latex swapped in
    context = _latex_context(tok.sent.as_doc(), name_map)

    chunks = list(doc.noun_chunks)
    token_to_chunk = {t.i: c for c in chunks for t in c}

    anchor, relation, rule = _anchor(tok)

    # reject a structural anchor that is a placeholder, a person, or a pronoun
    if anchor is not None and _bad_anchor(anchor):
        anchor = None

    if anchor is not None:
        anchor = _refine_anchor(anchor)
        desc, head = _np_and_head(doc, anchor, tok.i, token_to_chunk, chunks)
        desc = _clean_desc(desc)
        # accept only a real, noun-headed, non-noise description
        head_ok = head is None or head.pos_ in NOUN_POS
        if desc and head_ok and not _is_meaningless(desc):
            result.update(relation=relation, description=desc,
                          head=head.text if head is not None else None,
                          rule=rule, confidence="high")
            if audit is not None:
                audit.setdefault("extract_symbol_description", {})[context] = desc
            return result

    desc, head = _fallback_nearest_np(doc, tok, tok.i, chunks)
    desc = _clean_desc(desc)
    if desc and not _is_meaningless(desc):
        result.update(relation="denotes", description=desc,
                      head=head.text if head is not None else None,
                      rule="fallback_nearest", confidence="low")
        if audit is not None:
            audit.setdefault("extract_symbol_description", {})[context] = desc
    elif audit is not None:
        audit.setdefault("extract_symbol_description", {})[context] = None
    return result


def extract_symbol_description(text, symbol, model="en_core_web_sm", audit=None,
                              name_map=None):
    """
    Parse `text` with spaCy then return the full detail dict.

    Parameters
    ----------
    text : str
        Context text containing the placeholder.
    symbol : str
        Placeholder to describe.
    model : str
        spaCy model name.
    audit : dict, optional
        Flat audit dict forwarded to extract_from_doc.
    name_map : dict, optional
        Placeholder -> latex map, forwarded so the audit shows latex.

    Returns
    -------
    dict
        Full detail dict (see extract_from_doc).
    """
    nlp = _get_nlp(model)
    return extract_from_doc(nlp(text), symbol, audit=audit, name_map=name_map)


def get_description(text, symbol, model="en_core_web_sm", audit=None,
                   name_map=None):
    """
    Simple interface. Pass the text and a symbol/equation placeholder
    (e.g. "SYM77", "EQN3"); get back just the meaning as a string, or
    None if no meaning could be found.

        >>> get_meaning("... adjustable phase SYM77.", "SYM77")
        'adjustable phase'

    Parameters
    ----------
    text : str
        Context text containing the placeholder.
    symbol : str
        Placeholder to describe.
    model : str
        spaCy model name.
    audit : dict, optional
        Flat audit dict (method_name -> list of messages). Forwarded so the
        extraction step is recorded in the caller's per-equation audit trail.
    name_map : dict, optional
        Placeholder -> latex map, forwarded so the audit shows latex labels
        (e.g. "T_{max}") instead of placeholders ("SYM26").

    Returns
    -------
    str or None
        The extracted meaning, or None if nothing was found.
    """
    return extract_symbol_description(text, symbol, model, audit=audit,
                                     name_map=name_map)["description"]


# if __name__ == "__main__":
#     print(get_meaning(
#         "It produces the polarization-entangled two-photon state EQN5 "
#         "with adjustable phase SYM77.",
#         "SYM77",
#     ))
#     # -> 'adjustable phase'