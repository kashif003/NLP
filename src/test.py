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

_NLP = None


def _get_nlp(model="en_core_web_sm"):
    global _NLP
    if _NLP is None:
        import spacy
        _NLP = spacy.load(model)
    return _NLP


def _is_symbol(tok):
    return bool(SYMBOL_RE.match(tok.text))


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


def _extend_of_pp(doc, chunk, desc, chunks, symbol_i):
    """
    Append an immediately-following 'of'-PP, e.g. 'the PDL' -> 'the PDL of the
    loop'. Gated to short relational heads (<=1 noun in the base chunk) so we
    don't drag noisy tails onto already-descriptive NPs like
    'the maximum and minimum transmission values'.
    """
    noun_ct = sum(1 for t in chunk
                  if t.pos_ in NOUN_POS and t.i != symbol_i and not _is_symbol(t))
    if noun_ct > 1:
        return desc
    j = chunk.end
    if j < len(doc) and doc[j].lower_ == "of":
        for c2 in chunks:
            if c2.start == j + 1:
                extra = "".join(t.text_with_ws for t in c2
                                if t.i != symbol_i).strip()
                if extra:
                    return (desc + " of " + extra).strip()
            if c2.start > j + 1:
                break
    return desc


def _np_and_head(doc, anchor, symbol_i, token_to_chunk, chunks):
    """
    Resolve an anchor to (description_text, head_noun). Uses the noun chunk
    that *contains* the anchor (falling back to the chunk containing the
    symbol), so trailing-symbol NPs are captured whole. Head selection is
    attachment-agnostic: if the chunk root is the symbol/number, use the
    rightmost real noun instead.
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
        toks = [doc[i] for i in range(left.i, anchor.i + 1) if i != symbol_i]
        return "".join(t.text_with_ws for t in toks).strip(), anchor.text

    head_tok = chunk.root
    if head_tok.i == symbol_i or head_tok.like_num or _is_symbol(head_tok):
        nouns = [t for t in chunk
                 if t.pos_ in NOUN_POS and t.i != symbol_i and not _is_symbol(t)]
        if nouns:
            head_tok = nouns[-1]
    elif anchor.pos_ in NOUN_POS and not _is_symbol(anchor):
        head_tok = anchor

    desc = "".join(t.text_with_ws for t in chunk if t.i != symbol_i).strip()
    desc = _extend_of_pp(doc, chunk, desc, chunks, symbol_i)
    return desc, head_tok.text


def _fallback_nearest_np(doc, tok, symbol_i, chunks):
    """Last resort: nearest noun chunk to the left of the symbol (then right)."""
    best = None
    for chunk in chunks:
        if _is_symbol(chunk.root):
            continue
        dist = tok.i - chunk.root.i
        key = (0 if dist > 0 else 1, abs(dist))
        if best is None or key < best[0]:
            best = (key, chunk)
    if best is None:
        return None, None
    chunk = best[1]
    desc = "".join(t.text_with_ws for t in chunk if t.i != symbol_i).strip()
    return desc, chunk.root.text


def extract_from_doc(doc, symbol):
    """Core: run on an already-parsed spaCy Doc. Returns the full detail dict."""
    result = {"symbol": symbol, "relation": None, "description": None,
              "head": None, "rule": None, "confidence": None}
    tok = _find_symbol_token(doc, symbol)
    if tok is None:
        return result

    chunks = list(doc.noun_chunks)
    token_to_chunk = {t.i: c for c in chunks for t in c}

    anchor, relation, rule = _anchor(tok)
    if anchor is not None:
        anchor = _refine_anchor(anchor)
        desc, head = _np_and_head(doc, anchor, tok.i, token_to_chunk, chunks)
        if desc:
            result.update(relation=relation, description=desc, head=head,
                          rule=rule, confidence="high")
            return result

    desc, head = _fallback_nearest_np(doc, tok, tok.i, chunks)
    if desc:
        result.update(relation="denotes", description=desc, head=head,
                      rule="fallback_nearest", confidence="low")
    return result


def extract_symbol_description(text, symbol, model="en_core_web_sm"):
    """Parse `text` with spaCy then return the full detail dict."""
    nlp = _get_nlp(model)
    return extract_from_doc(nlp(text), symbol)


def get_meaning(text, symbol, model="en_core_web_sm"):
    """
    Simple interface. Pass the text and a symbol/equation placeholder
    (e.g. "SYM77", "EQN3"); get back just the meaning as a string, or
    None if no meaning could be found.

        >>> get_meaning("... adjustable phase SYM77.", "SYM77")
        'adjustable phase'
    """
    return extract_symbol_description(text, symbol, model)["description"]






print(get_meaning("It produces the polarization-entangled two-photon state EQN5 with adjustable phase SYM77.", "SYM77"))
# -> 'adjustable phase'


# text = "he photon pair source, described in detail in [44], is based on cavity-enhanced spontaneous parametric down-conversion (SPDC) in an interferometric configuration. It produces the polarization-entangled two-photon state EQN5  with adjustable phase SYM77 ."


# r = extract_symbol_description(text, "EQN5")
# print(r)