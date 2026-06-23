"""
Generic symbol / equation description extractor based on dependency parsing.

Reads the dependency tree and recovers the noun phrase the placeholder is
syntactically tied to. A few structural rules cover unlimited surface
phrasings, so it generalizes across papers far better than fixed frames.

Compliant with the no-prompting rule: spaCy's parser is discriminative (it
labels structure, it does not generate text). Run locally on DC1.07.

Setup (once):
    pip install spacy[transformers]
    python -m spacy download en_core_web_trf
"""

import re

SYMBOL_RE = re.compile(r"^(?:SYM|EQN|MEQN)\d+$")
NOUN_POS = {"NOUN", "PROPN"}

DEF_VERB_LEMMAS = {
    "give", "define", "describe", "express", "write", "compute", "calculate",
    "obtain", "denote", "represent", "quantify", "yield", "capture", "read",
}
PREP_DEF = {"by", "as", "in", "from", "via", "through"}
EXPAND_PREPS = {"for", "of", "in", "with", "about", "at"}

# --- noise filters --------------------------------------------------------
PRONOUNS = {"it", "we", "they", "i", "he", "she", "you", "one", "ones"}
VAGUE_HEADS = {
    "expression", "expressions", "equation", "relation", "relations",
    "form", "forms", "formula", "result", "results", "output", "outputs",
    "quantity", "quantities", "term", "terms", "value", "values",
    "function", "functions", "case", "cases", "thing", "things",
    "approach", "approaches", "factor", "factors",
}
REF_WORDS = {
    "fig", "figs", "figure", "figures", "table", "tables", "tab",
    "eq", "eqn", "equation", "section", "sec", "panel", "panels",
    "appendix", "ref", "refs",
}

_LABEL_RE = re.compile(r"^\(?[a-zA-Z0-9]{1,3}\)\s+")
_DEMO_RE = re.compile(r"^(this|that|these|those)\s+", re.I)
_ART_RE = re.compile(r"^(the|a|an)\s+", re.I)
_ORD_RE = re.compile(r"\s-(?:th|st|nd|rd)\b", re.I)
_TAILPREP_RE = re.compile(r"\s+(in|of|to|on|for|with|by|from|as|at|into|over|between)$", re.I)
_REF_RE = re.compile(
    r"^(figures?|figs?|tables?|tabs?|eqs?|eqns?|equations?|sections?|secs?|"
    r"appendix|app|panels?)\b\.?\s*\d", re.I)
_CITE_RE = re.compile(r"^(?:auto)?cite\w+$|^[a-z]+\d{4}[a-z]?$", re.I)
_VALUE_RE = re.compile(r"^[\d.,\s×x*+\-/()]*\d[\d.,\s×x*+\-/()]*[a-zA-Z%]{0,4}$")
_ID_RE = re.compile(r"^[A-Z]{2,5}\d+[a-zA-Z]?$")

_NLP = None


def _get_nlp(model="en_core_web_trf"):
    global _NLP
    if _NLP is None:
        import spacy
        try:
            _NLP = spacy.load(model)
        except OSError:
            print(f"⚠️ Model '{model}' not found. Falling back to 'en_core_web_sm'.")
            _NLP = spacy.load("en_core_web_sm")
    return _NLP


def extract_lhs(equation_text):
    """
    Extracts the Left-Hand Side (LHS) of a mathematical expression.
    """
    if not equation_text:
        return None
    match = re.split(r'(?:=|\\approx|\\equiv|\\sim|\\propto)', equation_text)
    if match and len(match) > 1:
        return match[0].strip()
    return None


def check_duplicate_lhs(name_map):
    """
    Analyzes the name_map to find which placeholders share the identical LHS logic.
    Returns a mapping of placeholder -> list of duplicate placeholders sharing its LHS.
    """
    if not name_map:
        return {}
    lhs_to_ph = {}
    for ph, eq_text in name_map.items():
        lhs = extract_lhs(eq_text)
        if lhs:
            lhs_to_ph.setdefault(lhs, []).append(ph)
            
    ph_to_duplicates = {}
    for lhs, ph_list in lhs_to_ph.items():
        if len(ph_list) > 1:
            for ph in ph_list:
                ph_to_duplicates[ph] = [item for item in ph_list if item != ph]
    return ph_to_duplicates


def _name(ph, name_map):
    if name_map is None:
        return ph
    return name_map.get(ph, ph)


def _is_symbol(tok):
    return bool(SYMBOL_RE.match(tok.text))


def _is_person(tok):
    return tok.ent_type_ == "PERSON"


def _is_pron(tok):
    return tok.pos_ == "PRON"


def _is_verb(tok):
    return tok.pos_ == "VERB"


def _bad_anchor(tok):
    return _is_symbol(tok) or _is_person(tok) or _is_pron(tok)


def _clean_desc(desc):
    if not desc:
        return desc
    desc = desc.lstrip("\\([{ \t").strip()
    desc = _LABEL_RE.sub("", desc).strip()
    desc = _DEMO_RE.sub("", desc).strip()
    desc = _ART_RE.sub("", desc).strip()
    desc = _ORD_RE.sub("", desc).strip()
    desc = _TAILPREP_RE.sub("", desc).strip()
    return desc

def _is_meaningless(desc, chunk=None):
    if not desc:
        return True
    low = desc.lower().strip()
    if SYMBOL_RE.match(desc):
        return True
    if _REF_RE.match(low):
        return True
    if _CITE_RE.match(low):
        return True
    if _VALUE_RE.match(desc):
        return True
    if _ID_RE.match(desc):
        return True

    core = re.sub(r"^(the|a|an)\s+", "", low).strip()

    # VAGUE HEAD CHECK (Definition 1): discard ONLY if the description is
    # exactly a vague word on its own (after stripping a leading article).
    # A multi-word phrase like "wave function" can never equal a single vague
    # word, so it is automatically kept.
    if core in VAGUE_HEADS:
        return True

    if core in PRONOUNS or core in REF_WORDS or low in PRONOUNS or low in REF_WORDS:
        return True
    return False

def _find_symbol_token(doc, symbol):
    for tok in doc:
        if tok.text == symbol:
            return tok
    return None


def _find_or_alias(anchor):
    doc = anchor.doc
    for t in anchor.subtree:
        if (t.dep_ == "cc" and t.lower_ == "or"
                and t.i - 1 >= 0 and doc[t.i - 1].text == ","):
            cand = t.head
            if cand.pos_ in NOUN_POS and not _is_symbol(cand):
                return cand
    return None


def _refine_anchor(anchor):
    alias = _find_or_alias(anchor)
    if alias is not None:
        return alias
    return anchor


def _anchor(tok):
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

    # 2A) INVERTED COPULA: 'The quantity is SYM'
    if dep in {"attr", "oprd"} and head.pos_ == "VERB":
        for c in head.children:
            if c.dep_ in {"nsubj", "nsubjpass"} and c.pos_ in NOUN_POS:
                return c, "denotes", "inverted_copula"

    # 2B) EXPANDED PREPOSITIONAL ANCHORS: 'The amplitude for SYM'
    if dep == "pobj" and head.lower_ in EXPAND_PREPS:
        prep_gov = head.head
        if prep_gov.pos_ in NOUN_POS and not _is_symbol(prep_gov):
            return prep_gov, "denotes", "prepositional_governor"

    # 3) SYM has an appositive child: 'SYM, the X' / 'SYM (the X)'
    for c in tok.children:
        if c.dep_ == "appos" and c.pos_ in NOUN_POS:
            return c, "denotes", "appos_child"

    # 4) SYM attaches to a noun head: 'the X SYM' (trailing symbol)
    if dep in {"appos", "compound", "flat", "nmod", "nummod",
               "dep", "npadvmod", "conj", "amod"} and head.pos_ in NOUN_POS:
        return head, "denotes", "trailing_np"

    # 5) parser made the PROPN symbol the chunk head; grab its noun modifier
    noun_mods = [c for c in tok.children
                 if c.dep_ in {"compound", "amod", "nmod", "appos"}
                 and c.pos_ in NOUN_POS and not _is_symbol(c)]
    if noun_mods:
        return noun_mods[-1], "denotes", "head_flip"

    return None, None, None


def _short_chunk(chunk, symbol_i):
    noun_ct = sum(1 for t in chunk
                  if t.pos_ in NOUN_POS and t.i != symbol_i and not _is_symbol(t))
    return noun_ct <= 1


def _extend_of_pp(doc, chunk, desc, chunks, symbol_i):
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
    If `chunk` is the object of a preposition, look at the noun phrase BEFORE
    that preposition (its governor) and use it.
      - 'of'  -> keep both: 'speed' of 'light' -> 'speed of light'
      - for/in/with/about/at -> keep ONLY the governor:
        'parity check matrix for the quantum CSS code' -> 'parity check matrix'
    """
    start = chunk.start            # index of chunk's first token
    prev_i = start - 1             # token right before the chunk
    if prev_i < 0:
        return desc

    prev = doc[prev_i].lower_      # that token, lowercased
    if prev not in EXPAND_PREPS:   # not a preposition we handle
        return desc

    # governor = the noun chunk ending right before the preposition
    gov = ""
    for c2 in chunks:
        if c2.end == prev_i:
            gov = "".join(t.text_with_ws for t in c2
                          if t.i != symbol_i and not _is_symbol(t)).strip()
            break
    if not gov:
        return desc

    if prev == "of":
        if not _short_chunk(chunk, symbol_i):   # keep old 'of' guard
            return desc
        return (gov + " of " + desc).strip()

    # for/in/with/about/at -> governor is the real head
    return gov

def _np_and_head(doc, anchor, symbol_i, token_to_chunk, chunks):
    chunk = token_to_chunk.get(anchor.i) or token_to_chunk.get(symbol_i)
    if chunk is None:
        # Fallback to robust token tree extraction instead of rigid backtracking loop
        toks = [t for t in anchor.subtree
                if t.i <= anchor.i and t.i != symbol_i and not _is_symbol(t)]
        if not toks:
            toks = [anchor]
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
    Phase 2 fallback: choose a noun chunk to describe the symbol by SCORING
    each candidate, not just taking the nearest one.

    Scoring factors (higher = better):
      - closer to the symbol               -> -distance
      - chunk is to the LEFT of the symbol -> +2 (definitions usually precede)
      - chunk immediately adjacent         -> +4 (left) / +3 (right)
      - a defining verb (is/denotes/...) sits between symbol and a RIGHT chunk
                                           -> +5
      - chunk preceded by of/with/by/as    -> +3
      - phrase length 2..4 words           -> +2 ; very long (>6) -> -3
    """
    best = None
    best_score = float("-inf")

    for chunk in chunks:
        root = chunk.root
        # never describe the symbol with a placeholder/person/pronoun/verb chunk
        if (_is_symbol(root) or _is_person(root)
                or _is_pron(root) or _is_verb(root)):
            continue

        is_left = chunk.end <= symbol_i
        dist = symbol_i - chunk.end if is_left else chunk.start - symbol_i

        score = -dist
        if is_left:
            score += 2
        if is_left and chunk.end == symbol_i:          # immediately left
            score += 4
        if (not is_left) and chunk.start == symbol_i + 1:  # immediately right
            score += 3

        # defining verb between symbol and a right-side chunk
        if not is_left:
            between = [doc[i].lemma_.lower()
                       for i in range(symbol_i + 1, chunk.start)
                       if 0 <= i < len(doc)]
            if any(w in DEF_VERB_LEMMAS for w in between):
                score += 5

        # preposition just before a left-side chunk (e.g. "of <NP>")
        if is_left and chunk.start - 1 >= 0:
            prev = doc[chunk.start - 1].lower_
            if prev in {"of", "with", "by", "as"}:
                score += 3

        length = len([t for t in chunk if t.i != symbol_i and not _is_symbol(t)])
        if 2 <= length <= 4:
            score += 2
        elif length > 6:
            score -= 3

        if score > best_score:
            best_score = score
            best = chunk

    if best is None:
        return None, None

    desc = "".join(t.text_with_ws for t in best
                   if t.i != symbol_i and not _is_symbol(t)).strip()
    desc = _extend_of_pp(doc, best, desc, chunks, symbol_i)
    desc = _prepend_of_governor(doc, best, desc, chunks, symbol_i)
    return desc, best.root


def _latex_context(doc, name_map, target_ph=None):
    text = doc.text
    if not name_map or target_ph is None:
        return text
    # only the TARGET placeholder becomes latex, wrapped in $...$.
    # every other placeholder (SYM/EQN/MEQN) is left in its encoded form.
    target_latex = name_map.get(target_ph, target_ph)
    return text.replace(target_ph, f"${target_latex}$")


def _sentence_key(tok, head, name_map, target_ph):
    sents = [tok.sent]
    if head is not None and head.sent.start != tok.sent.start:
        sents.append(head.sent)
    sents.sort(key=lambda s: s.start)
    parts = [_latex_context(s.as_doc(), name_map, target_ph=target_ph)
             for s in sents]
    return " ".join(parts)


def extract_from_doc(doc, symbol, audit=None, name_map=None):
    result = {"symbol": symbol, "relation": None, "description": None,
              "head": None, "rule": None, "confidence": None}
    tok = _find_symbol_token(doc, symbol)
    if tok is None:
        if audit is not None:
            audit.setdefault("extract_symbol_description", {})[
                _latex_context(doc, name_map)] = None
        return result

    chunks = list(doc.noun_chunks)
    token_to_chunk = {t.i: c for c in chunks for t in c}

    anchor, relation, rule = _anchor(tok)

    if anchor is not None and _bad_anchor(anchor):
        anchor = None

    if anchor is not None:
        anchor = _refine_anchor(anchor)
        desc, head = _np_and_head(doc, anchor, tok.i, token_to_chunk, chunks)
        desc = _clean_desc(desc)
        
        chunk_obj = token_to_chunk.get(anchor.i)
        head_ok = head is None or head.pos_ in NOUN_POS
        if desc and head_ok and not _is_meaningless(desc, chunk=chunk_obj):
            result.update(relation=relation, description=desc,
                          head=head.text if head is not None else None,
                          rule=rule, confidence="high")
            if audit is not None:
                key = _sentence_key(tok, head, name_map, symbol)
                audit.setdefault("extract_symbol_description", {})[key] = (
                    f"{desc} (rule={rule}, conf=high)"
                )
            return result

    desc, head = _fallback_nearest_np(doc, tok, tok.i, chunks)
    desc = _clean_desc(desc)
    
    fallback_chunk = token_to_chunk.get(head.i) if head else None
    if desc and not _is_meaningless(desc, chunk=fallback_chunk):
        result.update(relation="denotes", description=desc,
                      head=head.text if head is not None else None,
                      rule="fallback_nearest", confidence="low")
        if audit is not None:
            key = _sentence_key(tok, head, name_map, symbol)
            audit.setdefault("extract_symbol_description", {})[key] = (
                f"{desc} (rule=fallback_nearest, conf=low)"
            )
    elif audit is not None:
        key = _sentence_key(tok, None, name_map, symbol)
        audit.setdefault("extract_symbol_description", {})[key] = None
    return result


def extract_symbol_description(text, symbol, model="en_core_web_trf", audit=None,
                              name_map=None):
    nlp = _get_nlp(model)
    return extract_from_doc(nlp(text), symbol, audit=audit, name_map=name_map)


def get_description(text, symbol, model="en_core_web_trf", audit=None,
                   name_map=None, return_conf=False):
    """
    Return the description string for `symbol`.

    return_conf : if True, return (description, confidence) where confidence is
                  "high" for a Phase-1 structural rule, "low" for the Phase-2
                  fallback, or None if nothing was found. If False (default),
                  return just the description string (unchanged behavior).
    """
    res = extract_symbol_description(text, symbol, model, audit=audit,
                                     name_map=name_map)
    if return_conf:
        return res["description"], res["confidence"]
    return res["description"]