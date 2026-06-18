import spacy
import re
from cleaner import clean_text as clean_latex

nlp = spacy.load("en_core_web_sm")

COPULAS = {"is", "are", "was", "were", "be", "been"}
DET_WORDS = {"the", "this", "a", "an", "these", "those", "its", "their"}
PRONOUNS = {"we", "i", "they", "it", "he", "she", "you", "one", "this", "that"}
DEMONSTRATIVES = {"this", "that", "these", "those"}
REPORTING_VERBS = {"say", "show", "note", "prove", "demonstrate", "assume",
                   "suppose", "recall", "observe", "state", "claim"}


def strip_leading_numbering(sentence):
    """
    Remove leading numbering like "1.", "(4)", "1)" from sentence
    to prevent spaCy from picking a number as ROOT verb.

    Parameters
    ----------
    sentence : str
        Sentence possibly starting with numbering.

    Returns
    -------
    str
        Sentence with leading numbering removed.
    """
    sentence = re.sub(r'^\s*[\(\[]?\d+[\)\]\.]\s*', '', sentence)
    return sentence.strip()


def is_equation_token(token):
    """
    Check if a token is part of the [EQUATION] marker and should
    not be included in description text.

    Parameters
    ----------
    token : spacy.tokens.Token
        Token to check.

    Returns
    -------
    bool
        True if token is the equation marker.
    """
    return token.text in {"EQUATION", "[", "]"}


def is_latex_remnant(text):
    """
    Check if a token text is likely a leftover latex symbol after
    cleaning — single uppercase letter or short lowercase word.

    Parameters
    ----------
    text : str
        Token text to check.

    Returns
    -------
    bool
        True if text looks like a latex remnant.
    """
    if len(text) == 1 and text.isupper():
        return True
    if len(text) <= 3 and text.islower():
        return True
    return False


def subtree_contains_equation(token):
    """
    Check if the subtree of a token contains the [EQUATION] marker.
    Used to detect prepositions that lead directly to [EQUATION]
    and should not be included in the description.

    Parameters
    ----------
    token : spacy.tokens.Token
        Head token of the subtree to check.

    Returns
    -------
    bool
        True if subtree contains equation marker.
    """
    return any(is_equation_token(t) for t in token.subtree)


def get_subtree_text(token):
    """
    Get subtree text of a token, filtering out determiners, relative
    clauses, punctuation, lone hyphens, and equation markers. Stops
    at second prepositional phrase, equation marker, or any preposition
    whose subtree leads directly to [EQUATION] or has no meaningful
    content. Only stops at conjunction if already past first prep phrase
    to allow "symmetric and concave functions" style phrases.

    Parameters
    ----------
    token : spacy.tokens.Token
        Head token of the subtree.

    Returns
    -------
    str
        Cleaned subtree text.
    """
    tokens = []
    prep_count = 0
    for t in token.subtree:
        # stop immediately at equation marker
        if is_equation_token(t):
            break
        if t.text.lower() in DET_WORDS:
            continue
        if t.dep_ in {"relcl", "acl"}:
            continue
        if t.pos_ == "PUNCT":
            continue
        # filter lone hyphens from hyphenated words split by spaCy
        if t.text == "-":
            continue
        if t.dep_ == "prep":
            # stop if this prep leads directly to [EQUATION]
            if subtree_contains_equation(t):
                break
            # stop if prep has no meaningful pobj content
            pobj_tokens = [
                st for st in t.subtree
                if st.dep_ == "pobj" and not is_equation_token(st)
            ]
            if not pobj_tokens:
                break
            prep_count += 1
            if prep_count > 1:
                break
        # only stop at conjunction if past first prep phrase
        if t.dep_ == "cc" and prep_count > 0:
            break
        tokens.append(t.text)
    return " ".join(tokens)


def find_root_verb(doc):
    """
    Find the root verb of the sentence. Trusts spaCy ROOT label first
    regardless of POS tag to correctly handle copula verbs like "is"
    which are tagged AUX not VERB. Falls back to first VERB/AUX token
    if ROOT is a number, punctuation, or unknown token.

    Parameters
    ----------
    doc : spacy.tokens.Doc
        Parsed spaCy document.

    Returns
    -------
    spacy.tokens.Token or None
        Root verb token or None if not found.
    """
    for token in doc:
        if token.dep_ == "ROOT" and token.pos_ not in {"NUM", "PUNCT", "X"}:
            return token

    for token in doc:
        if token.pos_ in {"VERB", "AUX"}:
            return token

    return None


def handle_noun_root(doc, cleaned):
    """
    Handle sentences where spaCy picks a noun as ROOT or no root
    is found, meaning the sentence is a noun phrase label with no
    main verb. Returns the cleaned sentence text minus [EQUATION]
    and punctuation.

    Parameters
    ----------
    doc : spacy.tokens.Doc
        Parsed spaCy document.
    cleaned : str
        Cleaned sentence text.

    Returns
    -------
    str
        Noun phrase text as description.
    """
    tokens = [
        t.text for t in doc
        if not is_equation_token(t)
        and t.pos_ != "PUNCT"
        and t.text.lower() not in DET_WORDS
        and t.text != "-"
    ]
    return " ".join(tokens)


def find_as_pobj_in_subtree(root):
    """
    Search root children and their subtrees for a preposition "as"
    and return the pobj of that prep. Used as fallback when ccomp
    nsubj is a latex remnant to find the real description in patterns
    like "expressed as a convex combination".

    Parameters
    ----------
    root : spacy.tokens.Token
        Root verb token to search from.

    Returns
    -------
    spacy.tokens.Token or None
        The pobj token of the "as" prep or None if not found.
    """
    for child in root.children:
        if child.dep_ in {"prep", "conj", "advcl"}:
            for rc in child.subtree:
                if rc.dep_ == "prep" and rc.text.lower() == "as":
                    for gc in rc.children:
                        if gc.dep_ == "pobj" and not is_equation_token(gc):
                            return gc
    return None


def get_equation_description(sentence):
    """
    Extract noun phrase describing the equation from introductory sentence.
    Priority order:
    1. nsubjpass — passive subject e.g. "potential is defined as"
    2. attr if copula verb — e.g. "quantity is the spectral intensity"
    3. nsubj if not pronoun/demonstrative — e.g. "dipole moment is given by"
    4. xcomp dobj if nsubj is demonstrative — e.g. "this requires finding solution"
    5. dobj if nsubj is pronoun and not latex remnant — e.g. "we use Hamiltonian"
    6. ccomp subject for reporting verbs — e.g. "we show that X is separable"
    7. pobj of "as" prep for define pattern — e.g. "we define F as the set"
    8. conj verb dobj if root is gerund — e.g. "Defining X... we express QUBO"
    9. noun ROOT or no ROOT — sentence is a label, return full noun phrase
    10. fallback: first noun chunk before [EQUATION]

    Parameters
    ----------
    sentence : str
        Raw sentence with [EQUATION] marker.

    Returns
    -------
    str or None
        Best noun phrase description or None if not found.
    """
    cleaned = clean_latex(sentence)
    cleaned = strip_leading_numbering(cleaned)
    doc = nlp(cleaned)

    has_equation = any(t.text == "EQUATION" for t in doc)

    root = find_root_verb(doc)

    if root is None:
        return handle_noun_root(doc, cleaned)

    if root.pos_ in {"NOUN", "PROPN"}:
        return handle_noun_root(doc, cleaned)

    # priority 1: passive subject (nsubjpass)
    for child in root.children:
        if child.dep_ == "nsubjpass" and not is_equation_token(child):
            return get_subtree_text(child)

    # priority 2: predicate complement for copula (is/are)
    if root.lemma_.lower() in COPULAS:
        for child in root.children:
            if child.dep_ == "attr" and not is_equation_token(child):
                return get_subtree_text(child)

    # priority 3: active subject if not pronoun or demonstrative
    for child in root.children:
        if child.dep_ == "nsubj":
            if child.text.lower() not in PRONOUNS:
                return get_subtree_text(child)

    # priority 4: xcomp dobj when nsubj is demonstrative
    for child in root.children:
        if child.dep_ == "nsubj" and child.text.lower() in DEMONSTRATIVES:
            for sibling in root.children:
                if sibling.dep_ == "xcomp":
                    for grandchild in sibling.children:
                        if grandchild.dep_ == "dobj" and not is_equation_token(grandchild):
                            return get_subtree_text(grandchild)

    # priority 5: direct object when subject is pronoun
    for child in root.children:
        if child.dep_ == "nsubj" and child.text.lower() in PRONOUNS:
            for sibling in root.children:
                if sibling.dep_ == "dobj" and not is_equation_token(sibling):
                    if is_latex_remnant(sibling.text):
                        continue
                    return get_subtree_text(sibling)

    # priority 6: ccomp subject for reporting verbs
    if root.lemma_.lower() in REPORTING_VERBS:
        for child in root.children:
            if child.dep_ == "ccomp":
                for grandchild in child.children:
                    if grandchild.dep_ in {"nsubj", "nsubjpass"} and not is_equation_token(grandchild):
                        if is_latex_remnant(grandchild.text):
                            pobj = find_as_pobj_in_subtree(root)
                            if pobj:
                                return get_subtree_text(pobj)
                        return get_subtree_text(grandchild)

    # priority 7: pobj of "as" prep for define/express/denote pattern
    if root.lemma_.lower() in {"define", "express", "write", "denote", "introduce"}:
        for child in root.children:
            if child.dep_ == "prep" and child.text.lower() == "as":
                for grandchild in child.children:
                    if grandchild.dep_ == "pobj" and not is_equation_token(grandchild):
                        return get_subtree_text(grandchild)

    # priority 8: root is gerund (VBG) — find conj verb and take its dobj
    if root.tag_ == "VBG":
        for child in root.children:
            if child.dep_ == "conj" and child.pos_ == "VERB":
                for grandchild in child.children:
                    if grandchild.dep_ == "dobj" and not is_equation_token(grandchild):
                        return get_subtree_text(grandchild)

    # priority 9: fallback — first noun chunk before [EQUATION]
    eq_pos = cleaned.find("EQUATION") if has_equation else len(cleaned)
    for chunk in doc.noun_chunks:
        if chunk.start_char < eq_pos:
            tokens = [
                t.text for t in chunk
                if t.text.lower() not in DET_WORDS and not is_equation_token(t)
            ]
            if tokens:
                return " ".join(tokens)

    return None


def extract_meanings_from_contexts(equation_contexts):
    """
    Process output of get_equation_contexts() and extract a short
    meaning description for each equation from its context sentence.

    Parameters
    ----------
    equation_contexts : dict
        Output of get_equation_contexts(). Keys are eq_id strings,
        values are dicts with 'context' and 'referenced' keys.
        Example:
            {
                "S1.E1": {"context": "The dipole moment is given as [EQUATION]",
                          "referenced": False},
                "S1.E2": {"context": None, "referenced": False}
            }

    Returns
    -------
    dict
        Keys are eq_id strings, values are meaning strings or None.
        Example:
            {
                "S1.E1": "dipole moment",
                "S1.E2": None
            }
    """
    result = {}
    for eq_id, data in equation_contexts.items():
        context = data.get("context")
        if not context:
            result[eq_id] = None
            continue
        result[eq_id] = get_equation_description(context)
    return result


if __name__ == "__main__":
    from html_reader import HTMLReader

    paper_id = "2510.12545"
    reader = HTMLReader(paper_id)

    equation_contexts = reader.get_equation_contexts()
    equation_meanings = extract_meanings_from_contexts(equation_contexts)

    for eq_id, meaning in equation_meanings.items():
        context = equation_contexts[eq_id].get("context")
        print(f"EQ_ID:   {eq_id}")
        print(f"Context: {context}")
        print(f"Meaning: {meaning}")
        print()