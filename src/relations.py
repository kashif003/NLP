"""
Relation classifier between equations of one paper.

Grades:
  - the two equations share a DISCRIMINATING symbol (a real, multi-character,
    non-ubiquitous variable)                            -> "strong"
  - they share only a bare single-letter variable       -> "potential"
  - otherwise                                           -> "none"

Two guards keep the grades meaningful and general across papers:
  1. UBIQUITOUS symbols (present in ~every equation, e.g. 'z', 'a_{1}') carry
     no information -> removed before grading.
  2. NON-VARIABLE tokens -- pure numbers ('1', '0.5') and standard math
     operators / universal constants ('\\infty', '\\pi', '\\int', '\\partial')
     -- are not paper-specific variables, so sharing one says nothing about a
     relationship. Removed before grading. (This is what stops spurious
     'shares 1' / 'shares \\infty' strong links.)
  3. SINGLE-LETTER variables (e.g. 'm', 'i') are weak evidence (the same
     letter may be a different quantity or an index elsewhere), so a shared
     single letter alone is only 'potential', never 'strong'.
"""

import math
import re

from nltk.corpus import stopwords

UBIQUITY_RATIO = 0.8  # a symbol in >= this fraction of equations is "ubiquitous"

# latex of standard operators / universal constants that are NOT paper-specific
# variables. Sharing one of these between two equations is not a real relation.
NON_VARIABLE_LATEX = {
    r"\infty", r"\pi", r"\int", r"\iint", r"\iiint", r"\oint",
    r"\sum", r"\prod", r"\partial", r"\nabla", r"\cdot", r"\times",
    r"\pm", r"\mp", r"\approx", r"\sim", r"\propto", r"\to",
    r"\rightarrow", r"\leftarrow", r"\mapsto", r"\forall", r"\exists",
    r"\in", r"\otimes", r"\oplus", r"\langle", r"\rangle",
    r"\hbar", r"\mathrm{i}", r"\imath", r"\ldots", r"\dots", r"\cdots",
}

# a purely numeric literal: "1", "2", "0.5", "-0.5", "1.07623", "10^{-8}", ...
_NUMERIC_RE = re.compile(r"^[\s\d.,+\-*/^_{}()\\]*\d[\s\d.,+\-*/^_{}()\\]*$")

# a symbol sitting immediately to the left of an '=' (a "definition").
# starts with a letter or backslash, then letters/digits/_/^/braces/backslash.
_DEF_RE = re.compile(r"([A-Za-z\\][A-Za-z0-9_^{}\\]*)\s*=")

# English stopwords (the, is, of, ...) loaded once. Plus the lowercased remains
# of our placeholders (SYM3 -> 'sym', EQN2 -> 'eqn'), which must NOT count as
# shared content words.
_STOP = set(stopwords.words("english"))
_PLACEHOLDER_WORDS = {"sym", "eqn", "meqn", "tempeqn"}


def _content_words(text):
    """
    Turn a context string into a SET of content words: lowercased alphabetic
    words, with stopwords and placeholder remains removed.

    text : the context text around an equation.
    returns : set of meaningful words, e.g. {"wave", "function"}.
    """
    words = re.findall(r"[a-zA-Z]+", text.lower())
    return {w for w in words
            if w not in _STOP and w not in _PLACEHOLDER_WORDS}


def _is_single_letter(latex):
    """True if the symbol's latex is a single alphabetic character (e.g. 'z')."""
    s = latex.strip()
    return len(s) == 1 and s.isalpha()


def _eq_number(eq):
    """
    Turn an equation placeholder into its printed number, for the audit only.
    'EQN1' -> '1', 'EQN12' -> '12'.
    """
    return eq.replace("EQN", "", 1)


def _lhs(latex):
    """
    Return the SET of symbols an equation defines: the token left of every '='.
    'G_{t}=.. , G_{r}=.. and L_{r}=..' -> {'G_{t}', 'G_{r}', 'L_{r}'}.
    No '=' -> empty set.

    latex : the equation's latex string (eqn_mapping[eq]["latex"]).
    """
    return {m.group(1) for m in _DEF_RE.finditer(latex)}


def _norm(latex):
    """
    Normalize a latex string for comparison by removing all whitespace, so
    'M (t)' and 'M(t)' compare equal.

    latex : any latex string (a symbol's latex, or an equation's LHS).
    """
    return re.sub(r"\s+", "", latex)


def _is_meaningful_symbol(latex):
    """
    True if a symbol's latex is a paper-specific variable worth using to relate
    equations. Excludes empty strings, pure numeric literals, and standard
    operators / universal constants.
    """
    s = latex.strip()
    if not s:
        return False
    if s in NON_VARIABLE_LATEX:
        return False
    if _NUMERIC_RE.match(s):
        return False
    return True


def _ubiquitous_symbols(eqn_order, eq_to_syms):
    """
    Return the set of symbol placeholders that appear in (nearly) every
    equation of the paper. Excluded from the 'strong' decision because every
    pair trivially shares them.
    """
    n = len(eqn_order)
    if n < 2:
        return set()
    threshold = max(3, math.ceil(UBIQUITY_RATIO * n))
    counts = {}
    for eq in eqn_order:
        for sym in set(eq_to_syms.get(eq, [])):
            counts[sym] = counts.get(sym, 0) + 1
    return {sym for sym, c in counts.items() if c >= threshold}


def get_relations(target_eq, eqn_order, eq_to_syms, sym_mapping, eqn_mapping=None,
                  eq_context=None, audit=None):
    """
    Classify the relation of `target_eq` to every other equation in the paper.

    Parameters
    ----------
    target_eq : str
        The equation we are describing, e.g. "EQN3".
    eqn_order : list of str
        All equation placeholders, in their original order.
    eq_to_syms : dict
        {"EQN1": ["SYM4", "SYM52"], ...} — symbols contained in each equation.
    sym_mapping : dict
        {"SYM52": "z", ...} — used to print shared symbols and to classify them.
    audit : dict, optional
        Flat audit dict (method_name -> list of messages).

    Returns
    -------
    dict
        {"EQN1": {"grade": ..., "description": ...}, ...} for every OTHER
        equation in the paper.
    """
    relations = {}
    # convert the ubiquitous ID set to latex, so it can be subtracted from the
    # latex-based shared set below
    ubiquitous = {sym_mapping.get(s, s)
                  for s in _ubiquitous_symbols(eqn_order, eq_to_syms)}
    # target symbols as LATEX strings (e.g. "\phi"), not IDs (e.g. "SYM4"),
    # so two equations sharing the same latex via different IDs still match
    target_syms = {sym_mapping.get(s, s) for s in eq_to_syms.get(target_eq, [])}

    # content words of the target equation's context (for the 2nd-pass check)
    target_words = (_content_words(eq_context.get(target_eq, ""))
                    if eq_context is not None else set())

    # Using enumerate for an easy, efficient index lookup
    for other_idx, other in enumerate(eqn_order):
        if other == target_eq:
            continue

        other_syms = {sym_mapping.get(s, s) for s in eq_to_syms.get(other, [])}

        # discriminating shared symbols: drop ubiquitous ones and non-variables.
        # every element here is already latex, so we test it directly.
        shared = (target_syms & other_syms) - ubiquitous
        shared = {s for s in shared if _is_meaningful_symbol(s)}

        # does `other` DEFINE any shared symbol? i.e. is a shared symbol equal
        # to other's left-hand side? if so, other is a definition feeding this
        # equation -> a subset/defining relation, not just "same symbols".
        defining = []
        if eqn_mapping is not None and shared:
            defined = {_norm(d)
                       for d in _lhs(eqn_mapping.get(other, {}).get("latex", ""))}
            if defined:
                defining = [s for s in shared if _norm(s) in defined]

        # split the remaining shared symbols into multi-char vs single-letter
        multi = [s for s in shared if not _is_single_letter(s)]
        single = [s for s in shared if _is_single_letter(s)]

        if defining:
            # rule 0: other defines a shared symbol -> subset/defining -> strong
            grade = "strong"
            description = ("special case; equation " + _eq_number(other)
                           + " defines " + ", ".join(sorted(defining))
                           + " used here")
        elif multi:
            # rule 1: a discriminating multi-character variable -> strong
            grade = "strong"
            description = ("related; shares symbol(s): "
                           + ", ".join(sorted(multi)))
        elif single:
            # rule 2: only bare single-letter variable(s) shared -> potential
            grade = "potential"
            description = ("shares only single-letter symbol(s): "
                           + ", ".join(sorted(single)))
        else:
            # rule 3: no symbol relation
            grade = "none"
            description = ""

        # second pass: if still "none", compare CONTEXT. shared non-stopword
        # terms between the two equations' contexts -> upgrade to potential.
        if grade == "none" and eq_context is not None:
            other_words = _content_words(eq_context.get(other, ""))
            common = sorted(target_words & other_words)
            if common:
                grade = "potential"
                description = "shares context terms: " + ", ".join(common)

        relations[_eq_number(other)] = {"grade": grade, "description": description}

        # audit only meaningful relations (strong / potential)
        if audit is not None and grade != "none":
            audit.setdefault("get_relations", []).append(
                f"{_eq_number(target_eq)} -> {_eq_number(other)}: "
                f"{grade} ({description})"
            )

    return relations