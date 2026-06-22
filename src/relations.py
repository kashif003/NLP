"""
Relation classifier between equations of one paper.

Grades:
  - the two equations share a DISCRIMINATING symbol (a real, multi-character,
    non-ubiquitous variable)                            -> "strong"
  - they share only a bare single-letter variable       -> "potential"
  - they share nothing discriminating but are adjacent
    in the paper (next to each other in equation order) -> "potential"
  - otherwise                                           -> "none"

Three guards keep the grades meaningful and general across papers:
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

UBIQUITY_RATIO = 0.8  # a symbol in >= this fraction of equations is "ubiquitous"

# latex of standard operators / universal constants that are NOT paper-specific
# variables. Sharing one of these between two equations is not a real relation.
# Extend this set if you spot more in your papers.
NON_VARIABLE_LATEX = {
    r"\infty", r"\pi", r"\int", r"\iint", r"\iiint", r"\oint",
    r"\sum", r"\prod", r"\partial", r"\nabla", r"\cdot", r"\times",
    r"\pm", r"\mp", r"\approx", r"\sim", r"\propto", r"\to",
    r"\rightarrow", r"\leftarrow", r"\mapsto", r"\forall", r"\exists",
    r"\in", r"\otimes", r"\oplus", r"\langle", r"\rangle",
    r"\hbar", r"\mathrm{i}", r"\imath", r"\ldots", r"\dots", r"\cdots",
}

# a purely numeric literal: "1", "2", "0.5", "-0.5", "1.07623", "10^{-8}", ...
# (digits plus math punctuation/braces, but no real letters)
_NUMERIC_RE = re.compile(r"^[\s\d.,+\-*/^_{}()\\]*\d[\s\d.,+\-*/^_{}()\\]*$")


def _is_single_letter(latex):
    """True if the symbol's latex is a single alphabetic character (e.g. 'z')."""
    s = latex.strip()
    return len(s) == 1 and s.isalpha()


def _eq_number(eq):
    """
    Turn an equation placeholder into its printed number, for the audit only.
    'EQN1' -> '1', 'EQN12' -> '12'. If there is no 'EQN' prefix, the input is
    returned unchanged.
    """
    return eq.replace("EQN", "", 1)


def _is_meaningful_symbol(latex):
    """
    True if a symbol's latex is a paper-specific variable worth using to relate
    equations. Excludes empty strings, pure numeric literals, and standard
    operators / universal constants.

    Parameters
    ----------
    latex : str
        The symbol's latex (e.g. 'x^{i}', '1', '\\infty').

    Returns
    -------
    bool
        False for numbers and standard operators/constants, True otherwise.
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

    The threshold is max(3, ceil(UBIQUITY_RATIO * N)) so small papers are not
    over-filtered (a symbol must be in at least 3 equations to count).

    Parameters
    ----------
    eqn_order : list of str
        All equation placeholders in the paper.
    eq_to_syms : dict
        {"EQN1": ["SYM4", "SYM52"], ...}

    Returns
    -------
    set
        Symbol placeholders considered ubiquitous.
    """
    n = len(eqn_order)
    if n < 2:
        return set()
    threshold = max(3, math.ceil(UBIQUITY_RATIO * n))
    counts = {}
    for eq in eqn_order:
        for sym in set(eq_to_syms.get(eq, [])):   # set() so a symbol counts once per eq
            counts[sym] = counts.get(sym, 0) + 1
    return {sym for sym, c in counts.items() if c >= threshold}


def get_relations(target_eq, eqn_order, eq_to_syms, sym_mapping, audit=None):
    """
    Classify the relation of `target_eq` to every other equation in the paper.

    Parameters
    ----------
    target_eq : str
        The equation we are describing, e.g. "EQN3".
    eqn_order : list of str
        All equation placeholders, in their original order. Order is used for
        the adjacency rule.
    eq_to_syms : dict
        {"EQN1": ["SYM4", "SYM52"], ...} — symbols contained in each equation.
    sym_mapping : dict
        {"SYM52": "z", ...} — used to print shared symbols and to classify
        them (numeric / operator / single-letter / variable).
    audit : dict, optional
        Flat audit dict (method_name -> list of messages). Only "strong" and
        "potential" relations are logged; "none" leaves no trace.

    Returns
    -------
    dict
        {"EQN1": {"grade": ..., "description": ...}, ...} for every OTHER
        equation in the paper (including "none" pairs).
    """
    relations = {}
    ubiquitous = _ubiquitous_symbols(eqn_order, eq_to_syms)
    target_syms = set(eq_to_syms.get(target_eq, []))
    t_idx = eqn_order.index(target_eq)

    for other in eqn_order:
        if other == target_eq:
            continue

        other_syms = set(eq_to_syms.get(other, []))

        # discriminating shared symbols: drop ubiquitous ones, then drop any
        # that are not real variables (numbers / standard operators / constants)
        shared = (target_syms & other_syms) - ubiquitous
        shared = {s for s in shared
                  if _is_meaningful_symbol(sym_mapping.get(s, ""))}

        # split the remaining shared symbols into multi-char vs single-letter
        multi = [s for s in shared
                 if not _is_single_letter(sym_mapping.get(s, ""))]
        single = [s for s in shared
                  if _is_single_letter(sym_mapping.get(s, ""))]

        if multi:
            # rule 1: a discriminating multi-character variable -> strong
            grade = "strong"
            latex = [sym_mapping.get(s, s) for s in sorted(multi)]
            description = "shares symbol(s): " + ", ".join(latex)
        elif single:
            # rule 2: only bare single-letter variable(s) shared -> weak -> potential
            grade = "potential"
            latex = [sym_mapping.get(s, s) for s in sorted(single)]
            description = ("shares only single-letter symbol(s): "
                           + ", ".join(latex))
        elif abs(eqn_order.index(other) - t_idx) == 1:
            # rule 3: nothing discriminating shared but adjacent -> potential
            grade = "potential"
            description = "adjacent equation in the paper"
        else:
            # rule 4: no relation
            grade = "none"
            description = ""

        relations[_eq_number(other)] = {"grade": grade, "description": description}

        # audit only meaningful relations (strong / potential)
        if audit is not None and grade != "none":
            audit.setdefault("get_relations", []).append(
                f"{_eq_number(target_eq)} -> {_eq_number(other)}: "
                f"{grade} ({description})"
            )

    return relations