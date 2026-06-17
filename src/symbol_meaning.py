import re
import spacy

# load model once at module level
nlp = spacy.load("en_core_web_sm")

# spaCy lemmas for defining verbs
DEFINING_VERBS = {"be", "denote", "represent", "refer", "stand", "call", "term", "define"}

_STOP_WORDS = {
    "the", "a", "an", "is", "are", "of", "in", "for", "to", "and", "or",
    "not", "with", "on", "at", "by", "from", "as", "where", "which", "that",
    "this", "these", "those", "it", "its", "all", "such", "more", "then",
    "when", "above", "below", "following", "given", "using", "shown", "thus",
    "hence", "therefore", "here", "there", "our", "we", "be", "been", "being",
}


def clean_sentence(sentence: str, symbol: str) -> str:
    """
    Clean a sentence by replacing the target symbol with SYM and
    dropping unknown math tokens.

    Parameters
    ----------
    sentence : str
        The raw context sentence.
    symbol : str
        The latex symbol to find meaning for.

    Returns
    -------
    str
        Cleaned sentence with SYM placeholder and math tokens removed.
    """
    cleaned = sentence.replace(symbol, "SYM")
    doc = nlp(cleaned)
    kept = []
    for token in doc:
        text = token.text
        if text == "SYM":
            kept.append(text)
        elif re.search(r'[\\{}_^]', text):
            continue
        elif not re.search(r'[a-zA-Z]', text):
            continue
        elif token.is_oov and not token.is_alpha:
            continue
        else:
            kept.append(text)
    return " ".join(kept)


def _apply_regex_patterns(cleaned: str, sym_idx: int, tokens: list) -> str:
    """
    Try strict regex patterns on cleaned sentence to extract symbol meaning.
    These are the most reliable patterns when they match.

    Parameters
    ----------
    cleaned : str
        Cleaned sentence with SYM placeholder.
    sym_idx : int
        Token index of SYM in tokens list.
    tokens : list
        List of tokens from cleaned sentence.

    Returns
    -------
    str
        Extracted meaning or empty string if no pattern matches.
    """
    # Pattern 1: "SYM is/denotes/represents <NP>"
    # e.g. "SYM is the learning rate"
    match = re.search(
        r'SYM\s+(?:,\s*)?(is|are|denotes?|represents?|stands\s+for|refers\s+to)\s+([\w\s]+?)(?:[,.]|$)',
        cleaned, re.IGNORECASE
    )
    if match:
        meaning = match.group(2).strip()
        meaning = re.sub(r'^(the|a|an)\s+', '', meaning, flags=re.IGNORECASE)
        # trim to 4 words max
        words = meaning.split()
        if words:
            return " ".join(words[:4])

    # Pattern 2: "SYM , the <NP>" — appositive right of SYM
    # e.g. "SYM , the Stein exponent"
    match = re.search(
        r'SYM\s*,\s*(?:the|a|an)\s+([\w\s]+?)(?:[,.]|$)',
        cleaned, re.IGNORECASE
    )
    if match:
        meaning = match.group(1).strip()
        words = meaning.split()
        if words:
            return " ".join(words[:4])

    # Pattern 3: "<NP> , SYM" or "<NP> SYM" — NP directly left of SYM (1-4 words)
    # e.g. "invariant state SYM" or "error probability constraint , SYM"
    match = re.search(
        r'((?:\w+\s+){1,4}),?\s*SYM',
        cleaned, re.IGNORECASE
    )
    if match:
        meaning = match.group(1).strip().rstrip(',')
        meaning = re.sub(r'^(the|a|an)\s+', '', meaning, flags=re.IGNORECASE)
        words = meaning.split()
        if 1 <= len(words) <= 4:
            return " ".join(words)

    # Pattern 4: "<NP> denoted by SYM" or "<NP> called SYM"
    # e.g. "the Stein exponent denoted by SYM"
    match = re.search(
        r'([\w\s]+?)\s+(?:denoted(?:\s+by)?|called|termed)\s+.*?SYM',
        cleaned, re.IGNORECASE
    )
    if match:
        rest_doc = nlp(match.group(1).strip())
        for chunk in rest_doc.noun_chunks:
            meaning = re.sub(r'^(the|a|an)\s+', '', chunk.text.strip(), flags=re.IGNORECASE)
            words = meaning.split()
            if 1 <= len(words) <= 4:
                return meaning

    # Pattern 5: "of <NP> SYM" or "of the <NP> , SYM"
    # e.g. "copies of the true state, SYM"
    match = re.search(
        r'of\s+(?:the|a|an)?\s*((?:\w+\s+){0,3}\w+)\s*,?\s*SYM',
        cleaned, re.IGNORECASE
    )
    if match:
        meaning = match.group(1).strip()
        meaning = re.sub(r'^(the|a|an)\s+', '', meaning, flags=re.IGNORECASE)
        words = meaning.split()
        if 1 <= len(words) <= 4:
            return meaning

    # Pattern 6: "SYM denote/denotes <NP>" (let X denote ...)
    # e.g. "let SYM denote the state"
    match = re.search(
        r'SYM\s+denotes?\s+(?:the|a|an)?\s*([\w\s]+?)(?:[,.]|$)',
        cleaned, re.IGNORECASE
    )
    if match:
        meaning = match.group(1).strip()
        meaning = re.sub(r'^(the|a|an)\s+', '', meaning, flags=re.IGNORECASE)
        words = meaning.split()
        if words:
            return " ".join(words[:4])

    return ""


def get_noun_phrases(cleaned: str):
    """
    Extract noun phrases from cleaned sentence using spaCy noun chunks.

    Parameters
    ----------
    cleaned : str
        Cleaned sentence with SYM placeholder.

    Returns
    -------
    tuple
        (list of noun phrase dicts with 'text', 'start', 'end', list of spaCy tokens)
    """
    doc = nlp(cleaned)
    tokens = list(doc)
    noun_phrases = []
    for chunk in doc.noun_chunks:
        if "SYM" in chunk.text:
            continue
        noun_phrases.append({
            "text": chunk.text,
            "start": chunk.start,
            "end": chunk.end
        })
    return noun_phrases, tokens


def _score_noun_phrases(noun_phrases: list, sym_idx: int, tokens: list) -> str:
    """
    Score noun phrases by proximity and context to SYM and return the best one.

    Parameters
    ----------
    noun_phrases : list
        List of noun phrase dicts with 'text', 'start', 'end'.
    sym_idx : int
        Token index of SYM.
    tokens : list
        List of tokens from cleaned sentence.

    Returns
    -------
    str
        Best matching noun phrase or empty string.
    """
    best_np = None
    best_score = float("-inf")

    for np in noun_phrases:
        score = 0
        is_left = np["end"] <= sym_idx

        dist = sym_idx - np["end"] if is_left else np["start"] - sym_idx
        score -= dist

        # prefer left side
        if is_left:
            score += 2

        # directly adjacent to SYM
        if is_left and np["end"] == sym_idx - 1:
            score += 4
        if not is_left and np["start"] == sym_idx + 1:
            score += 3

        # defining verb between SYM and NP (right side)
        if not is_left:
            between = [tokens[i].text.lower() for i in range(sym_idx + 1, np["start"]) if i < len(tokens)]
            if any(w in DEFINING_VERBS for w in between):
                score += 5

        # preposition just before NP (left side)
        if is_left and np["start"] > 0 and np["start"] - 1 < len(tokens):
            prev = tokens[np["start"] - 1].text.lower()
            if prev in {"of", "with", "by", "as"}:
                score += 3

        # length preference: 2-4 words ideal, penalize very long
        length = len(np["text"].split())
        if 2 <= length <= 4:
            score += 2
        elif length > 6:
            score -= 3

        if score > best_score:
            best_score = score
            best_np = np

    if best_np:
        meaning = re.sub(r'^(the|a|an)\s+', '', best_np["text"].strip(), flags=re.IGNORECASE)
        # trim to 4 words max
        words = meaning.split()
        if len(words) > 4:
            meaning = " ".join(words[-4:]) if best_np["end"] <= sym_idx else " ".join(words[:4])
        return meaning

    return ""


def _is_garbage_meaning(meaning: str) -> bool:
    """Return True if the extracted meaning is just stop words or too short to be useful."""
    words = meaning.lower().split()
    if not words:
        return True
    meaningful = [w for w in words if w not in _STOP_WORDS and re.search(r'[a-zA-Z]{2,}', w)]
    return len(meaningful) == 0


def _extract_by_dependency(cleaned: str) -> str:
    """
    Use spaCy dependency parse to find a defining noun phrase for SYM.
    Handles two patterns:
      - SYM is/denotes X  →  return X
      - X denotes/called SYM  →  return X
    """
    doc = nlp(cleaned)
    sym_token = next((t for t in doc if t.text == "SYM"), None)
    if not sym_token:
        return ""

    # Pattern: SYM (subj) → defining verb → object/attr is the meaning
    if sym_token.dep_ == "nsubj" and sym_token.head.lemma_ in DEFINING_VERBS:
        for child in sym_token.head.children:
            if child.dep_ in ("attr", "dobj"):
                phrase = " ".join(t.text for t in child.subtree if not t.is_punct)
                phrase = re.sub(r'^(the|a|an)\s+', '', phrase.strip(), flags=re.IGNORECASE)
                words = phrase.split()
                if 1 <= len(words) <= 5:
                    return " ".join(words[:4])

    # Pattern: X (subj) → defining verb → SYM (obj/attr) → return X
    if sym_token.dep_ in ("dobj", "pobj", "attr") and sym_token.head.lemma_ in DEFINING_VERBS:
        for child in sym_token.head.children:
            if child.dep_ == "nsubj":
                phrase = " ".join(t.text for t in child.subtree if not t.is_punct)
                phrase = re.sub(r'^(the|a|an)\s+', '', phrase.strip(), flags=re.IGNORECASE)
                words = phrase.split()
                if 1 <= len(words) <= 5:
                    return " ".join(words[:4])

    return ""


def extract_symbol_meaning(symbol: str, context: str) -> str:
    """
    Extract a short meaning for a math symbol from its context sentence.
    Pipeline: strict regex → dependency parse → spaCy noun chunks with scoring.

    Parameters
    ----------
    symbol : str
        The latex symbol string (e.g. '\\rho')
    context : str
        The sentence in which the symbol appears.

    Returns
    -------
    str
        A short meaning of the symbol, or empty string if not found.
    """
    cleaned = clean_sentence(context, symbol)
    tokens = cleaned.split()

    sym_idx = next((i for i, t in enumerate(tokens) if t == "SYM"), None)
    if sym_idx is None:
        return ""

    # Step 1: strict regex patterns — most reliable when they match
    meaning = _apply_regex_patterns(cleaned, sym_idx, tokens)
    if meaning and not _is_garbage_meaning(meaning):
        return meaning

    # Step 2: dependency parse — find defining relation to SYM
    meaning = _extract_by_dependency(cleaned)
    if meaning and not _is_garbage_meaning(meaning):
        return meaning

    # Step 3: spaCy noun chunks scored by proximity
    noun_phrases, spacy_tokens = get_noun_phrases(cleaned)
    if noun_phrases:
        meaning = _score_noun_phrases(noun_phrases, sym_idx, spacy_tokens)
        if meaning and not _is_garbage_meaning(meaning):
            return meaning

    return ""


if __name__ == "__main__":
    import json

    with open("equations_data.json", "r") as file:
        data = json.load(file)

    for eq_idx, (eq_key, symbols) in enumerate(data.items()):
        print(f"{'='*60}")
        print(f"EQUATION {eq_idx + 1}: {eq_key}")
        print(f"{'='*60}")
        for symbol_data in symbols:
            symbol, context = symbol_data[0], symbol_data[1]
            print(f"  Symbol:  {symbol}")
            print(f"  Context: {context}")
            print(f"  Cleaned: {clean_sentence(context, symbol)}")
            print(f"  Meaning: {extract_symbol_meaning(symbol, context)}")
            print()
        print()