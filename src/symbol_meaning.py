import re
import spacy
from transformers import pipeline

# load models once at module level
nlp = spacy.load("en_core_web_sm")
ner = pipeline("ner", model="dslim/distilbert-NER", aggregation_strategy="simple")


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
    tokens = cleaned.split()
    kept = []
    for token in tokens:
        if token == "SYM":
            kept.append(token)
        # drop math tokens
        elif re.search(r'[\\{}_^]', token):
            continue
        # drop tokens with no alphabetic characters
        elif not re.search(r'[a-zA-Z]', token):
            continue
        else:
            doc = nlp(token)
            if all(t.is_oov and not t.is_alpha for t in doc):
                continue
            kept.append(token)

    return " ".join(kept)


def get_noun_phrases(cleaned: str):
    """
    Extract noun phrases from cleaned sentence using spaCy noun chunks
    since distilbert-NER gives entities not noun phrases.
    Used as fallback alongside NER results.

    Parameters
    ----------
    cleaned : str
        Cleaned sentence with SYM placeholder.

    Returns
    -------
    list of dict
        Each dict has 'text', 'start', 'end' (token indices).
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


def extract_symbol_meaning(symbol: str, context: str) -> str:
    """
    Extract a short meaning for a math symbol from its context sentence
    using distilbert-NER for entity detection and spaCy noun chunks
    as fallback, scored by proximity to SYM.

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

    # Step 1: try NER — find entity closest to SYM
    try:
        ner_results = ner(cleaned)
        candidates = []
        for entity in ner_results:
            entity_text = entity["word"].replace("##", "")
            if "SYM" in entity_text:
                continue
            # find token index of entity
            entity_tokens = entity_text.split()
            for i, t in enumerate(tokens):
                if t == entity_tokens[0]:
                    candidates.append({
                        "text": entity_text,
                        "start": i,
                        "end": i + len(entity_tokens)
                    })
                    break

        if candidates:
            # pick closest to SYM
            best = min(candidates, key=lambda c: abs(c["start"] - sym_idx))
            meaning = re.sub(r'^(the|a|an)\s+', '', best["text"].strip(), flags=re.IGNORECASE)
            if meaning:
                return meaning
    except Exception:
        pass

    # Step 2: fallback — spaCy noun chunks scored by proximity
    noun_phrases, spacy_tokens = get_noun_phrases(cleaned)
    if not noun_phrases:
        return ""

    best_np = None
    best_score = float("-inf")

    for np in noun_phrases:
        score = 0
        is_left = np["end"] <= sym_idx

        dist = sym_idx - np["end"] if is_left else np["start"] - sym_idx
        score -= dist

        if is_left:
            score += 2

        if is_left and np["end"] == sym_idx - 1:
            score += 4
        if not is_left and np["start"] == sym_idx + 1:
            score += 3

        # defining verb between SYM and NP (right side)
        if not is_left:
            between = tokens[sym_idx + 1: np["start"]]
            if any(w.lower() in {"is", "are", "denotes", "denote", "represents", "represent"} for w in between):
                score += 5

        # preposition just before NP
        if is_left and np["start"] > 0:
            prev = tokens[np["start"] - 1].lower()
            if prev in {"of", "with", "by", "as"}:
                score += 3

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