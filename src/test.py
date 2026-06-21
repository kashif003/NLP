import re
import spacy

# transformer parser: accurate dependency parsing, no text generation
nlp = spacy.load("en_core_web_trf")


def clean_text(text, eq_label):
    # mark the target equation with a clean, parser-friendly token
    text = text.replace(eq_label, " y ")
    # turn every other placeholder (SYM/EQ/EQN + digits) into a generic noun
    text = re.sub(r'\b(EQN|EQ|SYM)\d+\b', ' x ', text)
    # collapse extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text, "y"


def find_definiendum(text, eq_label):
    text, marker = clean_text(text, eq_label)
    doc = nlp(text)

    # locate the marker token
    eq_tok = None
    for tok in doc:
        if tok.text == marker:
            eq_tok = tok
            break
    if eq_tok is None:
        return None, "no_anchor"

    # CASE A: marker inside a noun phrase ("the output y") -> phrase minus marker
    for chunk in doc.noun_chunks:
        if chunk.start <= eq_tok.i < chunk.end:
            others = [t for t in chunk if t.i != eq_tok.i and t.pos_ not in ("DET", "PUNCT")]
            if others:
                words = [t.text for t in chunk if t.i != eq_tok.i]
                return " ".join(words).strip(), "noun_phrase"
            break

    # CASE A2: marker attached to a noun (apposition "modes, y") -> that noun's phrase
    if eq_tok.head.pos_ in ("NOUN", "PROPN"):
        target = eq_tok.head
        for chunk in doc.noun_chunks:
            if chunk.start <= target.i < chunk.end:
                return chunk.text, "apposition"
        return target.text, "apposition"

    # CASE B: climb to the governing verb, take its subject
    head = eq_tok.head
    while head.pos_ != "VERB" and head.head != head:
        head = head.head
    verb = head

    subj = None
    for child in verb.children:
        if child.dep_ in ("nsubj", "nsubjpass"):
            subj = child
            break

    reason = "subject"
    if subj is not None and subj.pos_ == "PRON":
        sents = list(doc.sents)
        i = sents.index(eq_tok.sent)
        resolved = None
        if i > 0:
            for tok in sents[i - 1]:
                if tok.dep_ in ("nsubj", "nsubjpass"):
                    resolved = tok
                    break
        if resolved is not None:
            subj = resolved
            reason = "pronoun_resolved"
        else:
            reason = "pronoun_unresolved"

    if subj is None:
        return None, "no_anchor"

    for chunk in doc.noun_chunks:
        if chunk.start <= subj.i < chunk.end:
            return chunk.text, reason
    return subj.text, reason


# reason -> numeric confidence (reliability of each extraction path)
CONFIDENCE = {
    "noun_phrase":        0.90,
    "apposition":         0.85,
    "subject":            0.80,
    "pronoun_resolved":   0.60,
    "pronoun_unresolved": 0.30,
    "no_anchor":          0.00,
}


def predict(text, eq_label):
    phrase, reason = find_definiendum(text, eq_label)
    return phrase, CONFIDENCE[reason], reason




text = "The spectrum of drift momenta SYM3 of those electrons is typically expressed in terms of the photoelectron ionisation amplitude, given in atomic units as EQN1 where the phase function EQN2 is the semi-classical action of the electron with the ionisation potential SYM4 [45]."

label = "EQN1"
print( predict(text, label))