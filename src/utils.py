def paper_ID_extractor(path, n= None):
    """Extracts paper IDs from a file by parsing the substring after the first colon on each line."""
    paper_list=[]
    with open(path, "r") as f:
        for line in f:
            line = line.strip()         
            if line:                     
                paper_list.append(line.split(":", 1)[1])
        if n:
            return paper_list[:n]
        else:
            return paper_list


import tqdm
import time
import os
import requests

from utils import paper_ID_extractor

def download_html(arxiv_id: str, save_dir: str = "./data/html_source") -> bool:
    """
    Download HTML version of an arxiv paper and save it locally.

    Parameters
    ----------
    arxiv_id : str
        The arxiv paper ID.
    save_dir : str
        Directory to save the HTML files.

    Returns
    -------
    bool
        True if download was successful, False otherwise.
    """
    url = f"https://arxiv.org/html/{arxiv_id}"
    response = requests.get(url, allow_redirects=True)

    if response.status_code != 200:
        return False

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{arxiv_id}.html")
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(response.text)

    return True


import nltk
from nltk.tokenize import sent_tokenize
import re
# Run once to download the required tokenizer data (local, no prompting)
nltk.download("punkt")
nltk.download("punkt_tab")  # needed for newer NLTK versions (>=3.8.2)

def get_sentences_around_label(text, label, window=1):
    """
    Extract sentence-level context around an equation label's occurrences in text.

    Splits the text into sentences and searches for two marker forms derived
    from the label: the main marker (the equation's own occurrence, e.g.
    "EQN1") and the mention marker (references to that equation, e.g. "MEQN1").
    For each matching sentence, a context window of surrounding sentences is
    collected.

    Args:
        text (str): The input text to search through.
        label (str): The wrapped equation label, e.g. "<EQN1>". The first and
            last characters are stripped to obtain the inner label, which is
            used directly as the main marker ("EQN1") and prefixed with "M"
            to form the mention marker ("MEQN1").
        window (int, optional): Number of sentences to include before and
            after each matching sentence. Defaults to 1.

    Returns:
        dict: A dictionary with three keys:
            - "main_context" (list[str]): Context windows for each occurrence
              of the main marker. Empty if none found.
            - "mention_context" (list[str]): Context windows for each occurrence
              of the mention marker. Empty if none found.
            - "window" (int): The window size used.
    """
    sentences = sent_tokenize(text)
    main_marker = label
    mention_marker = f"M{label}"

    # word-boundary matching: bracket-free markers would otherwise collide,
    # e.g. "EQN1" matching "EQN12" or matching inside the mention "MEQN1".
    main_re = re.compile(r"\b" + re.escape(main_marker) + r"\b")
    mention_re = re.compile(r"\b" + re.escape(mention_marker) + r"\b")

    result = {"main_context": [], "mention_context": [], "window": window}
    for i, sent in enumerate(sentences):
        start = max(0, i - window)
        end = min(len(sentences), i + window + 1)
        context = " ".join(sentences[start:end])

        if main_re.search(sent):
            result["main_context"].append(context)
        if mention_re.search(sent):
            result["mention_context"].append(context)
    return result





def strip_backslash(s):
    """
    Remove every backslash from a string, for use as a clean JSON key.

    s        : a latex string, e.g. "\\mathcal{L}" or "T_{max}"
    returns  : the same string with all backslashes removed,
               e.g. "mathcal{L}", "T_{max}" (unchanged if it had none)
    """
    return s.replace("\\", "")

from extract_description import get_description
def get_meanings(clean_text, eq, audit=None, name_map=None):
    """
    Get the meaning of a symbol/equation, trying the main context first and
    the mention context as a fallback.

    Parameters
    ----------
    clean_text : str
        Full paper text with placeholders.
    eq : str
        The placeholder to describe, e.g. "SYM3".
    audit : dict, optional
        Flat audit dict (method_name -> list of messages). Forwarded to
        get_description so the meaning-extraction steps are recorded.
    name_map : dict, optional
        Placeholder -> latex map, forwarded so the audit shows latex.

    Returns
    -------
    str or None
        Extracted meaning, or None if nothing was found.
    """
    full_context = get_sentences_around_label(clean_text, eq)
    main_context = " ".join(full_context["main_context"])
    eq_disc = get_description(main_context, eq, audit=audit, name_map=name_map)
    if eq_disc is None:
        mention_context = " ".join(full_context["mention_context"])
        eq_disc = get_description(mention_context, eq, audit=audit, name_map=name_map)
    return eq_disc