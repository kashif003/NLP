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


import time
import os
import requests
def download_html(arxiv_id: str, save_dir: str = "./data/html_source") -> bool:
    """
    Download HTML version of an arxiv paper and save it locally.

    Distinguishes two failure types:
      - request ERROR (timeout, connection drop): wait 15s and retry ONCE.
        If it errors again, give up and return False.
      - paper NOT AVAILABLE (request succeeds but status != 200): no retry,
        return False immediately.

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
    time.sleep(3)
    url = f"https://arxiv.org/html/{arxiv_id}"

    response = None
    try:
        response = requests.get(url, allow_redirects=True)
    except requests.RequestException:
        # network/connection ERROR -> wait and retry exactly once
        time.sleep(15)
        try:
            response = requests.get(url, allow_redirects=True)
        except requests.RequestException:
            # still failing after retry -> give up, caller records failure
            return False

    # request went through; if the paper is simply not available, do NOT retry
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
def get_sentences_around_label(text, label, window=1, sentences=None):
    """
    Extract sentence-level context around an equation label's occurrences.
    `sentences` lets the caller pass the paper already split into sentences,
    so sent_tokenize is not re-run on the whole paper for every symbol.
    """
    if sentences is None:
        sentences = sent_tokenize(text)

    main_marker = label
    mention_marker = f"M{label}"

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
def get_meanings(clean_text, eq, audit=None, name_map=None, sentences=None):
    """
    Get the meaning of a symbol/equation, trying the main context first and
    the mention context as a fallback.

    `sentences` is the paper pre-split into sentences (built once per paper in
    main.py) so the whole-paper sent_tokenize is not repeated for every symbol.

    Returns
    -------
    str or None
        Extracted meaning, or None if nothing was found.
    """
    full_context = get_sentences_around_label(clean_text, eq, sentences=sentences)
    main_context = " ".join(full_context["main_context"])
    eq_disc = get_description(main_context, eq, audit=audit, name_map=name_map)
    if eq_disc is None:
        mention_context = " ".join(full_context["mention_context"])
        eq_disc = get_description(mention_context, eq, audit=audit, name_map=name_map)
    return eq_disc