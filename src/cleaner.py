# this  file is used to clean the text.

import re
def _replace_if_math(m: re.Match) -> str:
    """Replace match with [EQ] only if it contains a latex command or subscript/superscript."""
    if re.search(r'\\[a-zA-Z]+|[a-zA-Z0-9][_\^]\{', m.group(0)):
        return ''     # here was sym
    return m.group(0)

def clean_text_2(text: str) -> str:
    # --- fix python-mangled latex commands first ---
    text = re.sub(r'\x08(inom|f|ar|ull)', r'\\b\1', text)
    text = re.sub(r'\t(heta|imes|op|au|ext|ilde|extbf|extit)', r'\\t\1', text)
    text = re.sub(r'\r(ho|ight|angle|ightarrow)', r'\\r\1', text)
    text = re.sub(r'\n(abla|u)', r'\\n\1', text)
    text = re.sub(r'\f(rac|orall)', r'\\f\1', text)
    text = re.sub(r'\v(ee|dash)', r'\\v\1', text)
    text = re.sub(r'\a(lpha|rrow)', r'\\a\1', text)

    # --- replace full latex equation environments with [EQN] ---
    text = re.sub(r'\$\$[^$]*\$\$', '[EQN]', text)
    text = re.sub(r'\$[^$]*\$', '[EQN]', text)
    text = re.sub(r'\\\(.*?\\\)', '[EQN]', text, flags=re.DOTALL)
    text = re.sub(r'\\\[.*?\\\]', '[EQN]', text, flags=re.DOTALL)
    text = re.sub(r'\\begin\{(equation|align|math|eqnarray)\*?\}.*?\\end\{\1\*?\}', '[EQN]', text, flags=re.DOTALL)

    # --- replace continuous math blobs (latex + math chars) with [EQ] ---
    text = re.sub(r'(\\[a-zA-Z]+(\{[^}]*\})?|[a-zA-Z0-9]|[()\/\^\_\=\+\-\*\{\}\|])+', _replace_if_math, text)

    # --- replace [MENTION] with EQREF ---
    text = text.replace("[MENTION]", "EQREF")

    # --- collapse sequences ---
    text = re.sub(r'(\[EQ\][\s()/^_=,.*+|<>-]*)+', '[EQ] ', text)
    text = re.sub(r'(\[EQN\]\s*)+', '[EQN] ', text)

    # --- final cleanup ---
    text = re.sub(r'[{}|\\]', '', text)
    text = re.sub(r'(?<!\w)[_\^](?!\w)', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text




