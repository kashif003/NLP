# downloading the text for quantum circuit.
from utils import *
from pathlib import Path
from file_reader import *
root = Path("paper_source")
paper_source_dirs = [p for p in root.iterdir() if p.is_dir()]

KEEP_TERMS = [
    "consists of", "composed of", "defined as", "represents",
    "implements", "used to", "describes", "models"
]



sentences = []
pattern = re.compile(r"\bquantum circuits?\b", flags=re.IGNORECASE)
for path in paper_source_dirs:
    try:
        reader = LatexReader(path)
        content = reader.process_contents()
    except Exception as e:
        print(f"Skipping {path} due to error: {e}")
        continue
    for page in content:
        sentence= sent_tokenize(page)
        for sent in sentence:
            sent_low = sent.lower()
            if "quantum circuit" in sent_low:
                highlighted = pattern.sub(r"[\g<0>]", sent_low)
                if any(term in highlighted for term in KEEP_TERMS):
                    highlighted = re.sub(
                        r"\[(quantum circuits?)\]",
                        lambda m: "[" + m.group(1).replace(" ", "_") + "]",
                        highlighted
                    )
                    sent = f"<s> {highlighted.strip()} </s>"
                    print(sent)
                    sentences.append(sent)
        
    


with open("new_sentence_list.txt", "w", encoding="utf-8") as file:
    for s in sentences:
        file.write(s.strip() + "\n")
