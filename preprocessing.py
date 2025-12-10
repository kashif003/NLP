import nltk
import scipy
import re
from nltk.tokenize import sent_tokenize
import unicodedata


class Preprocess():
    def __init__(self, text:list):
        # putting the text in lower case
        self.raw_text = text
        self.sentences_tokens= []
        self.processed_text= []

    def process_text(self):
        if self.processed_text:
            return
        self.processed_text= self.preprocess_text(self.raw_text)
    
    def preprocess_text(self, text:list):
        processed_text= []
        for i,page in enumerate(text, 1):
            text= page.lower()
            # normalizing the data.
            text = unicodedata.normalize("NFKD", text)
            # replaceing \n and double spaces
            text = re.sub(r"(-)?\n\d+\n+", r"\1 ", text)
            # Replace remaining newlines with space
            text = text.replace("\n", " ")
            # removing refrences
            text = re.sub(r"\[[0-9,\s\-–]+\]", "", text)
            # remove the dots .................
            text = re.sub(r"(?:\.\s+)+\.", " ", text)
            # removing the text like "- "
            text = re.sub(r"(- |[A-Za-z0-9][-\u2013][A-Za-z0-9]| –)", "", text)
            # mark the quotes
            text = re.sub(r'[“"]([^“"]+)[”"]', self.mark_quotes, text)    # double-quoted
            text = re.sub(r"[‘']([^‘']+)[’']", self.mark_quotes, text)    # single-quoted
            # remove the single apostrophes
            text = re.sub(r"(?<=\w)['’]+(?=\w)", "", text)
            # remove the double apostrophes
            text = re.sub(r"[\"'“”‘’]", "", text)
            # Replace multiple spaces with single space
            text = re.sub(r" {2,}", " ", text)
            if i==1:
                # remove the detail about the paper
                pattern_paper_detail = r"arxiv:.*?\b\d{1,2}\s+[A-Za-z]{3,9}\s+\d{2,4}\b[ ,]*"
                text= re.sub(pattern_paper_detail, "", text)
                m = re.search(r"abstract", text, flags=re.IGNORECASE)
                if m:
                    text = text[m.start():]
            processed_text.append(text)
        return processed_text

    def tokenize(self):
       self.sentences_tokens =sent_tokenize(self.text)
       return self.sentences_tokens
    
    def mark_quotes(self,m):
        inner = m.group(1)
        return f"<QUOTE> {inner} </QUOTE>"

    def _locate_figure(self):
        self.process_text() 

        pattern_figure = r'\bfigure[.:]?\s*\d+[a-zA-Z]?'
        pattern_fig    = r'\bfig[.:]?\s*\d+[a-zA-Z]?'

        matches = []

        for page_idx, page_text in enumerate(self.processed_text,1): 
            for m in re.finditer(pattern_figure, page_text, flags=re.IGNORECASE):
                prev_text = self.processed_text[page_idx - 2] if page_idx > 1 else ""
                next_text = self.processed_text[page_idx] if page_idx < len(self.processed_text) else ""
                combined_pages = prev_text + page_text + next_text
                matches.append({
                    "type": "figure",
                    "page": page_idx,         
                    "match": m.group(0),
                    "start": m.start(),
                    "end": m.end(),
                    "span": m.span(),
                    "page_text":combined_pages
                })

            for m in re.finditer(pattern_fig, page_text, flags=re.IGNORECASE):
                prev_text = self.processed_text[page_idx - 2] if page_idx > 1 else ""
                next_text = self.processed_text[page_idx] if page_idx < len(self.processed_text) else ""
                combined_pages = prev_text + page_text + next_text
                matches.append({
                    "type": "fig",
                    "page": page_idx,
                    "match": m.group(0),
                    "start": m.start(),
                    "end": m.end(),
                    "span": m.span(),
                    "page_text":combined_pages
                })

        return matches if len(matches) > 0 else None


# need to remove some extra steps like preprocess only the intrested pages.
# combine the text which is preprocessed.
# 

        
"""
preprocessing:
(done)
+ remove the narXiv:2509.04140v1  [quant-ph]  4 Sep 2025 like these things (it is only on the first pages of the papers)
(done)
+ need to remove the page numbers from  the text as well \n2\n.
(done)
+ remove the apstophirs shor'S like these things as ell for the text.
(done)
+ remove the refrences [32–34]

(not done yet: maybe need to remove the sentence before the location of the figure.)
+ *impt*: need to remove the txt from figures: nApplication link\nApplication Layer\nCryptographic\napplication\nQKD Network Controller\nQKD Network Manager\nManagement interface\nControl and\nManagement Layer\nApplication\n\xa0interface\nQKD Node\nQKD Link\nKey Manager\nKM Interface\nKey Management\nLayer\nQuantum Layer\nController\xa0\ninterface\nKM Link\n
(done)
+ remove the abstract from the paper and initial names.

+  “replace |+⟩|+⟩|+⟩|+⟩with −|+⟩|+⟩|+⟩|+⟩,”  remove the double inverted commas
# further preprocessing at later stages.

+ 
"""

"""
TO DO:

+ get figure, Fig all kinds differnt typed of names for figures.
+ need to check which mention of fig is the caption and which mention is the discription.
+ 



"""