import nltk
import scipy
import re
from nltk.tokenize import sent_tokenize
import unicodedata


class Pdf_Preprocess():
    def __init__(self, text:list):
        # putting the text in lower case
        self.raw_text = text
        self.sentences_tokens= []
        self.processed_text= None

    def get_processed_text(self):
        self.process_text()
        return self.processed_text
    def process_text(self):
        if self.processed_text:
            return
        self.processed_text= self.preprocess_text(self.raw_text)
        return self.processed_text
    
    def preprocess_text(self, text:list):
        processed_text= []
        for page_number,page in text:
            text = page.lower()
            # highlighting the figure.
            text= re.sub(r'\n\s*(fig(?:ure)?)[.]?\b', '<figure>',text,flags=re.IGNORECASE)
            # Normalizing the data
            text = unicodedata.normalize("NFKD", text)
            # Replace hyphenated line breaks like "-\n12\n"
            text = re.sub(r"(-)?\n\d+\n+", r"\1 ", text)
            # Replace remaining newlines with space
            text = text.replace("\n", " ")
            # Replace | and ` with space
            text = text.replace("|", " ").replace("`", " ")
            # Replace colon-underscore and plain underscore with space
            text = text.replace(":_", " ").replace("_", " ")
            # Remove dots from common honorifics (keep the words)
            for pat in [" mrs. ", " ms. ", " mr. ", " dr. ", " prof. ", " dr.-ing. "]:
                text = text.replace(pat, pat.replace(".", ""))
            # Removing references like [1], [2–5], [3, 4]
            text = re.sub(r"\[[0-9,\s\-–]+\]", "", text)
            # Remove dotted ellipsis-like runs "...."
            text = re.sub(r"(?:\.\s+)+\.", " ", text)
            # Removing text like "- " or A–B patterns
            text = re.sub(r"([A-Za-z0-9])[-\u2013]([A-Za-z0-9])", r"\1\2", text)
            # Mark the quotes (depending on self.mark_quotes)
            text = re.sub(r'[“"]([^“"]+)[”"]', self.mark_quotes, text)  # double-quoted
            text = re.sub(r"[‘']([^‘']+)[’']", self.mark_quotes, text)  # single-quoted
            # Remove single apostrophes inside words
            text = re.sub(r"(?<=\w)['’]+(?=\w)", "", text)
            # Remove remaining quote-like characters
            text = re.sub(r"[\"'“”‘’]", "", text)
            # Replace multiple spaces with single space
            text = re.sub(r" {2,}", " ", text)
            # Replace links with placeholder
            text = re.sub(r"http\S+", " hrefl ", text)
            # replace the emails as well.
            text = re.sub(r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}', ' email ', text)
            if page_number==1:
                # remove the detail about the paper
                pattern_paper_detail = r"arxiv:.*?\b\d{1,2}\s+[A-Za-z]{3,9}\s+\d{2,4}\b[ ,]*"
                text= re.sub(pattern_paper_detail, "", text)
                m = re.search(r"abstract", text, flags=re.IGNORECASE)
                if m:
                    text = text[m.start():]
            processed_text.append((page_number,text))
        return processed_text

    def tokenize(self):
       self.sentences_tokens =sent_tokenize(self.text)
       return self.sentences_tokens
    
    def mark_quotes(self,m):
        inner = m.group(1)
        return f"<QUOTE> {inner} </QUOTE>"

    def clean_captions(self):
        
        return self.raw_text



# need to remove some extra steps like preprocess only the intrested pages.
# combine the text which is preprocessed.
# 

        
"""
preprocessing:

(not done yet: maybe need to remove the sentence before the location of the figure.)

# further preprocessing at later stages.

"""

"""
TO DO:
+ filter the pages which contain the caption of figure.
    + if contain then get the figure, caption, fig number apage_number
"""