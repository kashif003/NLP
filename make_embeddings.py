from embeddings import *
from utils import *
import re
from nltk import sent_tokenize
from transformers import AutoTokenizer, AutoModel
model_name = "allenai/scibert_scivocab_uncased"  # scientific domain encoder
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
model.eval()

# loading text file.
# with open("papers.txt", "r") as file:
#     text_file= file.read()
# sentences_partition=sent_tokenize(text_file)
# processing a file
# sentences = []
# KEEP_TERMS = [
#     "consists of", "composed of", "defined as", "represents",
#     "implements", "used to", "describes", "models"
# ]
# pattern = re.compile(r"\bquantum circuits?\b", flags=re.IGNORECASE)
# for sennt in sentences_partition:
#     sent= sent_tokenize(sennt)
#     sent = sent.lower()
#     if "quantum circuit" in sent:
#         # add brackets around the matched phrase
#         highlighted = pattern.sub(r"[\g<0>]", sent)
#         if any(term in highlighted for term in KEEP_TERMS):
#             # replace space inside [quantum circuit] / [quantum circuits] with underscore
#             highlighted = re.sub(
#                 r"\[(quantum circuits?)\]",
#                 lambda m: "[" + m.group(1).replace(" ", "_") + "]",
#                 highlighted
#             )

#             sent = highlighted.strip()
#             sent = f"<s> {sent} </s>"
#             sentences.append(sent)


# defining the embedding for quantum circuit.


with open("new_sentence_list.txt", "r") as file:
    text= file.read()
parts = text.split("</s>")
sentences = [p.replace("<s>", "").strip() for p in parts if p.strip()]


