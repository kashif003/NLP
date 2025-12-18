import nltk
nltk.download('stopwords')
import gensim.downloader as api
from reader import *
from nltk.corpus import stopwords
import numpy as np
import torch
from utils import *
import torch.nn.functional as F


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def encode_sentences(sentences, model, tokenizer, device=DEVICE):
    sentences= clean_for_embeddings(sentences)
    print("*"*100)
    print(sentences)
    if isinstance(sentences, str):
        sentences = [sentences]
    if len(sentences) == 0:
        return None
    inputs = tokenizer(
        sentences,
        max_length=512,
        padding=True,
        truncation=True,
        return_tensors="pt"
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state[:, 0, :]  # (B, 768)
    embeddings = F.normalize(embeddings, dim=1)
    return embeddings

def get_anchor_embedding(sentence_list, model, tokenizer, device=DEVICE):
    embeddings = encode_sentences(sentence_list, model, tokenizer, device)
    if embeddings is None:
        raise ValueError("Anchor sentence list is empty")
    anchor = embeddings.mean(dim=0)
    anchor = F.normalize(anchor, dim=0)
    return anchor


