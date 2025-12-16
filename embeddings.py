import nltk
nltk.download('stopwords')
from config import Embeddings
import gensim.downloader as api
from file_reader import *
from nltk.corpus import stopwords
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.utils import simple_preprocess
import numpy as np
import torch
# Getting best embeddings for the quantum circuits.
import torch
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel

import torch
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel


# --------------------------------------------------
# Device setup (IMPORTANT)
# --------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------
# Encode sentences (BATCH + GPU)
# --------------------------------------------------
def encode_sentences(sentences, model, tokenizer, device=DEVICE):
    """
    Encode a list of sentences using a transformer model.
    Returns embeddings of shape: (batch_size, hidden_dim)
    """
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

    # CLS pooling
    embeddings = outputs.last_hidden_state[:, 0, :]  # (B, 768)

    # Normalize (important for cosine similarity speed & stability)
    embeddings = F.normalize(embeddings, dim=1)

    return embeddings


# --------------------------------------------------
# Anchor embedding (compute ONCE)
# --------------------------------------------------
def get_anchor_embedding(sentence_list, model, tokenizer, device=DEVICE):
    """
    Computes mean anchor embedding from a list of sentences.
    Output shape: (hidden_dim,)
    """
    embeddings = encode_sentences(sentence_list, model, tokenizer, device)

    if embeddings is None:
        raise ValueError("Anchor sentence list is empty")

    anchor = embeddings.mean(dim=0)
    anchor = F.normalize(anchor, dim=0)

    return anchor


# --------------------------------------------------
# FAST batch comparison (GPU)
# --------------------------------------------------
def compare_embeddings_batch(
    text_list,
    model,
    tokenizer,
    anchor,
    threshold=0.80,
    device=DEVICE
):
    """
    Compare a list of texts against an anchor embedding.
    Returns:
        (match: bool, max_score: float)
    """
    if not text_list:
        return False, 0.0

    # Encode all texts in ONE forward pass
    embeddings = encode_sentences(text_list, model, tokenizer, device)

    if embeddings is None:
        return False, 0.0

    # Anchor shape: (1, hidden_dim)
    anchor = anchor.unsqueeze(0)

    # Cosine similarity (vectorized)
    scores = torch.matmul(embeddings, anchor.T).squeeze(1)
    max_score = scores.max().item()

    return max_score >= threshold, max_score


# --------------------------------------------------
# OPTIONAL: caption + description helper
# --------------------------------------------------
def match_caption_and_description(
    caption,
    descriptions,
    model,
    tokenizer,
    anchor,
    threshold=0.80,
    device=DEVICE
):
    """
    First checks caption, then descriptions.
    descriptions = list of tuples (start_end, text)
    """

    # 1️⃣ Check caption
    cap_match, cap_score = compare_embeddings_batch(
        [caption], model, tokenizer, anchor, threshold, device
    )
    if cap_match:
        return True, cap_score

    # 2️⃣ Check descriptions
    desc_texts = [d[1] for d in descriptions if len(d[1]) > 0]

    if not desc_texts:
        return False, 0.0

    return compare_embeddings_batch(
        desc_texts, model, tokenizer, anchor, threshold, device
    )


# --------------------------------------------------
# Example usage (RUN THIS ONCE)
# --------------------------------------------------
"""if __name__ == "__main__":


    # Anchor
    anchor_texts = [
        "quantum circuit",
        "quantum gate model",
        "quantum computation circuit"
    ]

    anchor = get_anchor_embedding(anchor_texts, model, tokenizer)

    # Test texts
    captions = [
        "Figure 2: Quantum circuit implementing Grover search",
        "Loss curve during training",
        "Experimental setup for optics"
    ]

    match, score = compare_embeddings_batch(
        captions,
        model,
        tokenizer,
        anchor,
        threshold=0.80
    )"""

   





"""
+ need to get better text for bettern embeddings.
+ maybe get better embedding for the quantum circuit, which can seperate the other sentence embeddings.

"""











# need to train the model from scratch or load the predefined embedding and modify them.

# check all the things i need to do to get the model ready for training.

# need to get the model train it on corpra to get the rich embeddings of quantum circuit and then match the models accordingly.

"""
from utils import *
from nltk.tokenize import sent_tokenize


text_path= "papers.txt"
with open(text_path, "r") as file:
    text= file.read()
tokens= sent_tokenize(text) 
sentences= extract_qc_sentences(tokens)
print(len(sentences))

model, tokenizer= load_model_and_tokenizer("allenai/specter2_base")
embeddings_1=get_anchor_embedding(sentences,  model, tokenizer)




trail= "In quantum information theory, a quantum circuit is a model for quantum computation, similar to classical circuits, in which a computation is a sequence of quantum gates, measurements, initializations of qubits to known values, and possibly other actions. The minimum set of actions that a circuit needs to be able to perform on the qubits to enable quantum computation is known as DiVincenzo's criteria.Circuits are written such that the horizontal axis is time, starting at the left hand side and ending at the right. Horizontal lines are qubits, doubled lines represent classical bits. The items that are connected by these lines are operations performed on the qubits, such as measurements or gates. These lines define the sequence of events, and are usually not physical cables.The graphical depiction of quantum circuit elements is described using a variant of the Penrose graphical notation.[citation needed] Richard Feynman used an early version of the quantum circuit notation in 1986."
def_tokens= sent_tokenize(trail)
def_embeddings=  get_anchor_embedding(def_tokens,  model, tokenizer)


import torch.nn.functional as F
cos_sim = F.cosine_similarity(embeddings_1, def_embeddings, dim=0)
print(cos_sim.mean())     # 0.9736

import torch

torch.save(def_embeddings, "anchor_embeddings/anchor_embedding_2.pt")


"""