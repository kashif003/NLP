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
def encode_sentences(sentences, model, tokenizer):
    """
    Encode a list of sentences using the Specter2 adapter model.
    Returns embeddings as torch.Tensor (batch_size, hidden_dim)
    """
    if isinstance(sentences, str):
        sentences = [sentences]

    inputs = tokenizer(
        sentences,
        max_length=512,
        padding=True,
        truncation=True,
        return_tensors="pt"
    )

    with torch.no_grad():
        output = model(**inputs)
    
    # Use CLS token embeddings
    embeddings = output.last_hidden_state[:, 0, :]  # (batch_size, hidden_dim)
    return embeddings


def get_anchor_embedding(sentence_list,  model, tokenizer):
    """
    Computes the mean embedding (anchor) for a list of sentences.
    """
    embeddings = encode_sentences(sentence_list, model, tokenizer)  # (batch_size, hidden_dim)
    anchor_embedding = embeddings.mean(dim=0)     # (hidden_dim,)
    return anchor_embedding

def sentenize(partition_list):
    #Training
    """
    used in exercise maybe needs an upgrade after pipeline is ready
    """
    num_words_partition = 0
    sentences_partition = []
    for partition_tweet in partition_list:
        tweet_proc = partition_tweet.replace(" !", " .").replace(" ?", " .")
        sent_part = tweet_proc.split(" .")
        sent_part = list(filter(None, sent_part))
        for sent in sent_part:
            words = sent.strip().split(" ")
            # words = [word for word in words if word not in stopwords.words('english')]
            final_words = []
            for word in words:
                
                if word.isalnum(): 
                    final_words.append(word) 
                    
            sentences_partition.append(final_words)
            num_words_partition += len(final_words)
            
    return sentences_partition, num_words_partition





"""
+ need to get better text for bettern embeddings.
+ maybe get better embedding for the quantum circuit, which can seperate the other sentence embeddings.

"""











# need to train the model from scratch or load the predefined embedding and modify them.

# check all the things i need to do to get the model ready for training.

# need to get the model train it on corpra to get the rich embeddings of quantum circuit and then match the models accordingly.