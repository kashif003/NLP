# Embeddings
from gensim.models import Word2Vec
from gensim.models import FastText
Embeddings= {
    "models": {"Word2Vec":Word2Vec,"FastText":FastText},
    "corpus": {"fastetxt":"fasttext-wiki-news-subwords-300"}
}

# check paper with quantum circuits.
paper_configs= {"batch_size":300, "file_name":"quantum_circuit_papers.json", "max_embedding_pdf":50}
Definition_pattern= [
        r"quantum circuit is defined as",
        r"quantum circuit is a",
        r"quantum circuits are",
        r"quantum circuit refers to",
        r"quantum circuit describes",
        r"quantum circuit consists of",
        r"quantum circuit comprises",
        r"quantum circuit involves",
        r"qubits in the quantum circuit",
        r"quantum gates in the circuit",
        r"the circuit implements",
        r"used in quantum computation"
    ]

#  These configs are related to model.
model_configs= {"models":["allenai/specter2_base"]}
adapter_configs= {"allenai/specter2": {"source":"hf",
    "load_as":"specter2_proximity",
    "set_active":True}}