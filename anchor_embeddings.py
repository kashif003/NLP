import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import re
from nltk.tokenize import sent_tokenize
import os
from utils import *


model_name = "allenai/scibert_scivocab_uncased"
print(f"Loading {model_name}...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)



with open("Embeddings/text_list/positive_list.txt", "r") as f:
    text = f.read()
positive_sent = sent_tokenize(text)

# hand picked sentences which define qauntum circuit
seeds = [
    "quantum circuit is linear map of qubits that is composed of sequential quantum gates.",
    "quantum circuit consists of sequence of quantum gates consisting of layers of unitaries",
    "grapased approach for circuits qdislib implements circuit cutting using grapased approach where quantum circuits are represented as directed acyclic graphs dags.",
    "formally quantum circuit consists of sequence of twubit gates and associated qubit index tuples designating the qubits which the individual gates act on.",
    "brickwork quantum circuit on qubits consists of sequential operations with unitary gates arranged across at most layers."
]

seed_vectors = get_scibert_embedding(seeds, model, tokenizer)
candidate_vectors = get_scibert_embedding(positive_sent, model, tokenizer)

# embedding of seeds.
centroid = torch.mean(seed_vectors, dim=0).unsqueeze(0) 

# Compare all candidates to this target
scores = torch.mm(candidate_vectors, centroid.transpose(0, 1)).squeeze().tolist()

# Handle edge case if single sentence
if isinstance(scores, float):
    scores = [scores]

# Sort candidates by similarity score
scored_sentences = list(zip(positive_sent, scores))
scored_sentences.sort(key=lambda x: x[1], reverse=True)

final_anchors = list(seeds) 
threshold = 0.86
for sent, score in scored_sentences:
    if score > threshold:
        if sent not in final_anchors:
            final_anchors.append(sent)
            print(f"[Keep] {score:.4f}: {sent[:60]}...")
    else:
        pass 

print(f"\nFinal Rich psoitive List contains {len(final_anchors)} sentences.")

final_tensor_list = get_scibert_embedding(final_anchors)


master_anchor = torch.mean(final_tensor_list, dim=0)

# Save the Text List
with open("rich_positive_sentences.txt", "w") as f:
    f.write("\n".join(final_anchors))

# Save the Embeddings (Rich dictionary format)
save_path = "Embeddings/positive_anchor.pt" 
os.makedirs(os.path.dirname(save_path), exist_ok=True)


torch.save({
    'centroid': master_anchor,               
    'individual_vectors': final_tensor_list, 
    'sentences': final_anchors              
}, save_path)

