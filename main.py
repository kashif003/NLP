from utils import *
from embeddings import *
from file_reader import *
import os
from pdf_image_extractor import *
path_list="paper_list_11.txt"

embedding_path="anchor_embeddings/first_embeddings.pt"
if os.path.exists(embedding_path):
     print("laoding anchor embeddings!")
     anchor= torch.load(embedding_path)


model,tokenizer= load_model_and_tokenizer("allenai/specter2_base")


print(len(list(process_paper(path_list, anchor,model, tokenizer, threshold=0.95).keys())))