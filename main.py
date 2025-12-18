from utils import *
import os
from collections import defaultdict
import torch
from transformers import AutoTokenizer
from adapters import AutoAdapterModel
import time

# 1) download the pdf/ latex file.
paper_list_path= "paper_list_11.txt"
paper_list= paper_ID_extractor(paper_list_path)
# 2) loading the model and tokeinzer.

tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
model = AutoAdapterModel.from_pretrained("allenai/scibert_scivocab_uncased")
# load anchor embeddings
anchor = torch.load("anchor_embedding_2.pt")

meta_data= defaultdict(dict)
for i,paper_id in enumerate(paper_list):
     time.sleep(3)
     # always give preference to latex
     process_pdf= True
     download_paper(paper_id)
# 3) getting (caption, fig no, page no) and extracting images from pdf.
     figure_data=get_figure_data(paper_id)
# 4) gettting (caption, discription, start_end) from the pdf/latex sources.
     if  process_pdf or not os.path.exists(f"cache/latex_source/{paper_id}"):  # add not here
          captions, discriptions, start_end,fig_nos,page_number= get_caption_discription(paper_id, pdf_source=process_pdf)
          process_pdf=True
     else:
          captions, discriptions, start_end= get_caption_discription(paper_id, pdf_source=False)
# 5) compare caption and discriptions and gettting the index of captions. 
     caption_index=get_caption_index(model,tokenizer,  captions,discriptions, 0.6, anchor)
# 6) making a json file and saving figure.
     if process_pdf:
          print("processing pdf!")
          for idx_value in caption_index:
               meta_data[f"{paper_id}_{idx_value}.png"] = {
                    f"arxiv_id": {paper_id},
                    f"fig_no": fig_nos[idx_value],
                    f"page_number": page_number[idx_value],
                    f"start_end": start_end[idx_value],
                    f"discription": clean_for_embeddings(discriptions[idx_value])
               }
               # saving the images from temp folder.
               print("saving image:", meta_data[f"{paper_id}_{idx_value}.png"]["fig_no"])
               os.makedirs("quantum_circuit_images", exist_ok=True)

               src_path = os.path.join("cache/temp_Images", f"{paper_id}_{meta_data[f"{paper_id}_{idx_value}.png"]['fig_no']}.png")
               dst_path = os.path.join("quantum_circuit_images", f"{paper_id}_{meta_data[f"{paper_id}_{idx_value}.png"]['fig_no']}.png")  # fig name
               if not os.path.exists(src_path):
                    print(f"Source file {src_path} does not exist. Skipping.")
               shutil.copy2(src_path, dst_path)
               print("IMAGE SAVED SUCCESSFULLY", meta_data[f"{paper_id}_{idx_value}.png"]["fig_no"])
     else:
          print("processing latex!")
          json_file= get_meta_data(paper_id,caption_index,captions,discriptions, start_end,figure_json_file_path=f"cache/temp_Images/{paper_id}_metadata.json")
          meta_data.update(json_file)
     if i == 4:
          break
     print(meta_data)
     if i==4:
          break

import json

with open("output.json", "w", encoding="utf-8") as f:
    json.dump(meta_data, f, ensure_ascii=False, indent=4)