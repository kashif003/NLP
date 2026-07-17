from utils import *
import os
from collections import defaultdict
import torch
from transformers import AutoTokenizer
from adapters import AutoAdapterModel
import json
from transformers import pipeline
import torch
import pandas as pd

# 1) download the pdf/ latex file.
paper_list_path= "paper_list_11.txt"
paper_list= paper_ID_extractor(paper_list_path)

# 2) loading the model and tokeinzer.
tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
model = AutoAdapterModel.from_pretrained("allenai/scibert_scivocab_uncased")
# laoding anchor embeddings (if we dont have enchor embeddings then run anchor_embeddings.py)
anchor = torch.load("Embeddings/positive_anchor.pt")
csv_file= pd.DataFrame(columns=["paper_id", "No_Img"])
csv_rows = []
processed_ids = set()
meta_data= defaultdict(dict)
# main loop
for i,paper_id in enumerate(paper_list):
     # always give preference to latex
     process_pdf= False
     download_paper(paper_id)
     # small check need to remove this
     path_a = f"cache/latex_source/{paper_id}"
     path_b  = f"cache/pdf_source/{paper_id}.pdf"
     if not os.path.isdir(path_a) or not os.path.isfile(path_b):
          continue
# 3) getting (caption, fig no, page no) and extracting images from pdf.
     figure_data=get_figure_data(paper_id)

# 4) gettting (caption, discription, start_end) from the pdf/latex sources.
     if  process_pdf or  not os.path.exists(f"cache/latex_source/{paper_id}"):  
          captions, discriptions, start_end,fig_nos,page_number= get_caption_discription(paper_id, pdf_source=True)
          process_pdf=True
     else:
          captions, discriptions, start_end= get_caption_discription(paper_id, pdf_source=False)

# 5) compare caption and discriptions and gettting the index of quantum circuit captions.
     caption_index=get_caption_index(
                                   model=model,
                                   tokenizer=tokenizer,
                                   captions=captions,
                                   descriptions=discriptions,
                                   anchor=anchor,
                                   threshold=0.94,   # cosine similarity should be 0.94
                                   )
# 6) making a json file and saving figure.
     if process_pdf:
          print("processing pdf!")
          for idx_value in caption_index:
               # Define the source and destination paths for the image
               src_path = os.path.join("cache/temp_Images", f"{paper_id}_{fig_nos[idx_value]}.png")
               dst_path = os.path.join("quantum_circuit_images", f"{paper_id}_{fig_nos[idx_value]}.png")
               # Check if the source image exists before proceeding
               if not os.path.exists(src_path):
                    print(f"UNABLE TO LOCATE IMAGE AT: {src_path}")
                    continue

               # Attempt to move the image and only add metadata if successful
               try:
                    shutil.copy2(src_path, dst_path)
                    print("IMAGE SAVED SUCCESSFULLY", fig_nos[idx_value])

                    # Add metadata entry only if the image was successfully moved
                    meta_data[f"{paper_id}_{idx_value}.png"] = {
                         "arxiv_id": paper_id,
                         "fig_no": fig_nos[idx_value],
                         "page_number": page_number[idx_value],
                         "start_end": list(start_end[idx_value]) if isinstance(start_end[idx_value], set) else start_end[idx_value],
                         "discription": clean_for_embeddings(discriptions[idx_value])
                    }
               except Exception as e:
                    print(f"Failed to save image {src_path} to {dst_path}: {e}")
     else:
          print("processing latex!")
          json_file= get_meta_data(paper_id,caption_index,captions,discriptions, start_end,figure_json_file_path=f"cache/temp_Images/{paper_id}_metadata.json")
          meta_data.update(json_file)
          
     print("NUMBER OF PAPERS CHECKED:",i+1)
     img_count = sum(1 if paper_id in str(key) else 0 for key in meta_data)
     csv_rows.append({
        "paper_id": paper_id, 
        "No_Img": img_count
    })
     processed_ids.add(paper_id)
     if len(meta_data)==250:
          break

# gettting gate names and algorithm info from the description.
device = 0 if torch.cuda.is_available() else -1
qa_pipeline = pipeline(
    "question-answering", 
    model="deepset/roberta-base-squad2", 
    device=device
)
json_data = update_json_with_gates_algos(meta_data)
for paper_id in paper_list:
    if paper_id not in processed_ids:
        csv_rows.append({
            "paper_id": paper_id, 
            "No_Img": None  # This will appear as an empty cell (NaN) in the CSV
        })

# making a csv file.
csv_file = pd.DataFrame(csv_rows)
csv_file.to_csv("images_per_paper.csv", index=False)


with open("output_enriched.json", "w") as outfile:
    json.dump(json_data, outfile, indent=4)
    print("Saved updated data to 'output_enriched.json'")
