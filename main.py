from utils import *
import torch
from file_reader import  *
path= "paper_source/2509.04140"
pdf_path= "paper_pdf/2509.04140.pdf"
caption_discription=LatexReader(path).get_caption_description_dict()

model, toakenizer= laod_model_and_tokenizer("allenai/specter2_base")
anchor= torch.load("anchor_embeddings/first_embeddings.pt")
for caption, discription in caption_discription.items():
     get_data=False
     score_1=compare_embeddings(list(caption),model, toakenizer, anchor, threshold=0.9)
     if score_1[0]:
          # get the meta data and figure
          print("Getting the meta data")
          meta_data= get_meta_data_from_pdf(pdf_path, caption)
          if meta_data["fig_number"] is not None:
               get_data=True
     else:
          for disc in discription:
               score_2=compare_embeddings(disc[1],model, toakenizer, anchor, threshold=0.9)
               if score_2[0]:
                    print("Getting the meta data")
                    meta_data= get_meta_data_from_pdf(pdf_path, caption)
                    if meta_data["fig_number"] is not None:
                         get_data=True
                    # get meta data and figure
     if get_data:
          # gettting arxiv number
          match = re.search(r'(\d{4}\.\d{5})', pdf_path)
          if match:
               paper_id = match.group(1)
          # gettting discription
          for disc in discription:
               meta_data.setdefault("discription", []).append(disc[1])
          meta_data.setdefault("start_end", []).append(discription[0])
          meta_data["arxiv_id"] = paper_id
          print(meta_data.keys())
          get_data=False
          break