# avoide unnecessary printing
import warnings
import json
import os
import shutil
warnings.filterwarnings("ignore")
import logging
logging.disable(logging.CRITICAL)
# imports
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
else:
     print("Anchor embeddings do not exist! Making Anchor embedidngs") # need to add code here


def process_paper(path, anchor):
     # initializing empty json file
     json_file= {}
    # download paper
     paper_list= paper_ID_extractor(path)
     # setting a bool to process pdf or source file
     process_pdf=False
     figure_name=1
     for i,paper_id in enumerate(paper_list):
          if os.path.exists("temp_Images"):   # removiing the previous pdf images
               shutil.rmtree("temp_Images")
          if i==2:
               break
          paper_downloader(paper_id)
          os.makedirs("temp_Images", exist_ok=True)    #  making temp folder to check images.
          pdf_path= f"paper_pdf/{paper_id}.pdf"
          source_path= f"paper_source/{paper_id}"
          PdfImageExtractor("pdffigures2").extract_all_figures(pdf_path,"temp_Images")     # extracting all the images from the pdf
          if os.path.exists(source_path):
               print("Source file exists! Processing Latex files.") # need to edit this
          else:
               print("source file does not exist! Processing PDF.")
               process_pdf=True
          if process_pdf:
               print("Working on pdf.") # need to add the code here
          else:
               # Getting captions and respective discriptions.
               cap_disc= LatexReader(source_path).get_caption_description_dict()
               # loading model and tokenizer.
               model, toakenizer= load_model_and_tokenizer("allenai/specter2_base") # need to check this as well
               with open("temp_Images/figures_all.json", "r", encoding="utf-8") as f:
                    figure_list = json.load(f)
               figure_data= []
               for dict in figure_list:
                    figure_data.append((dict["caption"], dict["name"], dict["page"]))
          for i,(caption, discription) in enumerate(cap_disc.items()):
               caption_score= compare_embeddings(list(caption),model, toakenizer, anchor, threshold=0.10)
               if caption_score[0]:
                    meta_data= get_meta_data_from_pdf(figure_data, caption)
                    print(meta_data)
                    if meta_data["fig_number"] is not None:
                         print(f"QUANTUM CIRCUIT FOUND IN {paper_id} AT PAGE NUMBER {meta_data["page_number"]} USING CAPTION: {caption[:10]}. EXTRACTING THE META DATA.")
                         get_data=True
               else:
                    print("Processing the discriptions of the figures.")
                    for disc in discription:
                         score_2=compare_embeddings(disc[1],model, toakenizer, anchor, threshold=0.10)
                         if score_2[0]:
                              meta_data= get_meta_data_from_pdf(figure_data, caption)
                              if meta_data["fig_number"] is not None:
                                   print(f"QUANTUM CIRCUIT FOUND IN {paper_id} AT PAGE NUMBER {meta_data["page_number"]}. EXTRACTING THE META DATA.")
                                   get_data=True

               if get_data:
                    get_data=False
                    for disc in discription:
                         meta_data.setdefault("discription", []).append(disc[1])
                    meta_data.setdefault("start_end", []).append(discription[0])
                    meta_data["arxiv_id"] = paper_id
                    # saving the image in another folder.
                    print("saving image:", meta_data["fig_number"])
                    os.makedirs("quantum_circuit_images", exist_ok=True)
                    print("figure number", meta_data["fig_number"])
                    src_path = os.path.join("temp_Images/", f"figure_{meta_data["fig_number"]}.png")
                    dst_path = os.path.join("quantum_circuit_images", f"{figure_name}.png")

                    shutil.copy2(src_path, dst_path)
                    json_file[f"{figure_name}.png"]= meta_data
                    print("IMAGE SAVED SUCCESSFULLY",meta_data["fig_number"])
                    figure_name+=1

     return json_file

# checking the extraction of meta data.
print(len(list(process_paper(path_list, anchor).keys())))
# print(PdfReader("paper_pdf/2509.04140.pdf").preprocess_pdf()[5])
