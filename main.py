from pdf_image_extractor import *
from file_reader import *
import os
import shutil
def get_figure_data(paper_id):
    pdf_path= f"paper_pdf/{paper_id}.pdf"
    os.makedirs("temp_Images", exist_ok=True)
    extract_figures_from_pdf(pdf_path, "temp_Images")
    with open(f"temp_Images/{paper_id}_metadata.json", "r", encoding="utf-8") as f:
            figure_list = json.load(f)
    figure_data= []
    for dict in figure_list:
        figure_data.append((dict["caption"], dict["figure_number"], dict["page_number"]))
    return figure_list

def seperate_caption_discription(cap_disc):
    captions= []
    discriptions= []
    start_end=[]
    for caption, disc in cap_disc.items():
        discriptions.append([dis[1] for dis in disc])
        start_end.append([dis[0] for dis in disc])
        captions.append(caption)
    return captions, discriptions, start_end

def get_caption_discription(paper_id):
    source_path= f"paper_source/{paper_id}"
    if os.path.exists(source_path):
        print("Source file exists! Processing Latex files.") # need to edit this
    else:
        print("source file does not exist! Processing PDF.")
        process_pdf=True
    cap_disc= LatexReader(source_path).get_caption_description_dict()
    captions, discriptions, start_end=seperate_caption_discription(cap_disc)
    return captions, discriptions, start_end

def compare_embedding(text, anchor, model, tokenizer, threshold=0.95):
    if len(text)==0:
        return False, 0, None
    anchor_embed = anchor.unsqueeze(0)
    print("text:",text)
    print("*"*100)
    embeddings=encode_sentences(text,  model, tokenizer)
    score=list(F.cosine_similarity(embeddings, anchor_embed, dim=1))
    score= [[s.item()] for s in score]
    for idx in range(len(score)):
        for scr in score[idx]:
            if scr>=threshold:
                    return True, score,idx
    return False, score, None
        
def get_caption_index(caption,discriptions, threshold, anchor):
    caption_index= []
    for idx,(caption, disc) in enumerate(zip(caption,discriptions)):
        caption, score, index=compare_embedding(caption, anchor,model,tokenizer, threshold=threshold)
        discription, score, index=compare_embedding(disc, anchor,model,tokenizer, threshold=threshold)
        if caption or discription:
            print(caption,score, idx)
            caption_index.append(idx)
        else:
            print("Not found anything")
    return caption_index

def get_meta_data(figure_number, paper_id, caption_index, captions, discriptions, start_end, figure_json_file_path="temp_Images/figures_all.json"):
    with open(figure_json_file_path, "r", encoding="utf-8") as f:
        json_file = {}
        figure_list = json.load(f)

    # sorting the figure data properly
    figure_data = []
    meta_data = {}
    for dict in figure_list:
        figure_data.append((dict["caption"], dict["figure_number"], dict["page_number"]))

    for cap_idx in caption_index:
        caption = captions[cap_idx]
        meta_data = get_meta_data_from_pdf(figure_data, caption)
        if meta_data is None or meta_data["fig_number"] is None:
            print("No valid metadata found for caption. Skipping.")
            continue

        print(f"QUANTUM CIRCUIT FOUND at PAGE NUMBER {meta_data['page_number']} USING CAPTION. EXTRACTING THE META DATA.")

        # getting description and other things.
        discription = discriptions[cap_idx]
        start_and_end = start_end[cap_idx]
        if isinstance(discription, list):
            meta_data.setdefault("description", []).extend(discription)
            meta_data.setdefault("start_end", []).extend(start_and_end)
        meta_data["arxiv_id"] = paper_id

        print("saving image:", meta_data["fig_number"])
        os.makedirs("quantum_circuit_images", exist_ok=True)

        src_path = os.path.join("temp_Images/", f"{paper_id}_{meta_data['fig_number']}.png")
        dst_path = os.path.join("quantum_circuit_images", f"{paper_id}_{meta_data['fig_number']}.png")  # fig name

        if not os.path.exists(src_path):
            print(f"Source file {src_path} does not exist. Skipping.")
            continue

        shutil.copy2(src_path, dst_path)
        json_file[f"{paper_id}_{meta_data['fig_number']}.png"] = meta_data
        print("IMAGE SAVED SUCCESSFULLY", meta_data["fig_number"])

    return json_file
        


from utils import *
import torch

# getting the paper_list.
paper_list= paper_ID_extractor("paper_list_11.txt")
files=[]
for i, paper_id in enumerate(paper_list):
     paper_downloader(paper_id)
     figure_list=get_figure_data(paper_id)
     captions, discriptions, start_end=get_caption_discription(paper_id)
     model, tokenizer=load_model_and_tokenizer("allenai/specter2_base")
     anchor= torch.load("anchor_embeddings/anchor_embedding_2.pt")
     caption_index=get_caption_index(captions,discriptions, 0.95, anchor)
     continue
     
     json_file= get_meta_data("figure_number",paper_id,caption_index,captions,discriptions, start_end,figure_json_file_path=f"temp_Images/{paper_id}_metadata.json")
     if len(json_file)>0:
          files.append(json_file)
     if i ==20:
          break

print(files)




















