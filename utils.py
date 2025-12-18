def paper_ID_extractor(path):
    paper_list=[]
    with open(path, "r") as f:
        for line in f:
            line = line.strip()         
            if line:                     
                paper_list.append(line.split(":", 1)[1])
    return paper_list


import os
import arxiv
import time
import tarfile
import mimetypes
import json
def download_paper(paper_ID):
    """
    Downloads the paper PDF and LaTeX source from arXiv and stores them in a cache folder.
    """
    import logging
    logging.basicConfig(level=logging.INFO)
    # Ensure cache directories exist
    cache_dir = "cache"
    pdf_cache_dir = os.path.join(cache_dir, "pdf_source")
    latex_cache_dir = os.path.join(cache_dir, "latex_source")
    os.makedirs(pdf_cache_dir, exist_ok=True)
    os.makedirs(latex_cache_dir, exist_ok=True)
    if os.path.exists(os.path.join(latex_cache_dir, paper_ID)):
        logging.info("Paper already exists in cache, skipping download.")
        return
    client = arxiv.Client()
    search = arxiv.Search(id_list=[paper_ID])
    try:
        paper = next(client.results(search))
    except StopIteration:
        logging.error(f"Paper with ID {paper_ID} not found on arXiv.")
        return
    if not any(cat.startswith("quant-ph") for cat in paper.categories):
        logging.info(f"Paper {paper_ID} is not in the 'quant-ph' category. Skipping download.")
        return
    if not paper.source_url:
        logging.error(f"Paper {paper_ID} does not have a source URL. Skipping download.")
        return
    logging.info(f"Downloading paper with ID: {paper_ID}")
    try:
        pdf_path = paper.download_pdf(filename=f"{paper_ID}.pdf", dirpath=pdf_cache_dir)
        logging.info(f"PDF downloaded successfully: {pdf_path}")
    except Exception as e:
        logging.error(f"Failed to download PDF for {paper_ID}: {e}")
        return
    try:
        extract_dir_path = os.path.join(latex_cache_dir, paper_ID)
        os.makedirs(extract_dir_path, exist_ok=True)

        source_tar_path = paper.download_source(
            filename=f"{paper_ID}.tar.gz",
            dirpath=extract_dir_path
        )
        mime_type, _ = mimetypes.guess_type(source_tar_path)
        if mime_type not in ["application/gzip", "application/x-tar"]:
            logging.warning(f"Downloaded source is not a valid tar.gz file: {source_tar_path}")
            return
        if tarfile.is_tarfile(source_tar_path):
            with tarfile.open(source_tar_path, "r:*") as tar:
                tar.extractall(path=extract_dir_path)
            os.remove(source_tar_path)
            logging.info(f"LaTeX source extracted successfully for {paper_ID}.")
        else:
            logging.error(f"Source file is not a valid tar archive: {source_tar_path}")
    except Exception as e:
        logging.error(f"Failed to download or extract LaTeX source for {paper_ID}: {e}")


from image_extractor import *
def get_figure_data(paper_id):
    pdf_path = f"cache/pdf_source/{paper_id}.pdf"
    os.makedirs("cache/temp_Images", exist_ok=True)

    # Extract figures and descriptions
    extract_figures_from_pdf(pdf_path, "cache/temp_Images")

    # Load metadata
    with open(f"cache/temp_Images/{paper_id}_metadata.json", "r", encoding="utf-8") as f:
        figure_list = json.load(f)

    figure_data = []
    for figure in figure_list:
        caption = figure.get("caption", "")
        figure_number = figure.get("figure_number", None)
        page_number = figure.get("page_number", None)


        figure_data.append((caption, figure_number, page_number))

    return figure_data

def seperate_caption_discription_latex(cap_disc):
    captions= []
    discriptions= []
    start_end=[]
    for caption, disc in cap_disc.items():
        discriptions.append([dis[1] for dis in disc])
        start_end.append([dis[0] for dis in disc])
        captions.append(caption)
    return captions, discriptions, start_end

def seperate_caption_discription_pdf(cap_disc):
    captions= []
    discriptions= []
    start_end=[]
    fig_nos= []
    page_number=[]
    for fig_no, disc in cap_disc.items():
        fig_nos.append(disc["figure_label"])
        discriptions.append(disc["descriptions"])
        start_end.append(disc["start_end_positions"])
        captions.append(disc["caption"])
        page_number.append(disc["caption_page"])
    return captions, discriptions, start_end,fig_nos,page_number

from reader import *
def get_caption_discription(paper_id, pdf_source=True):
    """
    This function is used to get the caption and description of images.
    """
    latex_path = f"cache/latex_source/{paper_id}"
    pdf_path = f"cache/pdf_source/{paper_id}.pdf"


    if  pdf_source:
        cap_disc = PdfReader(pdf_path).get_caption_discription()
        captions, discriptions, start_end,fig_nos,page_number = seperate_caption_discription_pdf(cap_disc)
        return captions, discriptions, start_end,fig_nos,page_number
    elif not pdf_source:
        print("Source file exists! Processing Latex files.")
        cap_disc = LatexReader(latex_path).get_caption_description_dict()
        captions, discriptions, start_end = seperate_caption_discription_latex(cap_disc)
    else:
        print("No source found")
        return None

    return captions, discriptions, start_end




def sentenize(partition_list):
    #Training
    num_words_partition = 0
    sentences_partition = []
    for partition_tweet in partition_list:
        tweet_proc = partition_tweet.replace(" !", " .").replace(" ?", " .")
        sent_part = tweet_proc.split(" .")
        sent_part = list(filter(None, sent_part))
        for sent in sent_part:
            sent = sent.replace("%", "").replace("&", "").replace("$", "").replace("(", "").replace(")", "").replace("[", "").replace("]", "").replace("§", "").replace("=", "").replace(",", "")
            words = sent.strip().split(" ")
            final_words = []
            for word in words:
                
                if word.isalnum(): 
                    final_words.append(word) 
                    
            sentences_partition.append(final_words)
            num_words_partition += len(final_words)
    return sentences_partition, num_words_partition



import re

def clean_for_embeddings(text_input):
    def process_text(text):
        text = text.replace('\n', ' ').replace('\t', ' ')
        text = re.sub(r'(\w+)-\s+(\w+)', r'\1\2', text)
        text = re.sub(r'\\ref\{[^}]*\}|<ref>', ' <reference> ', text)
        text = re.sub(r'\\cite\{[^}]*\}', ' <citation> ', text)
        text = re.sub(r'\$.*?\$', ' ', text) # Remove inline math
        text = re.sub(r'\b(var|de|ex|sub|multi|inter|intra|pre|pro|con|infra|non)\s+(?=\w)', r'\1', text)
        text = re.sub(r'\b(single|variable|output|limit)\s+(photon|length|key|distance|noise)\b', r'\1-\2', text)
        text = re.sub(r'\b(p|q|d|n)\s+(opt|extra|lim|if|success|flip|t)\b', r'\1_\2', text)
        text = re.sub(r'[`_|{}~%&$\^§=,+\[\]\(\)]', ' ', text)
        text = re.sub(r'\b(\d+\.\d+\s*){2,}', ' ', text)
        text = re.sub(r'\b([a-z]\d?|\w/\w|√\w)\b', ' ', text)
        text = text.replace("!", ".").replace("?", ".")
        raw_sentences = text.split('.')
        seen = set()
        final_sentences = []
        for s in raw_sentences:
            words = s.strip().split()
            clean_words = [w for w in words if w.isalnum()]
            s_clean = " ".join(clean_words).lower()
            if s_clean not in seen and len(s_clean) > 5:
                final_sentences.append(" ".join(clean_words))
                seen.add(s_clean)
        result = ". ".join(final_sentences).lower()
        return re.sub(r'\s+', ' ', result).strip()

    if isinstance(text_input, list):
        return [process_text(text) for text in text_input]
    else:
        return process_text(text_input)

from embedding import *
import torch
def compare_embedding(text, anchor, model, tokenizer, threshold=0.95):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if len(text) == 0:
        return False, 0, None

    anchor_embed = anchor.unsqueeze(0).to(device)
    embeddings = encode_sentences(text, model, tokenizer)
    score = list(F.cosine_similarity(embeddings, anchor_embed, dim=1))
    print(score)
    print("*"*100)

    for  scr in score:
        if scr.item() >= threshold:
            return True, score 
    return False, score

def get_caption_index(model, tokenizer, caption, discriptions, threshold, anchor):
    caption_index = []
    for idx, (caption, disc) in enumerate(zip(caption, discriptions)):
        caption_match, caption_score = compare_embedding(caption, anchor, model, tokenizer, threshold=threshold)
        disc_match, disc_score = compare_embedding(disc, anchor, model, tokenizer, threshold=threshold)

        if caption_match or disc_match:
            caption_index.append(idx)
        else:
            print("Not found anything")
    return caption_index

import re
from difflib import SequenceMatcher
def compare_captions(c1: str, c2: str, threshold: float = 0.8):
    # normalize captions
    def _normalize(text: str) -> str:
        s = text.lower()
        s = re.sub(r"\b(?:fig|figure)\s+\d+\s*:?", "", s, flags=re.IGNORECASE)
        s = re.sub(r"[^\w\s]", " ", s)
        s = re.sub(r"\s+", " ", s).strip()
        return s
    n1 = _normalize(c1)
    n2 = _normalize(c2)
    score = SequenceMatcher(None, n1, n2).ratio()
    is_similar = score >= threshold
    return is_similar, score

def get_meta_data_from_pdf(figure_data, caption):
    """
    text: need to be selected according to embeddings.
    based on the caption/disc provided  this fucntion gets:
     1. arxiv number of pdf. done
     2. page_number where the figure if found.
     3. figure_number.
    """
    meta_data = {"fig_number": None, "page_number": None}  # Initialize with default values

    for data in figure_data:
        figure_caption = data[0]
        figure_number = data[1]
        page_number = data[2]
        figure_caption = clean_for_embeddings(figure_caption)
        sentences_1 = sent_tokenize(figure_caption)
        sentences_2 = sent_tokenize(caption)
        is_similar, score = compare_captions(sentences_1[0], sentences_2[0], threshold=0.5)
        print("Similarity score:", score)
        if is_similar:
            print("Match found! Extracting metadata.")
            meta_data["fig_number"] = figure_number
            meta_data["page_number"] = page_number
            print("Extracted figure number:", figure_number)
            print("Extracted page number:", page_number)
            break  
    if meta_data["fig_number"] is None:
        print("No matching caption found.")
    return meta_data

import shutil
def get_meta_data( paper_id, caption_index, captions, discriptions, start_end, figure_json_file_path):
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

        src_path = os.path.join("cache/temp_Images", f"{paper_id}_{meta_data['fig_number']}.png")
        dst_path = os.path.join("quantum_circuit_images", f"{paper_id}_{meta_data['fig_number']}.png")  # fig name

        if not os.path.exists(src_path):
            print(f"Source file {src_path} does not exist. Skipping.")
            continue

        shutil.copy2(src_path, dst_path)
        json_file[f"{paper_id}_{meta_data['fig_number']}.png"] = meta_data
        print("IMAGE SAVED SUCCESSFULLY", meta_data["fig_number"])
    return json_file
