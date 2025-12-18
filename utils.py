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

import torch
import torch.nn.functional as F

def compare_embedding(text_list, anchor, model, tokenizer, threshold=0.9, use_boost=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not text_list:
        return False, []
    if isinstance(text_list, str):
        text_list = [text_list]
    centroid = anchor['centroid'].to(device)
    individual_vectors = anchor['individual_vectors'].to(device)
    embeddings = get_scibert_embedding(text_list, model, tokenizer).to(device)
    score_centroid = F.cosine_similarity(embeddings, centroid, dim=1)
    all_individual_scores = torch.mm(embeddings, individual_vectors.transpose(0, 1))
    score_individual_max, _ = torch.max(all_individual_scores, dim=1)
    final_scores = torch.max(score_centroid, score_individual_max)
    score_list = final_scores.tolist()
    
    # this list is extracted from the extract_top_keywords(positive_file, top_n=50) in utily.py
    boost_keywords = boost_keywords = [("quantum", 0.05), ("circuit", 0.05), ("gates", 0.05), ("quantum circuit", 0.1), ("qubits", 0.05), ("consists", 0.05), ("composed", 0.05), ("circuits", 0.05), ("quantum gates", 0.1), ("quantum circuits", 0.1), ("layers", 0.05), ("operations", 0.05), ("unitary", 0.15), ("circuit composed", 0.1), ("sequence", 0.05), ("quantum circuit composed", 0.1), ("twubit", 0.05), ("label", 0.05), ("represented", 0.05), ("gate", 0.05), ("quantum circuit consists", 0.1), ("twubit gates", 0.15), ("applied", 0.05), ("circuit consists", 0.1), ("depth", 0.05), ("number", 0.05), ("brickwork", 0.05), ("consists sequence", 0.15), ("begin", 0.05), ("section", 0.05), ("implement", 0.05), ("sec", 0.05), ("represents", 0.05), ("label sec", 0.15), ("gates consisting", 0.15), ("gates applied", 0.15), ("implemented", 0.05), ("composed twubit gates", 0.15), ("consisting", 0.05), ("cutting", 0.05), ("elementary", 0.05), ("circuit consists sequence", 0.1), ("circuit qubits", 0.1), ("clifford gates", 0.15), ("applied qubits", 0.15), ("approach", 0.05), ("clifford", 0.05), ("cnot", 0.05), ("composed twubit", 0.15), ("theorem", 0.05)]

    boost_keywords.sort(key=lambda x: x[1], reverse=True)
    final_results = []
    has_positive_match = False
    for i, txt in enumerate(text_list):
        current_score = score_list[i]
        if use_boost:
            lower_text = txt.lower()
            for keyword, boost_amt in boost_keywords:
                if keyword in lower_text:
                    current_score += boost_amt
                    if current_score > 1.0: current_score = 1.0
                    break # Stop after finding the highest value keyword
        
        final_results.append(current_score)
        if current_score >= threshold:
            has_positive_match = True
    return has_positive_match, final_results


def get_caption_index(model, tokenizer, captions, descriptions, threshold, anchor):
    caption_index = []
    print(f"Scanning {len(captions)} images...")
    for idx, (caption, disc) in enumerate(zip(captions, descriptions)):
        caption_match, caption_score = compare_embedding(
            caption, anchor, model, tokenizer, 
            threshold=threshold, 
            use_boost=True 
        )
        disc_match, disc_score = compare_embedding(
            disc, anchor, model, tokenizer, 
            threshold=threshold, 
            use_boost=False 
        )
        if caption_match or disc_match:
            caption_index.append(idx)
        else:
            pass
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


def get_scibert_embedding(text_list,model,  tokenizer):
    """
    Generates normalized sentence embeddings using SciBERT mean pooling.
    """
    text_list= clean_for_embeddings(text_list)
    if not text_list:
        return torch.tensor([])
    inputs = tokenizer(text_list, padding=True, truncation=True, return_tensors="pt", max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    token_embeddings = outputs.last_hidden_state
    attention_mask = inputs['attention_mask']
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    mean_embeddings = sum_embeddings / sum_mask

    return F.normalize(mean_embeddings, p=2, dim=1)



from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import numpy as np
import nltk
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
def extract_top_keywords(positive_file, top_n=50):
    # 1. Load Data
    try:
        with open(positive_file, "r", encoding='utf-8') as f:
            pos_text = f.readlines()
    except FileNotFoundError:
        print(f"Error: File '{positive_file}' not found.")
        return []
    vectorizer = CountVectorizer(
        ngram_range=(1, 3), 
        stop_words='english',
        min_df=2 
    )
    try:
        X = vectorizer.fit_transform(pos_text)
    except ValueError:
        print("Error: Input text is too small or contains only stop words.")
        return []
    counts = np.asarray(X.sum(axis=0)).flatten()
    vocab = vectorizer.get_feature_names_out()
    ranked_indices = np.argsort(counts)[::-1]
    print("boost_keywords = [")
    for i in range(min(top_n, len(ranked_indices))):
        idx = ranked_indices[i]
        term = vocab[idx]
        word_count = len(term.split())
        score = 0.05
        if word_count == 1:
            score = 0.05
        elif "quantum" in term or "circuit" in term:
            score = 0.10
        else:
            score = 0.15
        strong_single_words = ["qiskit", "cirq", "ansatz", "unitary", "qubit"]
        if term in strong_single_words:
            score = 0.15
        print(f"    (\"{term}\", {score}),")
    print("]")

