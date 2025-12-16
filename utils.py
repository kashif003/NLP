import os
import time
import json
import re
import tarfile
import mimetypes
from typing import List
import arxiv
from nltk.tokenize import sent_tokenize
from transformers import AutoTokenizer
from adapters import AutoAdapterModel
from file_reader import LatexReader
from config import *
from latex_preprocessing import Latex_preprocessor
from embeddings import *
import torch.nn.functional as F
import nltk
from nltk.tokenize import sent_tokenize
from pdf_image_extractor import *



path_txt= "/media/kashif/hard drive/IV SEM/NLP/FInal_project/Additional Material-20251208/paper_list_11.txt"
def paper_ID_extractor(path):
    paper_list=[]
    with open(path, "r") as f:
        for line in f:
            line = line.strip()         
            if line:                     
                paper_list.append(line.split(":", 1)[1])
    return paper_list


def paper_downloader(paper_ID):
    time.sleep(3)
    client = arxiv.Client()
    search = arxiv.Search(id_list=[paper_ID])
    paper = next(client.results(search))
    if not any(cat.startswith("quant-ph") for cat in paper.categories):
        print(f"Paper {paper_ID} is not in quant-ph, categories: {paper.categories}. SKIPPING DOWNLOAD.")
        return
    if os.path.exists(f"paper_source/{paper_ID}"):
        print("Paper already exists, SKIPPING DOWNLOADING!!!!!!!!!!!!!!!!!")
        return
    if not paper.source_url:
        print("Paper not found:", paper_ID)
        return
    print(f'Downloading the paper with ID: {paper_ID} ', "-"*30)
    # Ensure output directories exist
    os.makedirs("paper_pdf", exist_ok=True)
    os.makedirs("paper_source", exist_ok=True)
    # downloading pdf
    pdf_path = paper.download_pdf(filename=f"{paper_ID}.pdf", dirpath="paper_pdf")
    # chcking if latex source exists.
    try:
        extract_dir_path = os.path.join("paper_source", paper_ID)
        os.makedirs(extract_dir_path, exist_ok=True)
        source_tar_path = paper.download_source(
            filename=f"{paper_ID}.tar.gz",
            dirpath=extract_dir_path
        )
        # Detect mimetype (simple check)
        mime_type, _ = mimetypes.guess_type(source_tar_path)
        if mime_type not in ["application/gzip", "application/x-tar"]:
            print(f"Downloaded source is not a tar.gz file: {source_tar_path}")
            return
        # Extract safely
        if tarfile.is_tarfile(source_tar_path):
            with tarfile.open(source_tar_path, "r:*") as tar:  
                tar.extractall(path=extract_dir_path)
            # Optionally remove the tar file
            os.remove(source_tar_path)
        else:
            print("Source file is not a valid tar archive:", source_tar_path)
    except Exception as e:
        print(f"Failed to download or extract source for {paper_ID}: {e}")


# used to delete the paper source as well as pdf
def delete_paper(path):
    if os.path.exists(path):
        print("Deleting the previous paper")
        os.remove(path)
    else:
        print("Path to delete the paper is not correct")

# used to get the pdfs which have quantum circuit. 
def get_quantum_circuit_pdf_text(json_path: str, max_pdfs: int, output_txt_path: str = "quantum_circuit_data/papers.txt"):
    """
    Reads the list of PDFs from a JSON file, downloads them, extracts LaTeX content,
    and saves the combined text to a text file.

    Args:
        json_path (str): Path to the JSON file containing PDF IDs.
        max_pdfs (int): Maximum number of PDFs to process.
        output_txt_path (str): Path to save the combined text file.
    
    Returns:
        list: A list of strings, each string containing the combined text of one PDF.
    """
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_txt_path), exist_ok=True)
    # Load the JSON file
    with open(json_path, "r") as f:
        json_file = json.load(f)
    pdf_content = []
    for i, pdf_id in enumerate(list(json_file.keys()), 1):
        print(f"\nProcessing PDF {i}/{max_pdfs}: {pdf_id}")
        # Download the PDF and source
        try:
            paper_downloader(pdf_id)
        except Exception as e:
            print(f"Failed to download {pdf_id}: {e}")
            continue
        # Read LaTeX source
        source_path = f"paper_source/{pdf_id}"
        reader = LatexReader(source_path)
        content_dict = reader.process_contents()
        print("content processed")
        if not content_dict:
            print(f"Problem with {pdf_id} – SKIPPING!")
            continue
        # Combine content of all files for this PDF
        temp_content = []
        for file_dict in content_dict:
            content = ". ".join(file_dict)
            temp_content.append(content)
        pdf_text = " ".join(temp_content)
        pdf_content.append(pdf_text)
        print(f"Saved content for {pdf_id}. Total PDFs collected: {len(pdf_content)}")
        if i >= max_pdfs:
            break
    # Save combined text to output file
    with open(output_txt_path, "w") as f:
        for item in pdf_content:
            f.write(item + "\n\n")  # optional extra newline between PDFs
    print(f"\nSaved all PDF text to {output_txt_path}. Total PDFs processed: {len(pdf_content)}")
    return pdf_content

# this fucntion is used to make a json dict of paper which have quantum circuit in them.
def find_and_save_quantum_circuit_titles(paper_list: List[str],
                                         output_json: str = "quantum_circuit_titles.json",
                                         batch_size: int = 200,
                                         sleep_seconds: int = 3) -> dict:
    """
    Given a list of arXiv IDs, find those with 'quantum circuit' or 'quantum circuits'
    in the title (case-insensitive), save them as JSON {paper_id: title}, and return the dict.
    """
    # match "quantum circuit" or "quantum circuits" as whole words, ignore case [web:42][web:48]
    target_pattern = re.compile(r"\bquantum circuits?\b", flags=re.IGNORECASE)
    total_checked = 0
    all_matching = []
    print("Checking for papers with 'quantum circuit(s)' in the title.")
    for i in range(0, len(paper_list), batch_size):
        batch_ids = paper_list[i:i + batch_size]
        print(f"Processing batch {i // batch_size + 1}, size {len(batch_ids)}...")

        search = arxiv.Search(
            id_list=batch_ids,
            max_results=len(batch_ids)
        )  # arxiv.Search usage follows the documented API.[web:46][web:49]
        for result in search.results():
            title = result.title or ""
            paper_id = result.get_short_id()
            if target_pattern.search(title):
                all_matching.append((paper_id, title))
        total_checked += len(batch_ids)
        time.sleep(sleep_seconds)
    print(f"\nTotal IDs checked: {total_checked}")
    print(f"Number with 'quantum circuit(s)' in title: {len(all_matching)}\n")
    matches_dict = {pid: title for pid, title in all_matching}
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(matches_dict, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(matches_dict)} entries to {output_json}")
    return matches_dict


# use to get the sentence of quantum circuit
def get_quantum_circuit_sentences(file_path, definition_pattern):
    # load the text from the quantum circuit papers
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
    # making sentneces of the text
    sentences = sent_tokenize(text)
    # filtering the text based on quantum circuit keyword
    quantum_sentences = [s for s in sentences if "quantum circuit" in s.lower()]
    # further filtering based on definition of quantum circuit
    quantum_sentences= []
    for sent in quantum_sentences:
        sent_lower= sent.lower()
        sentence= any(re.search(p, sent_lower) for p in definition_pattern)
        quantum_sentences.append(sentence)
    return quantum_sentences

# this fucntion is used to laod the model and tokenizer
def load_model_and_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoAdapterModel.from_pretrained(model_name)
    model.load_adapter(
    "allenai/specter2",
    source=adapter_configs["allenai/specter2"]["source"],
    load_as=adapter_configs["allenai/specter2"]["load_as"],
    set_active=adapter_configs["allenai/specter2"]["set_active"])
    model.set_active_adapters("specter2_proximity")
    model.eval()
    return model, tokenizer


# fucntion used to clean the sentence
def clean_sentences(sent):
    sent= sent.replace("(", " ").replace(")", " ").replace("*", "")
    sent = re.sub(r'\s+', ' ', sent)  # replaces 1 or more whitespace with a single space
    sent = sent.strip().replace(" .", ".").replace(". ", ".")
    return sent



# this fucntion is used to get the caption and text from the latex source
def get_caption_description(path):
    reader = LatexReader(path)
    figure_descriptions = []
    figure_captions = []
    for description in reader.get_figure_description():
        dis_text= Latex_preprocessor(description).clean_caption_description()
        if dis_text is None:
            continue
        figure_descriptions.append(dis_text)

    for caption in reader.get_figure_captions():
        cap_text= Latex_preprocessor(caption).clean_caption_description()
        if cap_text is None:
            continue
        figure_captions.append(cap_text)
    return figure_captions, figure_descriptions



# this fucntion is used to get the caption/description to match with the anchor embeddings and return only that text which is crossing the threshold.
def compare_embeddings(text_list,model, tokenizer, anchor, threshold=0.80):
    """
    text_list= caption_list or description list.
    anchor= embedding or anchor quantum circuit sentences.
    """
    score=0
    if len(text_list)==0:
        return False, 0
    anchor_emb = anchor.unsqueeze(0)
    for text in text_list:
        if len(text)==0:
            continue
        embeddings=get_anchor_embedding(text,  model, tokenizer)
        score=F.cosine_similarity(embeddings, anchor_emb, dim=1).item()
        if score>=threshold:
            return True, score
    return False, score



# this fucntion is used to get the meta data
def get_meta_data_from_pdf(figure_data, caption):
    """
    text: need to be selected according to embeddings.
    based on the caption/disc provided  this fucntion gets:
     1. arxiv number of pdf. done
     2. page_number where the figure if found.
     3. figure_number.
    """
    meta_data= {}
    for data in figure_data:
        figure_caption= data[0]
        figure_number= data[1]
        page_number= data[2]
        figure_caption= clean_json_caption(figure_caption)
        sentences_1 = sent_tokenize(figure_caption)
        sentences_2 = sent_tokenize(caption)
        is_similar, score=compare_captions(sentences_1[0], sentences_2[0], threshold= 0.5)
        if is_similar:
            meta_data["fig_number"]= figure_number
            meta_data["page_number"]= page_number
    if not meta_data:
        print("No caption found.")
        print("caption:",repr(caption))
    return meta_data



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
   

def clean_json_caption(text):
        text= text.lower()
        # highlight the figure refrence.
        text = re.sub(r'\\ref\{[^}]*fig[^}]*\}', '<Reference>', text)
        # remove the formating things.
        text = re.sub(r'\$.*?\$', '', text, flags=re.DOTALL)
        text = re.sub(r'\\[a-zA-Z]+\{.*?\}', '', text)
        text = re.sub(r'\\\w+\(.*?\)', '', text)
        text = re.sub(r"r'\\\w+\(.*?\)'", " ", text)
        text = re.sub(r'\\\S+', '', text)
        text = re.sub(r'-\n', ' ', text)
        text = re.sub(r'\n', ' ', text)
        text = re.sub(r'-', ' ', text)
        text = re.sub(r'\^', '', text)
        text = re.sub(r'[`_|{}~%&$\(\)\[\]§=,+]', ' ', text)
        text = re.sub(r"'", '', text)
        text = re.sub(r'"', '', text).replace("!", " .").replace(" ?", " .")
        # text= re.sub(" .", ". ", text) # gives the 

        text = re.sub(r" {2,}", " ", text)
        text = re.sub(r'fig\.\s+', '', text, flags=re.IGNORECASE).strip()
        text = re.sub(r"\b(?:fig|figure)\s+\d+\s*:?", "", text, flags=re.IGNORECASE).strip()

        return text


def process_paper(path, anchor, model, tokenizer, threshold=0.96):
    json_file= {}
    paper_list= paper_ID_extractor(path)
    figure_name=1

    for paper_idx,paper_id in enumerate(paper_list,1):
        process_pdf=False

        print("PAPER_NUMBER:", paper_idx)
        get_data=False
        # get 10 images for now only
        if len(list(json_file.keys()))>=10:
            break
        if os.path.exists("temp_Images"):   # removiing the previous pdf images
            shutil.rmtree("temp_Images")
        # downloading the paper 
        paper_downloader(paper_id)
        os.makedirs("temp_Images", exist_ok=True)    #  making temp folder to check images.
        pdf_path= f"paper_pdf/{paper_id}.pdf"
        source_path= f"paper_source/{paper_id}"
        # getting all images from the pdf
        PdfImageExtractor("pdffigures2").extract_all_figures(pdf_path,"temp_Images")     
        if os.path.exists(source_path):
            print("Source file exists! Processing Latex files.") # need to edit this
        else:
            print("source file does not exist! Processing PDF.")
            process_pdf=True
        if process_pdf:
            print("Working on pdf.") # need to add the code here
        else:
        # Getting captions and respective descriptions.
            cap_disc= LatexReader(source_path).get_caption_description_dict()
        # getting the data about the figures
        with open("temp_Images/figures_all.json", "r", encoding="utf-8") as f:
            figure_list = json.load(f)
        # sorting the figure data in order (caption, fig no, page no)
        figure_data= []
        for dict in figure_list:
            figure_data.append((dict["caption"], dict["name"], dict["page"]))
        # comparing the caption and description of the pdf
        for cap_idx,(caption, description) in enumerate(cap_disc.items()):
            # checking if caption is matching with anchor embeddings
            caption_score= compare_embeddings(caption,model, tokenizer, anchor ,threshold=threshold)
            if caption_score[0]:
                found_match = True
                # if it matches get the meta data (caption, fig no, page no)
                meta_data= get_meta_data_from_pdf(figure_data, caption)
                # if we are able to extract figure  number then we will extract the figure 
                if meta_data["fig_number"] is not None:
                    print(f"QUANTUM CIRCUIT FOUND IN {paper_id} AT PAGE NUMBER {meta_data["page_number"]} USING CAPTION. EXTRACTING THE META DATA.")
                    get_data=True
            else:
                # if caption does not match processing the description.
                print(f"Processing descriptions of the figures.\n {description} ")
                for disc in description:
                    score_2=compare_embeddings(disc[1],model, tokenizer, anchor, threshold=threshold)
                    print("score:",score_2[1])
                    # if we have a match then extracting meta data
                    if score_2[0]:
                        found_match = True
                        meta_data= get_meta_data_from_pdf(figure_data, caption)
                        if meta_data["fig_number"] is not None:
                            print(f"QUANTUM CIRCUIT FOUND IN {paper_id} AT PAGE NUMBER {meta_data["page_number"]}. EXTRACTING THE META DATA.")
                            get_data=True
                            break
            
            # getting the data (description, start_end, image, arxiv number)
            if get_data:
                get_data=False
                for disc in description:
                    meta_data.setdefault("description", []).append(disc[1])
                    meta_data.setdefault("start_end", []).append(description[0])
                meta_data["arxiv_id"] = paper_id
                # saving the image in another folder.
                print("saving image:", meta_data["fig_number"])
                os.makedirs("quantum_circuit_images", exist_ok=True)
                src_path = os.path.join("temp_Images/", f"figure_{meta_data["fig_number"]}.png")
                dst_path = os.path.join("quantum_circuit_images", f"{figure_name}.png")
                shutil.copy2(src_path, dst_path)
                json_file[f"{figure_name}.png"]= meta_data
                print("IMAGE SAVED SUCCESSFULLY",meta_data["fig_number"])
                figure_name+=1

    return json_file



KEEP_TERMS = [
    "consists of", "composed of", "defined as", "represents",
    "implements", "used to", "describes", "models"
]
# this fucntion is used to preprocess the sentences used for embeddings.

def extract_qc_sentences(tokens):
    def normalize(sent):
        sent = sent.lower()
        sent = re.sub(r"\bquantum circuits?\b", "[quantum circuit]", sent, flags=re.IGNORECASE).replace("%", "").replace("&", "").replace("$", "").replace("(", "").replace(")", "").replace("§", "").replace("=", "").replace(",", "")
        sent = re.sub(r"\d+", "[num]", sent)
        sent = sent.replace("\n", " ")
        sent = re.sub(r"\s+", " ", sent)
        return sent.strip()
    def is_trivial(s):
        if len(s.split()) < 8:
            return True
        if re.search(r"\b(fig|table|sec|section|see)\b", s):
            return True
        if re.match(r"^(introduction|methods|results)", s):
            return True
        return False
    def passes_keyword_gate(s):
        return any(term in s for term in KEEP_TERMS)
    sentences = [normalize(s) for s in tokens]
    sentences = [s for s in sentences if not is_trivial(s)]
    filtered = [s for s in sentences if passes_keyword_gate(s)]
    qc_sentences =filtered#  [s for s in filtered if "[quantumcircuit]"  in s.lower()]
    return qc_sentences



from wordcloud import WordCloud
import matplotlib.pyplot as plt
def word_cloud(sentences):
    cloud = WordCloud().generate(" ".join(sentences) )
    plt.imshow(cloud)
    plt.axis("off")
    plt.show()


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
            words = [word for word in words if word not in stopwords.words('english')]
            
            final_words = []
            for word in words:
                
                if word.isalnum(): 
                    final_words.append(word) 
                    
            sentences_partition.append(final_words)
            num_words_partition += len(final_words)
            
    return sentences_partition, num_words_partition




def get_word_embeddings(model, wordlist, topn):
    w_cloud = {}
    for t_w in wordlist:
        w_sim = {}
        if topn != 0:
            silimarity_ws = model.most_similar(positive=[t_w], topn=topn)
            for label, cosine in silimarity_ws:
                embedding = model.get_vector(label)
                w_sim[label] = embedding
        w_sim[t_w] = model.get_vector(t_w)
        w_cloud[t_w] = w_sim

    return w_cloud

def get_labels_embeddings(word_cloud):
    embeddings=[]
    labels=[]
    for key in word_cloud:
        word_embed = word_cloud.get(key)
        for word in word_embed:
            labels.append(word)
            embeddings.append(word_embed.get(word))
    
    return np.array(labels), np.array(embeddings)

from matplotlib import cm
from sklearn.manifold import TSNE

def tsne_plot(word_cloud, topn):
    labels, tokens = get_labels_embeddings(word_cloud)

    if topn != 0:
        colors = cm.rainbow(np.linspace(0, 1, len(labels))) 
    else:
        color = ["orange"]
    
    tsne_model = TSNE(perplexity=4, n_components=2, init='pca', n_iter=3500, random_state=10)
    tsne_values = tsne_model.fit_transform(tokens)

    x = []
    y = []
    
    for value in tsne_values:
        x.append(value[0])
        y.append(value[1])
        
    plt.figure(figsize=(16, 10)) 
    
    for i in range(len(x)):
        if topn != 0:
            if i % (topn+1) == 0:
                color = colors[i-1]
        plt.scatter(x[i],y[i],color=color)
        plt.annotate(labels[i], xy=(x[i], y[i]))
   
    plt.show()


