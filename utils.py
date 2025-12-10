import arxiv
import time

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
    client= arxiv.Client()
    search = arxiv.Search(id_list=[paper_ID])
    paper = next(client.results(search))
    if not paper.source_url:
        print("paper not found:",paper_ID)
    elif "quant-ph" not in paper.categories:
        print("wrong category paper",paper_ID, paper.categories)
    print(f'Downloading the paper with ID: {paper_ID} ',"-"*30)
    paper.download_pdf(filename=f"{paper_ID}.pdf", dirpath="paper_pdf" )

import os
def delete_paper(path):
    if os.path.exists(path):
        print("Deleting the previous paper")
        os.remove(path)
    else:
        print("Path to delete the paper is not correct")



