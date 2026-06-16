def paper_ID_extractor(path, n= None):
    """Extracts paper IDs from a file by parsing the substring after the first colon on each line."""
    paper_list=[]
    with open(path, "r") as f:
        for line in f:
            line = line.strip()         
            if line:                     
                paper_list.append(line.split(":", 1)[1])
        if n:
            return paper_list[:n]
        else:
            return paper_list


import os
import tarfile

import arxiv
import os
import tarfile
import urllib.request  # Used to handle the new download format
import arxiv

def download_paper(paper_ID):
    """
    Downloads the paper PDF and LaTeX source from arXiv and stores them in a cache folder.
    Checks independently if each format is already downloaded.
    """
    # Ensure cache directories exist
    cache_dir = "data"
    pdf_cache_dir = os.path.join(cache_dir, "pdf_source")
    latex_cache_dir = os.path.join(cache_dir, "latex_source")
    os.makedirs(pdf_cache_dir, exist_ok=True)
    os.makedirs(latex_cache_dir, exist_ok=True)

    # Define target paths to check presence independently
    expected_pdf_path = os.path.join(pdf_cache_dir, f"{paper_ID}.pdf")
    extract_dir_path = os.path.join(latex_cache_dir, paper_ID)

    # If BOTH already exist, we can safely skip the API call
    if os.path.exists(expected_pdf_path) and os.path.exists(extract_dir_path):
        print(f"Paper {paper_ID} is already fully cached (PDF & LaTeX).")
        return
    
    # Initialize client and fetch metadata
    client = arxiv.Client()
    search = arxiv.Search(id_list=[paper_ID])

    try:
        paper = next(client.results(search))
    except StopIteration:
        print(f"Paper {paper_ID} not found on arXiv.")
        return
        
    # Category filter
    if not any(cat.startswith("quant-ph") for cat in paper.categories):
        print(f"Paper {paper_ID} skipped (not in quant-ph).")
        return

    # 1. Download PDF if it doesn't exist yet
    if not os.path.exists(expected_pdf_path):
        try:
            # Using urllib.request.urlretrieve with the paper's pdf_url property
            urllib.request.urlretrieve(paper.pdf_url, expected_pdf_path)
            print(f"Successfully downloaded PDF for {paper_ID}")
        except Exception as e:
            print(f"[IMPORTANT] Unable to download the PDF for {paper_ID}:", e)
    else:
        print(f"PDF for {paper_ID} already exists. Skipping PDF download.")

    # 2. Download and Extract LaTeX Source if it doesn't exist yet
    if not os.path.exists(extract_dir_path):
        # Handle source_url whether it's a property or a method safely
        source_url = paper.source_url() if callable(getattr(paper, "source_url", None)) else paper.source_url
        
        if not source_url:
            print(f"No source URL available for {paper_ID}")
            return
            
        try:
            os.makedirs(extract_dir_path, exist_ok=True)
            source_tar_path = os.path.join(extract_dir_path, f"{paper_ID}.tar.gz")
            
            # Using urllib.request.urlretrieve with the paper's source_url
            urllib.request.urlretrieve(source_url, source_tar_path)
            
            if tarfile.is_tarfile(source_tar_path):
                with tarfile.open(source_tar_path, "r:*") as tar:
                    tar.extractall(path=extract_dir_path, filter='data') 
                os.remove(source_tar_path)
                print(f"Successfully extracted LaTeX source for {paper_ID}")
        except Exception as e:
            print(f"[IMPORTANT] Unable to download/extract LaTeX for {paper_ID}:", e)
    else:
        print(f"LaTeX source for {paper_ID} already exists. Skipping LaTeX download.")



