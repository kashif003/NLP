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
import arxiv
import time
import tarfile
import mimetypes
import json
import time
def download_paper(paper_ID):
    """
    Downloads the paper PDF and LaTeX source from arXiv and stores them in a cache folder.
    also checks if the paper is already dodwnloaded.
    """
    import logging
    logging.basicConfig(level=logging.INFO)
    # Ensure cache directories exist
    cache_dir = "data"
    pdf_cache_dir = os.path.join(cache_dir, "pdf_source")
    latex_cache_dir = os.path.join(cache_dir, "latex_source")
    os.makedirs(pdf_cache_dir, exist_ok=True)
    os.makedirs(latex_cache_dir, exist_ok=True)
    # If both PDF and latex cache exist, skip
    pdf_path_cached = os.path.join(pdf_cache_dir, f"{paper_ID}.pdf")
    latex_dir_cached = os.path.join(latex_cache_dir, paper_ID)
    if os.path.exists(pdf_path_cached) and os.path.isdir(latex_dir_cached):
        logging.info("Paper already exists in cache (pdf + latex), skipping download.")
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

    logging.info(f"Downloading paper with ID: {paper_ID}")
    # Always try to download PDF first
    try:
        pdf_path = paper.download_pdf(filename=f"{paper_ID}.pdf", dirpath=pdf_cache_dir)
        logging.info(f"PDF downloaded successfully: {pdf_path}")
    except Exception as e:
        logging.error(f"Failed to download PDF for {paper_ID}: {e}")
        return

    # Then attempt to download and extract LaTeX source. If unavailable, keep the PDF only.
    try:
        extract_dir_path = os.path.join(latex_cache_dir, paper_ID)
        os.makedirs(extract_dir_path, exist_ok=True)

        # download_source may raise if no source is available
        source_tar_path = None
        try:
            source_tar_path = paper.download_source(
                filename=f"{paper_ID}.tar.gz",
                dirpath=extract_dir_path
            )
        except Exception:
            logging.warning(f"No LaTeX source available for {paper_ID}; PDF only.")
            # remove empty extract dir if nothing downloaded
            try:
                if not any(os.scandir(extract_dir_path)):
                    os.rmdir(extract_dir_path)
            except Exception:
                pass
            return

        if not source_tar_path:
            logging.warning(f"No LaTeX source returned for {paper_ID}; PDF only.")
            return

        mime_type, _ = mimetypes.guess_type(source_tar_path)
        if mime_type not in ["application/gzip", "application/x-tar"] and not tarfile.is_tarfile(source_tar_path):
            logging.warning(f"Downloaded source is not a valid tar archive: {source_tar_path}")
            # keep PDF, remove any invalid file
            try:
                os.remove(source_tar_path)
            except Exception:
                pass
            return

        if tarfile.is_tarfile(source_tar_path):
            with tarfile.open(source_tar_path, "r:*") as tar:
                tar.extractall(path=extract_dir_path)
            try:
                os.remove(source_tar_path)
            except Exception:
                pass
            logging.info(f"LaTeX source extracted successfully for {paper_ID}.")
        else:
            logging.warning(f"Source file is not a valid tar archive: {source_tar_path}; PDF only.")
            try:
                os.remove(source_tar_path)
            except Exception:
                pass
            return
    except Exception as e:
        logging.error(f"Failed to download or extract LaTeX source for {paper_ID}: {e}")