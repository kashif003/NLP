import tqdm
import time
import os
import requests

from utils import paper_ID_extractor

def download_html(arxiv_id: str, save_dir: str = "./data/html_source") -> bool:
    """
    Download HTML version of an arxiv paper and save it locally.

    Parameters
    ----------
    arxiv_id : str
        The arxiv paper ID.
    save_dir : str
        Directory to save the HTML files.

    Returns
    -------
    bool
        True if download was successful, False otherwise.
    """
    url = f"https://arxiv.org/html/{arxiv_id}"
    response = requests.get(url, allow_redirects=True)

    if response.status_code != 200:
        return False

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{arxiv_id}.html")
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(response.text)

    return True


if __name__ == "__main__":
    paper_list = paper_ID_extractor("paper_list_12.txt", 100)

    success, failed = [], []

    for i, arxiv_id in enumerate(tqdm.tqdm(paper_list)):
        print(f"[{i+1}/{len(paper_list)}] Downloading {arxiv_id}...")
        if download_html(arxiv_id):
            success.append(arxiv_id)
        else:
            failed.append(arxiv_id)
        time.sleep(15)  # respect robots.txt crawl-delay

    print(f"\nDownload complete: {len(success)} success, {len(failed)} failed")
    print(f"Failed IDs: {failed}")