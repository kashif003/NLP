import tqdm
import time
import os
from utils import paper_ID_extractor
from utils import download_paper
paper_list =  paper_ID_extractor("./paper_list_11.txt", 99 )

# the code below is used to download  the paper
for id in tqdm.tqdm(paper_list):
    time.sleep(3)
    download_paper(id)
print("download finished")
