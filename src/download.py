import tqdm
import time
import os
from utils import paper_ID_extractor
from utils import download_paper
paper_list =  paper_ID_extractor("paper_list_12.txt", 100 )

# the code below is used to download  the paper
for i,id in tqdm.tqdm(enumerate(paper_list)):
    print(i,"[INFO] waiting for delay...")
    time.sleep(15)
    download_paper(id)
print("download finished")
