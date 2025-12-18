from utils import *

paper_list=  paper_ID_extractor("paper_list_11.txt")


for paper in paper_list:   
    download_paper(paper)
