       # print(repr(preprocessor.tokenize()[0]))



from utils import *
from read_pdf import *
from preprocessing import *
import os

list_path= "paper_list_11.txt"
paper_list= paper_ID_extractor(list_path)

        
for i,paper in enumerate(paper_list,1):
    # downloading the paperpath
    path=f"paper_pdf/{paper}.pdf"
    # paper_downloader(paper)
    # read the paper using pdf parser.
    reader = PdfReader(path)
    # getting all pages as list
    all_pages= reader.pages
    # print(repr(all_pages[1]))
    # prerocessing the page
    print("Looking for figure", "-"*30)
    print("reading the paper:", paper, "-"*30)
    preprocessor=Preprocess(all_pages)
    preprocessor.process_text()
    if preprocessor._locate_figure() is None:
        print(f"No figure found in {paper} ! Skipping the paper", "-"*30)
        continue
    else:
        print("Figure found in",paper)
        # print(repr(preprocessor._locate_figure()))
    break

        # need to send the list to a function which will decide if the figure is related to quantum circuits or not.
#############
        # take the paragraph around a figure and further preprocess it and then look for quantum circuits and related things.

    # break

    if i==10:
        break

# s.\nA   s.\nT
# make th preprocssor for all pages


"""
need to do basic cleaning, and then check if figure is in the text if it is then +- one page do advanced cleaning and process the text to find things related to quantum circuits.
"""

# Get the intrested figure

# get the meta data for the json file.

# get the figure and store it


# from pdf_image_extractor import *
# extractor = PdfImageExtractor(tool_root= "pdffigures2")
# extractor.extract(pdf_path="paper_pdf/2405.20827.pdf", page= 17,out_dir= "Images", name="kashif")


