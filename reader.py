from text_preprocessor import *
from nltk.tokenize import sent_tokenize
import fitz 
import spacy
import re
class PdfReader:
    """Parses PDF files to extract text and figure metadata (captions, descriptions) using spaCy."""
    def __init__(self, pdf_path: str):
        """Initializes the reader with a PDF path and loads the English spaCy model."""
        self.path = pdf_path
        self.raw_text = [] 
        self.figure_details = {}
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            raise OSError("Please run: python -m spacy download en_core_web_sm")

    def read_pdf(self):
        """Extracts and caches raw text from every page of the PDF."""
        if self.raw_text:
            return self.raw_text
        with fitz.open(self.path) as doc:
            for i, page in enumerate(doc, 1):
                text = page.get_text("text")
                self.raw_text.append((i, text))
        return self.raw_text

    def get_caption_discription(self, sentence=10, caption=3):
        """Scans the PDF for 'Figure X' patterns and extracts the surrounding sentences as captions and descriptions."""
        self.read_pdf()
        self.figure_details = {}
        # Regex to capture Figure/Fig and the number separately
        figure_pattern = r'(?i)\b(Fig(?:ure)?\.?\s*(\d+(?:\.\d+)?))\b'
        for page_number, page_text in self.raw_text:
            # Process page with spaCy for accurate offsets and sentences
            doc = self.nlp(page_text)
            sentences = list(doc.sents)
            for idx, sent in enumerate(sentences):
                match = re.search(figure_pattern, sent.text)
                if match:
                    # 1. Normalize Key (e.g., Figure_2)
                    fig_num = match.group(2)
                    norm_key = f"Figure_{fig_num}"
                    # 2. Extract raw text segments
                    # Caption: Current sentence + next 1
                    caption_slice = sentences[idx : idx + caption]
                    caption_raw = "".join([s.text_with_ws for s in caption_slice]).strip()
                    # Description: Current sentence + next 4
                    desc_slice = sentences[idx : idx + sentence]
                    description_raw = "".join([s.text_with_ws for s in desc_slice]).strip()
                    # Character offsets relative to the page text
                    start_pos = desc_slice[0].start_char
                    end_pos = desc_slice[-1].end_char
                    # 3. Aggregation Logic
                    if norm_key not in self.figure_details:
                        # This is the first time we see the figure (Primary Caption)
                        self.figure_details[norm_key] = {
                            "figure_label": f"{fig_num}",
                            "caption": caption_raw,
                            "caption_page": page_number,  # The specific page where caption is found
                            "descriptions": [description_raw],
                            "start_end_positions": [(start_pos, end_pos)],
                            "all_pages_mentioned": [page_number]
                        }
                    else:
                        # Append new description and position to existing figure
                        self.figure_details[norm_key]["descriptions"].append(description_raw)
                        self.figure_details[norm_key]["start_end_positions"].append((start_pos, end_pos))
                        
                        # Add to the list of all pages mentioned if not already there
                        if page_number not in self.figure_details[norm_key]["all_pages_mentioned"]:
                            self.figure_details[norm_key]["all_pages_mentioned"].append(page_number)
        return self.figure_details

import os
import re

class LatexReader:
    """Reads .tex files from a directory to extract and map figure captions and descriptions."""
    def __init__(self, folder_path: str):
        """Initializes the reader with a folder path."""
        self.path = folder_path
        self.docs = []
        self.raw_content = []
        self.processed_content = []

    def _load_tex_files(self) -> None:
        """Finds all .tex files in the directory."""
        if not os.path.isdir(self.path):
            print(f"Error: '{self.path}' is not a valid directory.")
            return
        tex_files = [f for f in os.listdir(self.path) if f.endswith('.tex')]
        self.docs = tex_files

    def get_docs(self):
        """Returns the list of found .tex filenames."""
        return self.docs

    def get_raw_content(self):
        """Reads the content of all .tex files into memory."""
        self._load_tex_files()
        print("Number of files:",len(self.docs))
        if self.raw_content:
            print("Raw content already loaded. Skipping reload.")
            return self.raw_content
        if not self.docs:
            print("Warning: No .tex files found to load.")
            return []

        content_list = []
        for doc_name in self.docs:
            file_path = os.path.join(self.path, doc_name)
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    content = file.read()
                    content_list.append(content)
                    print(f"✓ Loaded: {doc_name} ({len(content)} characters)")
            except FileNotFoundError:
                print(f"✗ Error: File not found - {doc_name}")
            except Exception as e:
                print(f"✗ Error reading {doc_name}: {str(e)}")
        print(f"Total files loaded: {len(content_list)}/{len(self.docs)}\n")
        self.raw_content = content_list
        return self.raw_content

    def process_contents(self):
        """Processes raw content using the Latex_preprocessor."""
        self.get_raw_content()
        if self.processed_content:
            print("Processed content already loaded. Skipping reprocess.")
            return self.processed_content
        if not self.raw_content:
            print("Warning: No raw content to process.")
            return []
        for i, content in enumerate(self.raw_content, 1):
            try:
                preprocessor = Latex_preprocessor(content)
                preprocessor.process_text()
                self.processed_content.append(preprocessor.processed_text)
                print(f"✓ Processed file {i} ({len(preprocessor.processed_text)} characters)")
            except Exception as e:
                print(f"✗ Error processing file {i}: {str(e)}")
        print(f"Total files processed: {len(self.processed_content)}/{len(self.raw_content)}\n")
        return self.processed_content

    def get_processed_content(self):
        """Returns the list of processed file contents."""
        return self.processed_content

    def locate_figure(self):
        """Extracts all figure labels from the loaded files."""
        print("GEtting raw content")
        self.get_raw_content()
        # Handles optional arguments in \begin{figure}[...]
        pattern = r'\\begin\{(?:figure\*?|wrapfigure|subfigure).*?\}.*?\\label\{(.*?)\}.*?\\end\{(?:figure\*?|wrapfigure|subfigure).*?\}'
        labels = []
        for text in self.raw_content:
            found_labels = re.findall(pattern, text, re.DOTALL)
            if len(found_labels)==0:
                continue
            labels.append(found_labels)
        return [label for sublist in labels for label in sublist]
    

    def get_figure_discription(self):
        """Extracts paragraphs referencing figures via labels."""
        label_list = self.locate_figure()
        text = "".join(self.raw_content)
        discription = []
        for label in label_list:
            needle = "\\ref{" + label + "}"         
            pos = text.find(needle)
            if pos == -1:
                continue
            # find paragraph start: previous double newline
            start = text.rfind("\n\n", 0, pos)
            if start == -1:
                start = 0
            else:
                start += 2  # skip the \n\n itself
            # find paragraph end: next double newline
            end = text.find("\n\n", pos + len(needle))
            if end == -1:
                end = len(text)
            para = text[start:end].strip()
            discription.append(para)
        return discription


    def _extract_caption_from_block(self,block):
        """Extracts the caption text from a LaTeX block, handling nested braces."""
        match = re.search(r'\\caption(?:\[[^\]]*\])?\s*\{', block)
        if not match:
            return None
        start = match.end()
        brace_level = 1
        i = start
        while i < len(block) and brace_level > 0:
            if block[i] == '{':
                brace_level += 1
            elif block[i] == '}':
                brace_level -= 1
            i += 1
        return block[start:i-1].strip()

    

    def get_figure_captions(self):
        """Extracts all captions found strictly inside figure environments."""
        self.get_raw_content()

        figure_env_pattern = r'''
            \\begin\{(figure\*?|wrapfigure|subfigure)\}  # begin env
            (.*?)                                        # content
            \\end\{\1\}                                  # end same env
        '''
        captions = []
        for text in self.raw_content:
            for _, block in re.findall(
                figure_env_pattern,
                text,
                re.DOTALL | re.VERBOSE
            ):
                caption = self._extract_caption_from_block(block)
                if caption:
                    print(caption)
                    print("+"*100)
                    captions.append(caption)
        return captions
    

    def get_caption_label_map(self):
        """Maps figure labels to their corresponding captions."""
        self.get_raw_content()
        figure_env_pattern = r'''
            \\begin\{(figure\*?|wrapfigure|subfigure)\}
            (.*?)
            \\end\{\1\}
        '''
        caption_label = {}
        for text in self.raw_content:
            for _, block in re.findall(
                figure_env_pattern,
                text,
                re.DOTALL | re.VERBOSE
            ):
                caption = self._extract_caption_from_block(block)
                label_match = re.search(r'\\label\{([^}]+)\}', block)
                if caption and label_match:
                    label = label_match.group(1)
                    caption_label[label] = caption.strip()

        return caption_label
    
    def get_description_label_map(self):
        """Maps figure labels to lists of surrounding text contexts (descriptions)."""
        label_list = self.locate_figure()
        text = "".join(self.raw_content)
        desc_map = {}
        for label in label_list:
            needle = "\\ref{" + label + "}"
            start_pos = 0
            while True:
                pos = text.find(needle, start_pos)
                if pos == -1:
                    break
                # paragraph start
                start = text.rfind("\n\n", 0, pos)
                start = 0 if start == -1 else start + 2
                # paragraph end
                end = text.find("\n\n", pos + len(needle))
                end = len(text) if end == -1 else end
                description = text[start:end].strip()
                desc_map.setdefault(label, []).append(
                    ((start, end), description)
                )
                start_pos = pos + len(needle)
        return desc_map

    def get_caption_description_dict(self):
        """Combines cleaned captions and descriptions into a dictionary keyed by caption."""
        caption_map = self.get_caption_label_map()       # label -> caption
        description_map = self.get_description_label_map()  # label -> [desc]
        result = {}
        for label, caption in caption_map.items():
            caption= Latex_preprocessor(caption).clean_caption_discription()
            if label in description_map:
                discription_list= [(dics[0],Latex_preprocessor(dics[1]).clean_caption_discription()) for dics in description_map[label]]
                result[caption] = discription_list
        return result


