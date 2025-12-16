import fitz  # PyMuPDF
import os
from latex_preprocessing import *
from pdf_preprocessing import *
import re
from nltk.tokenize import sent_tokenize
# reads pdf
class PdfReader:
    def __init__(self, pdf_path: str):
        self.path = pdf_path
        self.raw_text= []
        self.preprocessed_text= None
        self.figure_details= None

    def __get_processed_text__(self):
        if self.preprocessed_text is None:
            self.preprocess_pdf()
        return self.preprocessed_text
    def read_pdf(self):
        if self.raw_text:
            return
        raw_text= []
        with fitz.open(self.path) as doc:
            for i,page in enumerate(doc,1):
                text=page.get_text("text")
                raw_text.append((i,text))
        self.raw_text=raw_text
        return self.raw_text
    
    def preprocess_pdf(self):
        self.read_pdf()
        preprocessor= Pdf_Preprocess(self.raw_text)
        self.preprocessed_text= preprocessor.process_text()
        return self.preprocessed_text
    
    def get_pages_with_figures(self):
        self.preprocess_pdf()
        # Robust figure-detection regex
        pattern = r'(?i)\bfig(?:ure)?\.?\s*\d+[a-zA-Z]?\b'
        self.figure_details = []
        # assume self.raw_text is list of (page_number, page_text)
        for page_number, page_text in self.raw_text:
            matches = re.findall(pattern, page_text)
            if len(matches)<1:
                continue
            self.figure_details.append((page_number, matches))
        print("Figure details: (page number, [figure/fig matches])\n",self.figure_details)


    def get_figure_discription(self,sentences=2):
        self.get_pages_with_figures()
        result = []

        for page_number, figures in  self.figure_details:
            previous_page= self.raw_text[page_number-2][1] if page_number > 1 else None
            next_page= self.raw_text[page_number+2][1] if page_number < len(self.raw_text) else None

            combined_pages = f"{previous_page} {self.raw_text[page_number-1]} {next_page}"

            # tokenize the combined figures
            sents = sent_tokenize(combined_pages)
            for idx, sent in enumerate(sents):
                # Check if sentence contains any figure reference
                if any(fig in sent for fig in figures):
                    # Get context sentences
                    start = max(0, idx - sentences)
                    end = min(len(sents), idx + sentences + 1)  # +1 because slice is exclusive
                    snippet = " ".join(sents[start:end])
                    result.append(snippet)

        return result



class LatexReader:
    def __init__(self, folder_path: str):
        self.path = folder_path
        self.docs = []
        self.raw_content = []
        self.processed_content = []

    def _load_tex_files(self) -> None:
        """Load all .tex files from the folder and store their names."""
        if not os.path.isdir(self.path):
            print(f"Error: '{self.path}' is not a valid directory.")
            return
        tex_files = [f for f in os.listdir(self.path) if f.endswith('.tex')]
        self.docs = tex_files

    def get_docs(self):
        """Return list of loaded .tex file names."""
        return self.docs

    def get_raw_content(self):
        """
        Load all text from .tex files into a list.
        Returns:
            List[str]: Each element is the content of a .tex file
        """
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
        """
        Process all text from .tex files using Latex_preprocessor.
        Returns:
            List[str]: Processed content of each .tex file
        """
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
        """Return processed content list."""
        return self.processed_content

    def locate_figure(self):
        """
        Extract all figure labels from the loaded .tex files.
        Returns:
            List[List[str]]: Each element is a list of labels found in a file
        """
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
        """
        Extract full caption from a single figure block using brace counting.
        Returns None if no caption is found.
        """
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
        """
        Extract captions that are strictly inside figure environments.

        Returns:
            List[str]: Flat list of figure captions
        """
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
        """
        Returns:
            dict[label -> caption]
        """
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
        """
        Returns:
            dict[label -> list[((start, end), description)]]
        """
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
        """
        Returns:
            dict[caption -> list[descriptions]]
        """
        caption_map = self.get_caption_label_map()       # label -> caption
        description_map = self.get_description_label_map()  # label -> [desc]

        result = {}

        for label, caption in caption_map.items():
            caption= Latex_preprocessor(caption).clean_caption_discription()
            if label in description_map:
                discription_list= [(dics[0],Latex_preprocessor(dics[1]).clean_caption_discription()) for dics in description_map[label]]

                result[caption] = discription_list
        return result


"""
(After pipeline is ready)
+ need to update the caption extraction it is not getting the caption properly on  2404.12603.
+ need the better discription extraction method (get discription based on the sentences.)
"""






