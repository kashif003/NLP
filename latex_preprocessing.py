# this file will preprocess the latex text only.


import nltk
import scipy
import re
from nltk.tokenize import sent_tokenize
import unicodedata


class Latex_preprocessor():
    def __init__(self, text:str):
        # putting the text in lower case
        self.raw_text = text
        self.sentences_tokens= []
        self.processed_text= []

    def process_text(self):
        if self.processed_text:
            return
        self.processed_text= self.preprocess_text(self.raw_text)
    
    def preprocess_text(self, text: str):
        text = " " + text.lower() + " "
        text = text.replace("&", "").replace("(", "").replace(")", "").replace("[", "").replace("]", "").replace("§", "").replace("=", "").replace(",", "").replace("`", " ").replace(":", " ").replace("_", " ")
        # remove text formatting
        text = self.remove_latex_formatting(text)
        # highlight_figures
        text = self.highlight_figures(text)
        # remove comments
        text = self.remove_comments(text)
        # remove bib
        text = self.remove_all_bibliographies(text)
        # remove author
        text = self.remove_author_affiliation(text)
        # remove_preamble_and_title
        text = self.remove_preamble_and_title(text)
        # remove tables
        text = self.remove_tables(text)
        # removing the equations from the latex
        text = self.remove_latex_equations(text)
        # removing the citations
        text = self.remove_latex_citations(text)
        # normalize brackets
        text = self.normalize_brackets(text)
        # replace \\
        text = re.sub(r"\\", "", text)
        # remove empty brackets
        text = re.sub(r"\(\s*\)", "", text)
        # UNCOMMENT THESE IMPORTANT STEPS:
        # Normalizing the data
        text = unicodedata.normalize("NFKD", text)
        # Replace hyphenated line breaks like "-\n12\n"
        text = re.sub(r"(-)?\n\d+\n+", r"\1 ", text)
        # Replace remaining newlines with space
        text = text.replace("\n", " ")
        # Replace | and ` with space
        text = text.replace("|", " ").replace("`", " ")
        # Replace colon-underscore and plain underscore with space
        text = text.replace(":_", " ").replace("_", " ")
        # Remove dots from common honorifics (keep the words)
        for pat in [" mrs. ", " ms. ", " mr. ", " dr. ", " prof. ", " dr.-ing. "]:
            text = text.replace(pat, pat.replace(".", ""))
        # Remove dotted ellipsis-like runs "...."
        text = re.sub(r"(?:\.\s+)+\.", " ", text)
        # Removing text like "- " or A–B patterns
        text = re.sub(r"(- |[A-Za-z0-9][-\u2013][A-Za-z0-9]| –)", "", text)
        # Replace multiple spaces with single space
        text = re.sub(r" {2,}", " ", text)
        # Replace links with placeholder
        text = re.sub(r"http\S+", " hrefl ", text)
        # Replace emails as well
        text = re.sub(r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}', ' email ', text)
        # Remove tilde
        text = re.sub(r"~", "", text)
        # Remove double quotes
        text = re.sub(r'"', '', text)
        # Remove single quotes
        text = re.sub(r"'", '', text)
        # Remove smart/curly quotes
        text = re.sub(r'"', '', text)
        text = re.sub(r'"', '', text)
        # From notebook - Pattern processing
        sent_patt = re.compile(r'(?<!\.|\!|\?|\:|\;|\s)\w[.,:;!?]\s+')
        multi_sym = re.compile(r'[!.,?=-]{2,}')
        time = re.compile(r'([0-2][0-9]:[0-5][0-9]([pm]|[am])*)|([0-2]*[0-9]*:*[0-5][0-9]([pm]|[am])+)|([0-9][0-9]*([pm]|[am])+)')
        date = re.compile(r'([0-3]*[0-9]\/[0-9]*\/[0-9]+)')
        all_sym = multi_sym.finditer(text)
        all_time = time.finditer(text)
        all_date = date.finditer(text)
        for m in all_sym:
            text = text.replace(m.group(), ' ' + m.group()[0] + ' ', 1)
        for m in all_time:
            text = text.replace(m.group(), ' tiform ', 1)
        for m in all_date:
            text = text.replace(m.group(), ' dtform ', 1)
        text = " " + text + " "
        all_pts = sent_patt.finditer(text)
        for m in all_pts:
            text = text.replace(m.group(), m.group()[0] + ' ' + m.group()[1] + ' ', 1)

        return text.strip()

    def remove_latex_formatting(self,text: str) -> str:
        """
        Remove all LaTeX text formatting commands (\textit, \textbf, etc.)
        Handles nested formatting like \textit{\textit{...}}.
        """
        # Keep removing formatting until none are left (for nested cases)
        max_iterations = 10
        for _ in range(max_iterations):
            pattern = r"\\text[a-z]+\{([^{}]*)\}"
            new_text = re.sub(pattern, r"\1", text)
            # If no changes, we're done
            if new_text == text:
                break
            text = new_text
        
        return text

    def highlight_figures(self,text: str) -> str:
        """
        Replace LaTeX figure environments with highlighted captions.
        Extracts caption and wraps it with <caption_start> and <caption_end>.
        """
        pattern = r"\\begin\{figure\}.*?\\caption\{([^}]*)\}.*?\\end\{figure\}"
        replacement = r"<caption_start>\1<caption_end>"
        return re.sub(pattern, replacement, text, flags=re.DOTALL)

    def remove_comments(self, text: str) -> str:
        """
        Remove all LaTeX comments (lines starting with %).
        """
        # Remove lines that start with % (after optional whitespace)
        text = re.sub(r"^\s*%.*$", "", text, flags=re.MULTILINE)
        # Remove trailing comments (% to end of line)
        text = re.sub(r"[^\\]%.*$", "", text, flags=re.MULTILINE)
        return text


    def normalize_brackets(self,text: str) -> str:
        """
        Replace all types of brackets with normal parentheses ().
        Handles: [], {}, ⟨⟩, ⟪⟫, 「」, 【】, ⦃⦄, etc.
        """
        # Square brackets [] -> ()
        text = text.replace('[', '(').replace(']', ')')
        
        # Curly braces {} -> ()
        text = text.replace('{', '(').replace('}', ')')
        
        # Angle brackets ⟨⟩ -> ()
        text = text.replace('⟨', '(').replace('⟩', ')')
        text = text.replace('〈', '(').replace('〉', ')')
        text = text.replace('⟪', '(').replace('⟫', ')')
        
        # Asian brackets -> ()
        text = text.replace('「', '(').replace('」', ')')
        text = text.replace('『', '(').replace('』', ')')
        text = text.replace('【', '(').replace('】', ')')
        text = text.replace('〔', '(').replace('〕', ')')
        
        # Mathematical brackets -> ()
        text = text.replace('⦃', '(').replace('⦄', ')')
        text = text.replace('⦅', '(').replace('⦆', ')')
        text = text.replace('((', '(').replace('))', ')')
        
        return text
    
    def remove_all_bibliographies(self, text: str) -> str:
        """
        Remove all common LaTeX bibliography formats:
        - thebibliography environment
        - \\bibliography{} and \\bibliographystyle{} commands
        - biblatex \\printbibliography command
        """
        # Remove \begin{thebibliography}...\end{thebibliography}
        text = re.sub(r"\\begin\{thebibliography\}.*?\\end\{thebibliography\}", 
                    "", text, flags=re.DOTALL)
        # Remove \bibliography{...} command (BibTeX/natbib)
        text = re.sub(r"\\bibliography\{[^}]*\}", "", text)
        # Remove \bibliographystyle{...} command
        text = re.sub(r"\\bibliographystyle\{[^}]*\}", "", text)
        # Remove \printbibliography (biblatex) with optional parameters
        text = re.sub(r"\\printbibliography(\[[^\]]*\])?", "", text)
        # Remove \addbibresource{...} command (biblatex)
        text = re.sub(r"\\addbibresource\{[^}]*\}", "", text)
        return text

    def remove_preamble_and_title(self, text: str) -> str:
        """
        Remove LaTeX preamble (everything before \\begin{document})
        and the \\maketitle command from the given text.
        """
        # Remove everything from the start up to and including \begin{document}
        text = re.sub(r"^.*?\\begin\{document\}", "", text, flags=re.DOTALL)
        # Optionally remove \maketitle (and trailing blank lines)
        text = re.sub(r"\\maketitle\s*", "", text)
        return text.strip()
    
    def remove_author_affiliation(self,text: str) -> str:
        """
        Remove all \\author{...} and \\affiliation{...} commands from LaTeX text.
        """
        # Remove \author{...}
        text = re.sub(r"\\author\{[^}]*\}", "", text)
        # Remove \affiliation{...}
        text = re.sub(r"\\affiliation\{[^}]*\}", "", text)
        
        return text
    
    def remove_tables(self, text: str) -> str:
        """
        Remove all LaTeX tabular environments from the given text.
        """
        pattern = r"\\begin\{tabular\}.*?\\end\{tabular\}"
        return re.sub(pattern, "", text, flags=re.DOTALL)
    
    def remove_latex_citations(self, text: str) -> str:
        """Remove LaTeX citations while preserving surrounding text."""
        # Remove \ref family (optional ~ or spaces before)
        ref_pattern = r"[~\s]*\\(ref|eqref|autoref|pageref)\{[^}]*\}"
        text = re.sub(ref_pattern, "", text)
        # Remove \cite family with optional suffixes (with optional trailing space)
        cite_pattern = r"[~\s]*\\cite(t|p|author|year|alp|alt|num)?\{[^}]*\}\s*"
        text = re.sub(cite_pattern, " ", text)  # Replace with space to avoid merged words
        
        return text

    def remove_latex_equations(self,text):
        # 1. Remove \begin{equation} ... \end{equation} and other math environments
        envs = [
            "equation", "equation*", "align", "align*", "gather", "gather*",
            "multline", "multline*", "flalign", "flalign*"
        ]
        # Build regex for environments
        env_pattern = "|".join(envs)
        text = re.sub(
            rf"\\begin{{({env_pattern})}}.*?\\end{{\1}}",
            "",
            text,
            flags=re.DOTALL
        )
        # 2. Remove display math: $$ ... $$
        text = re.sub(r"\$\$.*?\$\$", "", text, flags=re.DOTALL)
        # 3. Remove inline math: $ ... $ 
        text = re.sub(r"\$.*?\$", "", text)
        # 4. Remove inline math: \( ... \)
        text = re.sub(r"\\\(.*?\\\)", "", text, flags=re.DOTALL)
        # 5. Remove display math: \[ ... \]
        text = re.sub(r"\\\[.*?\\\]", "", text, flags=re.DOTALL)
        return text


    def tokenize(self):
       self.sentences_tokens =sent_tokenize(self.text)
       return self.sentences_tokens
    

    def locate_figure(self):
        """
        this fucntion is used to check where the fig/ figure is mention in the text
        """
        self.process_text()
        return self.processed_text
    
    def clean_caption_discription(self, Tuple=False):
        if self.raw_text is None:
            return
        self.processed_text=self.raw_text
        if Tuple:
            text=self.raw_text[1].lower()
        else:
            text=self.raw_text.lower()
        # highlight the figure refrence.
        text = re.sub(r'\\ref\{[^}]*fig[^}]*\}', '<Reference>', text)
        # remove the formating things.
        text = re.sub(r'\$.*?\$', '', text, flags=re.DOTALL)
        text = re.sub(r'\\[a-zA-Z]+\{.*?\}', '', text)
        text = re.sub(r'\\\w+\(.*?\)', '', text)
        text = re.sub(r"r'\\\w+\(.*?\)'", " ", text)
        text = re.sub(r'\\\S+', '', text)
        text = re.sub(r'-\n', ' ', text)
        text = re.sub(r'\n', ' ', text)
        text = re.sub(r'-', ' ', text)
        text = re.sub(r'\^', '', text)
        text = re.sub(r'[`_|{}~%&$\(\)\[\]§=,+]', ' ', text)
        text = re.sub(r"'", '', text)
        text = re.sub(r'"', '', text).replace("!", " .").replace(" ?", " .")
        # text= re.sub(" .", ". ", text) # gives the 

        text = re.sub(r" {2,}", " ", text)
        text = re.sub(r'fig\.\s+', '', text, flags=re.IGNORECASE).strip()

        return text





# need to remove some extra steps like preprocess only the intrested pages.
# combine the text which is preprocessed.
# 


'''
(important)
+ need to modify the text cleaning pipeline for quantum circuits embeddings.
+ need to mdoify the pipeline for  discription and captions.
    + highlight the label, caption etc etc
'''