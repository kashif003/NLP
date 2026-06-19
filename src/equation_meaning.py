# this file will be used to get the meaning of the equation.



"""
1. take the sentence above the eqaution.
2. take the 3 sentences betweent the eqautions

"""



import re
def remove_symbols(text):
    text = re.sub(r"\[SYM\d+\]", "", text)
    return text


import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('stopwords')

def remove_stopwords(text):
    stop_words = set(stopwords.words('english'))
    
    word_tokens = word_tokenize(text)
    
    filtered_text = [word for word in word_tokens if word.lower() not in stop_words]
    
    # 4. Rejoin the filtered words back into a single string
    return " ".join(filtered_text)

def define_equation(eqn, context):
    context = clean_text(context)
    context = remove_symbols(context)

    return context



from html_parser import PaperTextExtractor
from utils import  get_sentences_around_label
if __name__ == "__main__":
    
    paper_id =  "2404.04958"  #"2506.19219"    

    extractor = PaperTextExtractor(paper_id)
    eqn = "[EQ(5)]"

    clean_text, eqn_mapping, sym_mapping = extractor.extract()

    context = get_sentences_around_label(clean_text, eqn, window=0)

    print("Main_context:")
    print(context["main_context"])
    print("mention_context:")
    print(context["mention_context"])
    print()
    print("[REMOVING STOP WORDS...........]")
    print(remove_stopwords("".join(context["main_context"])))


#-- 