# this file will be used to get the meaning of the equation.



"""
1. take the sentence above the eqaution.
2. take the 3 sentences betweent the eqautions

"""



import re
def remove_symbols(text):
    text = re.sub(r"\[SYM\d+\]", "", text)
    return text

def clean(text):
    text = re.sub(r"\[\s*\d+(?:\s*[,\-–]\s*\d+)*\s*\]", "", text)

    # 4) Normalize whitespace and remove spaces before punctuation
    text = text.replace("\n", " ")
    text = re.sub(r"\s+([,.;:!?)])", r"\1", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def define_equation(eqn, context):
    context = clean_text(context)
    context = remove_symbols(context)

    return context



from html_parser import PaperTextExtractor
from utils import  get_sentences_around_label
if __name__ == "__main__":
    
    paper_id =  "2404.04958"  #"2506.19219"    

    extractor = PaperTextExtractor(paper_id)
    eqn = "EQN1"

    clean_text, eqn_mapping, sym_mapping = extractor.extract()

    context = get_sentences_around_label(clean_text, eqn, window=1)

    print("Main_context:")
    print(context["main_context"])
    print("mention_context:")
    print(context["mention_context"])
    print()
    print("[POS TAGGING...........]")

    import nltk
    from nltk.tokenize import word_tokenize
    from nltk import pos_tag
    nltk.download('punkt_tab')
    nltk.download('averaged_perceptron_tagger_eng')

    text = clean(" ".join( context["main_context"]))
    words = word_tokenize(text)

    pos_tags = pos_tag(words)
    print("[ORIGINAL]:", text)

    print("\nPoS Tagging Result:")
    for word, pos_tag in pos_tags:
        print(f"{word}: {pos_tag}")