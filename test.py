from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import numpy as np
import nltk

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

def extract_top_keywords(positive_file, top_n=50):
    # 1. Load Data
    try:
        with open(positive_file, "r", encoding='utf-8') as f:
            pos_text = f.readlines()
    except FileNotFoundError:
        print(f"Error: File '{positive_file}' not found.")
        return []
    vectorizer = CountVectorizer(
        ngram_range=(1, 3), 
        stop_words='english',
        min_df=2 
    )
    try:
        X = vectorizer.fit_transform(pos_text)
    except ValueError:
        print("Error: Input text is too small or contains only stop words.")
        return []
    counts = np.asarray(X.sum(axis=0)).flatten()
    vocab = vectorizer.get_feature_names_out()
    ranked_indices = np.argsort(counts)[::-1]
    print("boost_keywords = [")
    for i in range(min(top_n, len(ranked_indices))):
        idx = ranked_indices[i]
        term = vocab[idx]
        word_count = len(term.split())
        score = 0.05
        if word_count == 1:
            score = 0.05
        elif "quantum" in term or "circuit" in term:
            score = 0.10
        else:
            score = 0.15
        strong_single_words = ["qiskit", "cirq", "ansatz", "unitary", "qubit"]
        if term in strong_single_words:
            score = 0.15
        print(f"    (\"{term}\", {score}),")
    print("]")

