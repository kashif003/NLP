# combined approchs:
1) Ammar:
    + purely on latex.
    + image rendering
    + TF-IDF for caption matching.
    + re for filtering (gates,)
    + S-bert: 51 Q-problems defined and embedded and matched the discription with each problem.
    + 150+ quantum terms to find relevant captions with S bert reranking with tfidf (w1*tfidf+ w2*s-bert score).
    
2) Darshan:
    + latex + pdf, priority -> latex.
    + from latex took caption, from pdf took special proximityof fig or figure.
    + embedded each fig caption text embedded set of refrance phrases about quantum circuits and gated level structure and compute cosine sim.
    + used classical cv to detect horizonatal line segments.
    + checked wire counts etc.
    + if no circut feature then the circuit is rejected.

3) Tanay:
    + used pdf only.
    + extracted text blocks first and located fig using re pattern.
    + check where the fig is and cropped the pdf 300 DPI to get the fig. (downside: we can have the text also in the image or image is cropped).
    + checked where the fig number appears and extracted 5 sentences before and after the text.
    + collected 1k paper from the last papers.
        + used 1k for training 1k for testing.
        + used caption of 1k training sets, got positive and negative words using tfidf and similar words in  both are excluded.
        + made a fucntion using the positive words to classify the caption if it is a quantum circuit or not.
    + applied re pattern  from gpt to get the gates.
    + algos:
        + direct re matching.
        + noun checking: gets 3-4 nouns phrases and combined them in short discriptive summary.



