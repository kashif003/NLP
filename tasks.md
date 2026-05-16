# Problem:

1. we have list of papers iDs, start in serial order. 
2. Get  the images of quantum circuit and discription:(PNG format)
    + page number of circuit.
    + Fig number
    + Quantum gates in that image
    + Disscussed algo's
    + discriptive text from paper, indicate the start and end of text (tuple).
    + list the number of quantum circuits found in the images if None write 0.
3. 


# TIPS:
1. can use discription from LATEX.
2. can use CV to extract the image from  PDF. (not sure)



# our approach

1. if latex avaliable.
    + Get all captions form latex.
    + process these captions.
        + check if the caption has quantum circuit. words in it.
        + mark this as positive sentence.
    + if quantum circuit:
        + confirm with CV horizontal lines or vertical lines.
    + extract the image. from latex or render the image with latex code.
    + get the tikz figures as well.
    + metadata:
        + Go to pdf approach for page number.
        + Go to pdf for checking  the figure number as well.
        + re pattern from the disription and caption.
        + check for the quantum problem S-bert will tell which quanutm problem alligns with the given caption.

2. if latex is not avalibale:
    + get caption:
        + using fig, Figure etc etc etc.
        + (need to check) take the caption.
        + take discription of the image.


# TODO:
1. look for main_file.tex -> /documentclass  (done)
i) In the all files look for /input or /include -> /figure or /figures. (done)
    + look for the keywords from gpt.
        + e.g: gate, quantikz etc 
    + if found re-render the image-> pdf->jpeg   (not done)

when images are in png or pdf:
ii) if fig found as pdf/png instead of latex code (.tex).
    + check the caption of figure and also the refrence text of the figure.
    + based on that if it is a quantum circuit  (semantic understanding)



# work for tommorow
1. need to extract all the .tar.gz files.
2. need to improve the condition for getting the quantum circuit.
3. work on more papers
3. need to work only pdf papper



ref of all figs -> clean also replace the label by "figure"-> decide if it is quantum circuit or not ->  goolge/gpt define quantum circuit -> embed -> compare the embeddings of refrence and goofle/gpt definition.