# NLP: Quantum Circuit Diagram Mining Pipeline

An automated pipeline that mines **arXiv papers** (quant-ph category) to build a labeled dataset of **quantum circuit diagram images**, each paired with its caption, surrounding description, and extracted metadata (algorithm name, gate types, qubit count). It combines PDF/LaTeX parsing, SciBERT sentence embeddings, and a question-answering model to automatically decide which figures in a paper are quantum circuit diagrams (as opposed to plots, charts, or other figures) and what those circuits represent.

## What it does, end to end

1. **Reads a list of arXiv paper IDs** (`paper_list_11.txt` — ~26,900 IDs) to process.
2. **Downloads each paper** — both the PDF and, when available, the LaTeX source — from arXiv, caching them locally (skips re-downloading if already cached, and only proceeds for papers in the `quant-ph` category).
3. **Extracts every figure from the PDF** using `docling`, matching each image to its caption either via Docling's direct caption link or by spatial proximity (closest caption on the same page).
4. **Extracts caption + surrounding description text** for every figure mention:
   - From the **PDF** (`reader.py::PdfReader`) using PyMuPDF (`fitz`) + spaCy sentence segmentation, scanning for "Figure/Fig N" patterns.
   - From the **LaTeX source** (`reader.py::LatexReader`), when available, by parsing `\begin{figure}...\end{figure}` / `\caption{}` / `\label{}` / `\ref{}` blocks directly — LaTeX is preferred over PDF text extraction when both exist, since it's cleaner.
5. **Cleans the extracted text** (`text_preprocessor.py`) — two dedicated preprocessors strip LaTeX formatting commands, comments, bibliographies, author/affiliation blocks, tables, equations, citations/references, and normalize brackets/quotes/whitespace, readying the text for embedding.
6. **Scores each caption/description against a "quantum circuit" anchor concept** (`utils.py::compare_embedding`, built via `anchor_embeddings.py`):
   - A small set of hand-written seed sentences describing what a quantum circuit is are embedded with **SciBERT** (`allenai/scibert_scivocab_uncased`).
   - Their centroid becomes an "anchor" embedding; candidate sentences from a positive example pool that score above a similarity threshold are folded in to enrich the anchor (saved to `Embeddings/positive_anchor.pt`).
   - Each figure's caption/description is embedded the same way and compared via cosine similarity against this anchor, with keyword boosting (e.g. "quantum circuit", "circuit diagram") and hard-negative filtering (e.g. "plot", "flowchart", "heatmap", "bar chart") to reject non-circuit figures.
7. **Saves matches** — images that pass the similarity threshold (default 0.94 in `main.py`) are copied into `quantum_circuit_images/`, with metadata (arXiv ID, figure number, page number, description, position in text) recorded per image.
8. **Extracts structured metadata from the description** (`utils.py::extract_metadata_final`) using regex gate-name matching (CNOT, Hadamard, Toffoli, Pauli-X/Z, rotation gates, SWAP, measurement) plus a **question-answering model** (`deepset/roberta-base-squad2`) to infer the algorithm/protocol name and qubit count from the surrounding text.
9. **Writes final outputs**:
   - `output_enriched.json` — per-image metadata (arXiv ID, figure/page number, description, algorithm, gates, qubits).
   - `images_per_paper.csv` — a summary of how many qualifying images were found per paper.

## Repository structure

```
NLP/
├── paper_list_11.txt          # ~26,900 arXiv IDs (format: "arXiv:XXXX.XXXXX") to process
├── .gitignore                 # ignores data/ and venv/
└── src/
    ├── main.py                 # Orchestrates the full pipeline end-to-end
    ├── utils.py                 # Paper download, embedding comparison, metadata extraction, QA-based gate/algorithm extraction
    ├── reader.py                # PdfReader (PyMuPDF + spaCy) and LatexReader (regex-based) for extracting captions/descriptions
    ├── text_preprocessor.py     # Latex_preprocessor and Pdf_Preprocess text-cleaning classes
    ├── image_extractor.py       # Docling-based figure/caption extraction from PDFs
    ├── embedding.py             # SciBERT-based sentence encoding + anchor centroid computation
    └── anchor_embeddings.py     # Builds/refines the "quantum circuit" anchor embedding from seed + positive examples
```

**Note:** the code references some paths/folders not present in the repo itself (created at runtime), including `cache/` (downloaded PDFs/LaTeX sources), `Embeddings/` (anchor embeddings + positive example text), and `quantum_circuit_images/` (final output images) — these are populated when the pipeline runs, and `data/` is already gitignored for this purpose.

## Requirements

Not currently pinned in a `requirements.txt` (see cleanup notes below), but based on the imports across `src/`, you'll need:

- `torch`, `transformers`, `adapters` (AutoAdapterModel)
- `arxiv` (arXiv API client for downloading papers)
- `docling`, `docling-core` (PDF figure/caption extraction)
- `PyMuPDF` (`fitz`) (PDF text extraction)
- `spacy` + the `en_core_web_sm` model (`python -m spacy download en_core_web_sm`)
- `nltk` (with the `punkt` and `stopwords` resources)
- `gensim`
- `scikit-learn` (TF-IDF/CountVectorizer keyword extraction)
- `pandas`
- `scipy`

## Getting started

1. **Install dependencies** (once a `requirements.txt` exists — see cleanup notes):
   ```bash
   pip install torch transformers adapters arxiv docling docling-core pymupdf spacy nltk gensim scikit-learn pandas scipy
   python -m spacy download en_core_web_sm
   ```
2. **Build the anchor embedding** (first run only, or to refresh it):
   ```bash
   cd src
   python anchor_embeddings.py
   ```
   This expects a `Embeddings/text_list/positive_list.txt` file of example positive sentences to exist beforehand, and writes `Embeddings/positive_anchor.pt`.
3. **Run the full pipeline**:
   ```bash
   python main.py
   ```
   This works through `paper_list_11.txt`, downloading papers, extracting and classifying figures, and stopping once 250 matching images have been collected (a hardcoded limit in `main.py`). Results land in `quantum_circuit_images/`, `output_enriched.json`, and `images_per_paper.csv`.

## Cleanup notes

Worth addressing as part of a broader cleanup pass:
- **No `requirements.txt`** — dependencies must currently be inferred from imports; adding one (with pinned versions) would make this reproducible.
- **Hardcoded paths and constants** scattered through `main.py` (e.g. `Embeddings/positive_anchor.pt`, the 250-image cap, the 0.94 similarity threshold) would be cleaner as config/CLI arguments.
- **`main.py` references `shutil`** without importing it directly (relies on the `from utils import *` wildcard import picking it up transitively) — fragile; should be imported explicitly.
- Several **typos** in docstrings/variable names throughout (`discription` → `description`, `enchor` → `anchor`, `dodwnloaded` → `downloaded`, `seperate_...` → `separate_...`, `fucntion` → `function`) — harmless but worth a pass for polish.
- `image_extractor.py`'s duplicate-detection logic (`unique_metadata`) and the `Pdf_Preprocess`/`Latex_preprocessor` classes have some overlapping responsibility with `reader.py` — could likely be consolidated.
- Wildcard imports (`from utils import *`, `from reader import *`, etc.) throughout make it hard to trace where a given function actually comes from — explicit imports would help readability.
- No tests currently exist for the text-cleaning regex pipelines, which are intricate enough (especially `text_preprocessor.py`) to benefit from unit tests given a few sample LaTeX/PDF snippets.
