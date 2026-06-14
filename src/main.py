import tqdm
import time
import os
from utils import paper_ID_extractor
from utils import download_paper
paper_list =  paper_ID_extractor("./paper_list_11.txt", 100 )

"""
1. 2404.12603
2. 2412.06623
list:
    [2412.06623, 2404.12603]

"""
# reading the latex files and storing them as dict-> key: name of file, value: list of tuples: (name of .tex file, content)
from read_latex import read_all_tex_files
files = {}
for id in [2412.06623, 2404.12603]:
    path = f"./data/latex_source/{id}"
    files[f"{id}"] = read_all_tex_files(path)



# getting the main.tex file.
first_file = files["2412.06623"]
second_file = files["2404.12603"]
from read_latex import (
    find_main_tex,
    get_included_files,
    find_quantum_circuit_file,
    extract_file_reference_context,
    extract_quantikz_code,
    create_standalone_circuit_pdf,
    get_all_figure_references,
    get_image_filenames_from_figure,
    classify_quantum_figures,
)
for k, v in files.items():
    main_file = find_main_tex(v)

    if main_file:
        filename, content = main_file
        # Get all included file names
        included_files = get_included_files(content)    # list
        if not included_files:
            print("No reffered files found in ", k)
            
            # Extract figure information instead
            print(f"\nExtracting figure information from paper {k}...")
            all_figures = get_all_figure_references(v)
            
            if all_figures:
                print(f"Found {len(all_figures)} figures:\n")
                for fig_label, fig_info in all_figures.items():
                    print(f"  Figure: {fig_label}")
                    print(f"  Caption: {fig_info['caption']}")
                    print(f"  Source file: {fig_info['source_file']}")
                    
                    if fig_info['references']:
                        print(f"  References ({len(fig_info['references'])}):")
                        for idx, ref in enumerate(fig_info['references'], 1):
                            print(f"    Reference {idx}:")
                            print(f"    Context passage:\n{ref['passage']}\n")
                    else:
                        print(f"  No references found in text\n")
                # Classify figures and copy images for quantum-circuit figures
                classification = classify_quantum_figures(all_figures, threshold=3)
                images_dir = "./images"
                os.makedirs(images_dir, exist_ok=True)

                base_dir = f"./data/latex_source/{k}"
                for fig_label, info in all_figures.items():
                    clf = classification.get(fig_label, {})
                    if clf.get('is_quantum'):
                        print(f"\nFigure {fig_label} classified as quantum (score={clf.get('score')}). Copying images if present...")
                        image_names = get_image_filenames_from_figure(info.get('figure_code',''))
                        if not image_names:
                            print(f"  No \includegraphics found in {info.get('source_file')}")
                            continue

                        for img in image_names:
                            # try exact path first
                            src_path = os.path.join(base_dir, img)
                            if not os.path.exists(src_path):
                                # search by basename recursively under base_dir
                                basename = os.path.basename(img)
                                found = None
                                for root, dirs, files in os.walk(base_dir):
                                    if basename in files:
                                        found = os.path.join(root, basename)
                                        break
                                if found:
                                    src_path = found
                                else:
                                    print(f"  Image not found for {img}")
                                    continue

                            dst_name = f"{k}_{fig_label}_{os.path.basename(src_path)}"
                            dst_path = os.path.join(images_dir, dst_name)
                            try:
                                import shutil
                                shutil.copy(src_path, dst_path)
                                print(f"  Copied image to {dst_path}")
                            except Exception as e:
                                print(f"  Failed to copy image {src_path}: {e}")
        else:
            # Find files with quantum circuits
            quantum_files = find_quantum_circuit_file(v) # list of files which have quantizk in them with content
            if quantum_files:
                print(f"\nPaper {k}:")
                # For each quantum circuit file, extract where it's referenced
                for circuit_file, circuit_content in quantum_files:
                    print(f"\n  --- Processing {circuit_file} ---")
                    references = extract_file_reference_context(v, circuit_file)
                    
                    if references:
                        for ref in references:
                            print(f"    Source file: {ref['source_file']}")
                            print(f"    Reference line: {ref['full_line']}")
                    
                    # Extract quantikz code from the circuit file
                    quantikz_codes = extract_quantikz_code(circuit_content) # list of k,
                    
                    if quantikz_codes:
                        print(f"    Found {len(quantikz_codes)} quantum circuit(s)")
                        
                        # Create images folder if it doesn't exist
                        images_dir = "./images"
                        
                        # Save each circuit as a PDF
                        for idx, circuit_code in enumerate(quantikz_codes):
                            circuit_name = f"{k}_{circuit_file.replace('.tex', '')}_circuit_{idx+1}"
                            output_pdf = os.path.join(images_dir, f"{circuit_name}.pdf")
                            
                            print(f"    Compiling circuit {idx+1} to PDF: {output_pdf}")
                            success = create_standalone_circuit_pdf(circuit_code, output_pdf, circuit_name)
                            
                            if success:
                                print(f"    ✓ Successfully saved: {output_pdf}")
                            else:
                                print(f"    ✗ Failed to compile circuit {idx+1}")
                    else:
                        print(f"    No quantikz code found in {circuit_file}")
            else:
                print(f"\nPaper {k}: No quantum circuit files found")

# checking which files are mentionedReference 4:
