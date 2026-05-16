import os
import re
import subprocess
import tempfile
import shutil


def extract_figures_with_captions(tex_files_tuple):
    """
    Extracts all figures and their captions from .tex files.
    Returns a dictionary with figure labels as keys and caption info as values.
    
    Args:
        tex_files_tuple (tuple): Tuple of tuples containing (filename, content)
        
    Returns:
        dict: Dictionary with structure:
              {
                  'fig:label1': {
                      'caption': 'Figure caption text',
                      'source_file': 'filename.tex',
                      'figure_code': '...full figure environment...'
                  },
                  ...
              }
    """
    figures = {}
    
    for filename, content in tex_files_tuple:
        # Pattern to match \begin{figure}...\end{figure} with caption and label
        figure_pattern = r'\\begin\{figure\}.*?\\end\{figure\}'
        figure_matches = re.finditer(figure_pattern, content, re.DOTALL)
        
        for figure_match in figure_matches:
            figure_code = figure_match.group(0)
            
            # Extract caption
            caption_pattern = r'\\caption\{([^}]+)\}'
            caption_match = re.search(caption_pattern, figure_code)
            caption = caption_match.group(1) if caption_match else "No caption"
            
            # Extract label
            label_pattern = r'\\label\{([^}]+)\}'
            label_match = re.search(label_pattern, figure_code)
            label = label_match.group(1) if label_match else f"unlabeled_{len(figures)}"
            
            figures[label] = {
                'caption': caption,
                'source_file': filename,
                'figure_code': figure_code
            }
    
    return figures


def extract_figure_references(content, figure_label):
    """
    Finds where a figure is referenced (\ref{fig:label}) in the content and extracts context.
    
    Args:
        content (str): Content to search in
        figure_label (str): The figure label to search for (e.g., 'fig:diffuser')
        
    Returns:
        list: List of dicts with keys:
              - 'passage': the surrounding passage where figure is mentioned
              - 'reference_line': the full line with \ref{...}
    """
    references = []
    
    # Pattern to match \ref{figure_label}
    ref_pattern = rf'\\ref\{{{figure_label}\}}'
    ref_matches = re.finditer(ref_pattern, content)
    
    for ref_match in ref_matches:
        # Extract the entire paragraph containing the reference
        # Find the start of the paragraph (search backwards for double newline)
        para_start = content.rfind('\n\n', 0, ref_match.start())
        if para_start == -1:
            para_start = 0
        else:
            para_start += 2  # Skip the \n\n
        
        # Find the end of the paragraph (search forwards for double newline)
        para_end = content.find('\n\n', ref_match.end())
        if para_end == -1:
            para_end = len(content)
        
        passage = content[para_start:para_end].strip()
        passage = passage
        reference_line = ref_match.group(0)
        
        references.append({
            'passage': passage,
            'reference_line': reference_line
        })
    
    return references


def get_all_figure_references(tex_files_tuple):
    """
    Creates a complete dictionary of all figures with their captions and reference passages.
    
    Args:
        tex_files_tuple (tuple): Tuple of tuples containing (filename, content)
        
    Returns:
        dict: Dictionary with figure labels as keys and detailed info as values:
              {
                  'fig:label': {
                      'caption': 'Caption text',
                      'source_file': 'file.tex',
                      'references': [
                          {'passage': '...context...', 'reference_line': '\\ref{fig:label}'},
                          ...
                      ]
                  }
              }
    """
    # Get all figures with captions
    figures = extract_figures_with_captions(tex_files_tuple)
    
    # Combine all content for reference searching
    all_content = "\n".join([content for _, content in tex_files_tuple])
    
    # For each figure, find where it's referenced
    for fig_label in figures:
        references = extract_figure_references(all_content, fig_label)
        figures[fig_label]['references'] = references   #TODO add the cleaning fucntion over here to clean the pdf.
    
    return figures


def get_image_filenames_from_figure(figure_code):
    """
    Extract filenames used in \includegraphics commands inside a figure.

    Args:
        figure_code (str): full figure environment text

    Returns:
        list: list of image file paths or basenames as written in the LaTeX
    """
    pattern = r'\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}'
    matches = re.findall(pattern, figure_code)
    return matches


def classify_quantum_figures(figures_dict, threshold=3): # TODO  need to change this condition
    """
    Classify figures as quantum-circuit figures using heuristic rules.

    Args:
        figures_dict (dict): output of get_all_figure_references()
        threshold (int): score threshold above which figure is marked quantum

    Returns:
        dict: mapping figure_label -> {'score': int, 'is_quantum': bool}
    """
    # Strong signals: explicit quantikz environment or package, gate/control commands
    results = {}
    for label, info in figures_dict.items():
        text_parts = [info.get('caption', '') or '']
        for ref in info.get('references', []):
            text_parts.append(ref.get('passage', '') or '')
        text_parts.append(info.get('figure_code', '') or '')
        text = "\n".join(text_parts)

        # explicit quantikz usage
        has_explicit = bool(re.search(r'\\begin\{quantikz\}', text, re.I) or re.search(r'\\usepackage\{quantikz\}', text, re.I))
        # gate/control commands
        has_gate_cmd = bool(re.search(r'\\gate\{|\\ctrl\{|\\targ\{|\\meter', text, re.I))
        # check image filenames for hints like 'circuit' or 'quantum'
        image_names = get_image_filenames_from_figure(info.get('figure_code', ''))
        has_image_hint = any(re.search(r'circuit|quantum', img, re.I) for img in image_names)

        # Conservative decision: mark as quantum only if strong signals present
        is_q = has_explicit or has_gate_cmd or has_image_hint

        # Compute a diagnostic score for debugging/tuning
        score = 0
        if has_explicit:
            score += 10
        if has_gate_cmd:
            score += 6
        if has_image_hint:
            score += 4

        results[label] = {'score': score, 'is_quantum': bool(is_q)}

    return results


def read_all_tex_files(path):
    """
    Reads all .tex files from a given directory and returns them as a tuple of tuples.
    
    Args:
        path (str): Directory path to search for .tex files
        
    Returns:
        tuple: A tuple of tuples, each containing (filename, file_content)
               Example: (('file1.tex', 'content1'), ('file2.tex', 'content2'))
    """
    tex_files = []
    
    # Walk through the directory and subdirectories
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith('.tex'):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    tex_files.append((file, content))
                except Exception as e:
                    print(f"Error reading file {file_path}: {e}")
    
    return tuple(tex_files)


def find_main_tex(tex_files_tuple):
    """
    Finds the main .tex file from a tuple of (filename, content) tuples.
    The main file is identified by the presence of \documentclass command.
    
    Args:
        tex_files_tuple (tuple): Tuple of tuples containing (filename, content)
        
    Returns:
        tuple: (filename, content) of the main file, or None if not found
    """
    for (filename, content) in tex_files_tuple:
        if r'\documentclass' in content:
            return (filename, content)
    
    return None


def get_included_files(content):
    """
    Extracts all included file names from a .tex file content.
    Finds all \input{filename} and \include{filename} commands.
    
    Args:
        content (str): Content of the main .tex file
        
    Returns:
        list: List of filenames referenced by \input and \include commands
              Example: ['abstract', 'introduction', 'chapter1']
    """
    # Pattern to match \input{filename} or \include{filename}
    pattern = r'\\(?:input|include)\s*\{([^}]+)\}'
    
    matches = re.findall(pattern, content)
    
    return matches


def find_quantum_circuit_file(tex_files_tuple):
    """
    Finds the .tex file(s) that contain quantum circuit drawings using quantikz package.
    
    Args:
        tex_files_tuple (tuple): Tuple of tuples containing (filename, content)
        
    Returns:
        list: List of tuples (filename, content) that contain quantikz package
              Example: [('diffuser.tex', 'content...'), ('circuit.tex', 'content...')]
    """
    quantum_files = []
    
    for filename, content in tex_files_tuple:
        # Check for quantikz package usage
        if r'\usepackage{quantikz}' in content or r'\begin{quantikz}' in content:
            quantum_files.append((filename, content))
    
    return quantum_files


def extract_file_reference_context(tex_files_tuple, circuit_filename):
    """
    Finds where a quantum circuit file is referenced/included and extracts the surrounding context.
    
    Args:
        tex_files_tuple (tuple): Tuple of tuples containing (filename, content)
        circuit_filename (str): Name of the circuit file to search for (e.g., 'diffuser.tex' or 'diffuser')
        
    Returns:
        list: List of dicts with keys:
              - 'source_file': the file that references the circuit
              - 'context': the cleaned surrounding paragraph where it's mentioned
              - 'full_line': the full line containing the reference
              Example: [{'source_file': 'main.tex', 'full_line': '\\input{diffuser}', 'context': '...cleaned passage...'}]
    """
    # Remove .tex extension if present for matching
    circuit_name = circuit_filename.replace('.tex', '')
    
    results = []
    
    for source_filename, content in tex_files_tuple:
        # Look for \input{circuit_name} or \include{circuit_name}
        pattern = rf'\\(?:input|include)\s*\{{{circuit_name}(?:\.tex)?\}}'
        
        matches = re.finditer(pattern, content)
        
        for match in matches:
            # Extract the entire paragraph containing the reference
            # Find the start of the paragraph (search backwards for double newline)
            para_start = content.rfind('\n\n', 0, match.start())
            if para_start == -1:
                para_start = 0
            else:
                para_start += 2  # Skip the \n\n
            
            # Find the end of the paragraph (search forwards for double newline)
            para_end = content.find('\n\n', match.end())
            if para_end == -1:
                para_end = len(content)
            
            context = content[para_start:para_end].strip()
            context = context
            
            full_line = match.group(0)
            
            results.append({
                'source_file': source_filename,
                'full_line': full_line,
                'context': context
            })
    
    return results


def extract_quantikz_code(content):
    """
    Extracts all quantikz environment blocks from LaTeX content.
    
    Args:
        content (str): Content of a .tex file
        
    Returns:
        list: List of strings, each containing a \begin{quantikz}...\end{quantikz} block
    """
    pattern = r'\\begin\{quantikz\}.*?\\end\{quantikz\}'
    matches = re.findall(pattern, content, re.DOTALL)
    return matches


def create_standalone_circuit_pdf(circuit_code, output_pdf_path, circuit_name="circuit"):
    """
    Creates a standalone LaTeX document from quantum circuit code and compiles it to PDF.
    
    Args:
        circuit_code (str): The quantikz code block
        output_pdf_path (str): Path where to save the PDF
        circuit_name (str): Name for the circuit (used for temp files)
        
    Returns:
        bool: True if successful, False otherwise
    """
    # Create a minimal LaTeX template with quantikz circuit
    latex_template = f"""\\documentclass{{article}}
\\usepackage{{quantikz}}
\\usepackage{{geometry}}
\\geometry{{margin=1cm}}

\\begin{{document}}

\\thispagestyle{{empty}}

{circuit_code}

\\end{{document}}
"""
    
    try:
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_pdf_path), exist_ok=True)
        
        # Use temporary directory for compilation
        with tempfile.TemporaryDirectory() as temp_dir:
            tex_file = os.path.join(temp_dir, f"{circuit_name}.tex")
            
            # Write LaTeX file
            with open(tex_file, 'w', encoding='utf-8') as f:
                f.write(latex_template)
            
            # Compile to PDF using pdflatex
            result = subprocess.run(
                ['pdflatex', '-interaction=nonstopmode', '-output-directory', temp_dir, tex_file],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode != 0:
                print(f"pdflatex compilation failed:\n{result.stdout}\n{result.stderr}")
                return False
            
            # Copy PDF to output location
            temp_pdf = os.path.join(temp_dir, f"{circuit_name}.pdf")
            if os.path.exists(temp_pdf):
                import shutil
                shutil.copy(temp_pdf, output_pdf_path)
                return True
            else:
                print(f"PDF file not found at {temp_pdf}")
                return False
                
    except subprocess.TimeoutExpired:
        print(f"pdflatex compilation timed out for {circuit_name}")
        return False
    except Exception as e:
        print(f"Error compiling circuit to PDF: {e}")
        return False









# tes_files = read_all_tex_files("./data/latex_source/2404.12603")
# print(tes_files[0])