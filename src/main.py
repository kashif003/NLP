
from pathlib import Path
from utils import paper_ID_extractor, download_html, strip_backslash, get_meanings
paper_list = paper_ID_extractor("./paper_list_12.txt")





# Set the directory to the current folder ('.' means current directory)
current_dir = Path('./data/html_source')
html_files = []
# Loop through and print everything in the directory
for item in current_dir.iterdir():
    if item.is_file():  # This ensures we only print files, not folders
        file_name = item.name
        html_files.append(file_name[:-5])

# make sure the results folder exists, otherwise open(..., "w") will crash
Path("./results/with_audit").mkdir(parents=True, exist_ok=True)

from html_parser import HTML_Reader, map_symbols_to_equations
from tqdm import tqdm
from utils import get_sentences_around_label
from relations import get_relations

from extract_description import get_description



from tqdm import tqdm
dataset = {}
for paper_id in tqdm(paper_list[:5]):
    print("[INFO] Downloading the paper:", paper_id)
    downloaded =download_html(paper_id)
    if not downloaded:
        print("[IMPORTANT] Unable to download the paper:", paper_id)
        dataset[f"arXiv:{paper_id}"] = "Unable to download the paper"
        continue
    
    dataset[f"arXiv:{paper_id}"] = {}
    extractor = HTML_Reader(paper_id)
    clean_text, eqn_mapping, sym_mapping = extractor.extract()
    eq_to_syms = map_symbols_to_equations(eqn_mapping, sym_mapping)
    equaitons = list(eqn_mapping.keys())

    # placeholder -> latex lookup, so the audit shows real latex (T_{max})
    # instead of placeholders (SYM26 / EQN1). Token search still uses placeholders.
    name_map = dict(sym_mapping)
    name_map.update({e: d["latex"] for e, d in eqn_mapping.items()})

    print("[INFO] GETTING Meaning of equations......")
    for i, eq in enumerate(equaitons):
        index = i + 1
        if eq not in dataset[f"arXiv:{paper_id}"]:
            dataset[f"arXiv:{paper_id}"][index] = {}

        # fresh audit trail for THIS equation only (flat: method -> messages)
        eq_audit = {}

        # audit: record the equation that was extracted (latex from html_parser)
        latex = eqn_mapping[eq]["latex"]
        eq_audit["extract_equations_method"] = latex          #TODO 1. change

        symbols = eq_to_syms[eq]  #TODO 2. first check which symbols are in equation.

        # audit: record which symbols were matched into this equation
        eq_audit["map_symbols_to_equations"] = {latex: [sym_mapping[s] for s in symbols]}

        # eq_audit.setdefault("map_symbols", []).append(
        #     f"map_symbols_to_equations: symbols={[sym_mapping[s] for s in symbols]}" #TODO 3.
        # )


        
                    
        eq_meaning = get_description(clean_text, eq, audit=eq_audit, name_map=name_map)
        dataset[f"arXiv:{paper_id}"][index]["equation"] = eqn_mapping[eq]["latex"]             # replace eq with eqn_mapping[eq]["latex"]
        dataset[f"arXiv:{paper_id}"][index]["meaning"] = eq_meaning

    

        for sym in symbols:
            eq_meaning = get_meanings(clean_text, sym, audit=eq_audit, name_map=name_map)
            if "symbols" not in dataset[f"arXiv:{paper_id}"][index]:
                dataset[f"arXiv:{paper_id}"][index]["symbols"] = {}

            dataset[f"arXiv:{paper_id}"][index]["symbols"][strip_backslash(sym_mapping[sym])] = eq_meaning

        # relations to every other equation in the paper (simple v1 rules)
        relations = get_relations(eq, equaitons, eq_to_syms, sym_mapping,
                                  audit=eq_audit)
        dataset[f"arXiv:{paper_id}"][index]["relations"] = relations

        # store this equation's complete audit trail in the output
        dataset[f"arXiv:{paper_id}"][index]["audit-trail"] = eq_audit

import json
with open(f"./results/with_audit/dataset.json", "w") as file:
    json.dump(dataset, file)