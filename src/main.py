from html_reader import HTMLReader
import time
# paper_ids = [
# "2401.13724",
# "2408.07125",
# "2508.05295",
# "2510.12545",
# "2410.08937"
# ]

# start = time.time()
# for paper_id in paper_ids:
#     print("paper ID:",paper_id )
#     reader = HTMLReader(paper_id)

#     content = reader.file_content
#     equattions= reader.finding_all_equations()
#     #symbols = reader.get_all_paragraph_symbols()
#     print(equattions)
#     # full_context = reader.get_symbol_full_context()
#     # for symbol, data in full_context.items():
#     #     print(f"\nSymbol: {symbol}")
#     #     print(f"In equations: {data['equations']}")
#     #     print(f"All contexts:")
#     #     for i, ctx in enumerate(data['contexts']):
#     #         print(f"  [{i+1}] {ctx}")
# print("done")
# end  = time.time()
# print(end-start)



from pathlib import Path

# Set the directory to the current folder ('.' means current directory)
current_dir = Path('./data/html_source')
html_files = []
# Loop through and print everything in the directory
for item in current_dir.iterdir():
    if item.is_file():  # This ensures we only print files, not folders
        file_name = item.name
        html_files.append(file_name[:-5])

from html_parser import PaperTextExtractor, map_symbols_to_equations
from tqdm import tqdm
from utils import get_sentences_around_label 

from test import get_meaning


def get_meanings(clean_text,eq):
        full_context = get_sentences_around_label(clean_text,eq)
        main_context = " ".join(full_context["main_context"])
        eq_disc = get_meaning(main_context, eq)
        if eq_disc is None:
            mention_context = " ".join(full_context["mention_context"])
            eq_disc = get_meaning(mention_context, eq)
        return eq_disc
import json
for paper_id in tqdm(html_files):
    equation_meaning_dict = {}
    extractor = PaperTextExtractor(paper_id)
    clean_text, eqn_mapping, sym_mapping = extractor.extract()
    eq_to_syms = map_symbols_to_equations(eqn_mapping, sym_mapping)
    equaitons = list(eqn_mapping.keys())

    print("[INFO] GETTING Meaning of equations......")
    for i,eq in enumerate(equaitons):
        index = i+1
        if eq not in equation_meaning_dict:
            equation_meaning_dict[index] = {}
        
        eq_meaning = get_meaning(clean_text, eq)
        equation_meaning_dict[index]["equation"] = eq               #TODO replace eq with eq_mapping[eq]["latex"]
        equation_meaning_dict[index]["meaning"] = eq_meaning

        symbols = eq_to_syms[eq]
        for sym in symbols:
            eq_meaning = get_meanings(clean_text, sym)
            if "symbols" not in equation_meaning_dict[index]:
                equation_meaning_dict[index]["symbols"] = {}

            equation_meaning_dict[index]["symbols"][sym] = eq_meaning        #TODO replace sym with sym_mapping[sym] and get real names without \
    
    with open(f"./results/{paper_id}.json", "w") as file:
        json.dump(equation_meaning_dict, file)
    break

