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

# make sure the results folder exists, otherwise open(..., "w") will crash
Path("./results/with_audit").mkdir(parents=True, exist_ok=True)

from html_parser import HTML_Reader, map_symbols_to_equations
from tqdm import tqdm
from utils import get_sentences_around_label
from relations import get_relations

from extract_description import get_description


def _strip_backslash(s):
    """
    Remove every backslash from a string, for use as a clean JSON key.

    s        : a latex string, e.g. "\\mathcal{L}" or "T_{max}"
    returns  : the same string with all backslashes removed,
               e.g. "mathcal{L}", "T_{max}" (unchanged if it had none)
    """
    return s.replace("\\", "")


def get_meanings(clean_text, eq, audit=None, name_map=None):
    """
    Get the meaning of a symbol/equation, trying the main context first and
    the mention context as a fallback.

    Parameters
    ----------
    clean_text : str
        Full paper text with placeholders.
    eq : str
        The placeholder to describe, e.g. "SYM3".
    audit : dict, optional
        Flat audit dict (method_name -> list of messages). Forwarded to
        get_description so the meaning-extraction steps are recorded.
    name_map : dict, optional
        Placeholder -> latex map, forwarded so the audit shows latex.

    Returns
    -------
    str or None
        Extracted meaning, or None if nothing was found.
    """
    full_context = get_sentences_around_label(clean_text, eq)
    main_context = " ".join(full_context["main_context"])
    eq_disc = get_description(main_context, eq, audit=audit, name_map=name_map)
    if eq_disc is None:
        mention_context = " ".join(full_context["mention_context"])
        eq_disc = get_description(mention_context, eq, audit=audit, name_map=name_map)
    return eq_disc


import json
html_files = [ "2404.04958"]
for paper_id in tqdm(html_files):
    equation_meaning_dict = {}
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
        if eq not in equation_meaning_dict:
            equation_meaning_dict[index] = {}

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
        equation_meaning_dict[index]["equation"] = eqn_mapping[eq]["latex"]             # replace eq with eqn_mapping[eq]["latex"]
        equation_meaning_dict[index]["meaning"] = eq_meaning

    

        for sym in symbols:
            eq_meaning = get_meanings(clean_text, sym, audit=eq_audit, name_map=name_map)
            if "symbols" not in equation_meaning_dict[index]:
                equation_meaning_dict[index]["symbols"] = {}

            equation_meaning_dict[index]["symbols"][_strip_backslash(sym_mapping[sym])] = eq_meaning

        # relations to every other equation in the paper (simple v1 rules)
        relations = get_relations(eq, equaitons, eq_to_syms, sym_mapping,
                                  audit=eq_audit)
        equation_meaning_dict[index]["relations"] = relations

        # store this equation's complete audit trail in the output
        equation_meaning_dict[index]["audit-trail"] = eq_audit

    with open(f"./results/with_audit/with_audit_{paper_id}.json", "w") as file:
        json.dump(equation_meaning_dict, file)