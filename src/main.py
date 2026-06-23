from pathlib import Path
from utils import paper_ID_extractor, download_html, strip_backslash, get_meanings
from html_parser import HTML_Reader, map_symbols_to_equations
from tqdm import tqdm
from utils import get_sentences_around_label
from relations import get_relations
from extract_description import get_description, extract_lhs
import json

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

# We removed "dataset = {}" from here so papers stay separated!



for paper_id in tqdm(paper_list[:10]):
    # Initialize a dedicated dataset dictionary for THIS paper only
    paper_dataset = {}
    
    print("[INFO] Downloading the paper:", paper_id)
    downloaded = download_html(paper_id)
    if not downloaded:
        print("[IMPORTANT] Unable to download the paper:", paper_id)
        paper_dataset[f"arXiv:{paper_id}"] = "Unable to download the paper"
        
        # Even if it fails, we save the failure status to its own file
        with open(f"./results/with_audit/{paper_id}.json", "w") as file:
            json.dump(paper_dataset, file, indent=4)
        continue
    
    paper_dataset[f"arXiv:{paper_id}"] = {}
    extractor = HTML_Reader(paper_id)
    clean_text, eqn_mapping, sym_mapping = extractor.extract()
    eq_to_syms = map_symbols_to_equations(eqn_mapping, sym_mapping)
    equaitons = list(eqn_mapping.keys())

    # placeholder -> latex lookup, so the audit shows real latex (T_{max})
    # instead of placeholders (SYM26 / EQN1). Token search still uses placeholders.
    name_map = dict(sym_mapping)
    name_map.update({e: d["latex"] for e, d in eqn_mapping.items()})

    print("[INFO] GETTING Meaning of equations......")

    # eq_context[eq] = equation meaning + all its symbol meanings (NOT raw text).
    # audits[eq]     = that equation's audit dict, carried into pass 2.
    eq_context = {}
    audits = {}

    # for the "same LHS -> same meaning" pass:
    eq_lhs = {}            # eq -> its LHS latex (or None)
    eq_matched = {}        # eq -> True if its LHS matched one of its symbols
    eq_meaning_by_eq = {}  # eq -> the meaning chosen in pass 1

    # ---------- PASS 1: meanings + symbols, and build eq_context ----------
    for i, eq in enumerate(equaitons):
        index = i + 1
        if index not in paper_dataset[f"arXiv:{paper_id}"]:
            paper_dataset[f"arXiv:{paper_id}"][index] = {}

        # fresh audit trail for THIS equation only (flat: method -> messages)
        eq_audit = {}
        audits[eq] = eq_audit

        # audit: record the equation that was extracted (latex from html_parser)
        latex = eqn_mapping[eq]["latex"]
        eq_audit["extract_equations_method"] = latex

        symbols = eq_to_syms[eq]

        # audit: record which symbols were matched into this equation
        eq_audit["map_symbols_to_equations"] = {latex: [sym_mapping[s] for s in symbols]}

        paper_dataset[f"arXiv:{paper_id}"][index]["equation"] = latex

        # --- symbol meanings FIRST (needed to decide the equation meaning) ---
        # sym_meanings_by_ph maps placeholder -> meaning, so the LHS match below
        # can reuse a symbol's meaning directly by its placeholder.
        sym_meanings_by_ph = {}
        for sym in symbols:
            sym_meaning = get_meanings(clean_text, sym, audit=eq_audit, name_map=name_map)
            sym_meanings_by_ph[sym] = sym_meaning
            if "symbols" not in paper_dataset[f"arXiv:{paper_id}"][index]:
                paper_dataset[f"arXiv:{paper_id}"][index]["symbols"] = {}
            paper_dataset[f"arXiv:{paper_id}"][index]["symbols"][strip_backslash(sym_mapping[sym])] = sym_meaning

        # --- equation meaning ---
        # Phase 1 + Phase 2 run together inside get_description; conf tells which
        # produced it ("high" = Phase-1 structural rule, "low" = Phase-2 fallback).
        eq_meaning, conf = get_description(clean_text, eq, audit=eq_audit,
                                           name_map=name_map, return_conf=True)

        # does the LHS equal one of this equation's symbols? (raw latex on both)
        lhs = extract_lhs(latex)
        lhs_symbol = None
        if lhs:
            for s in symbols:
                if sym_mapping[s] == lhs:
                    lhs_symbol = s
                    break

        # REPLACE only a Phase-2 (low-confidence) meaning, and only when the LHS
        # matches a symbol that has a meaning. Phase-1 (high) results are kept.
        if (conf != "high") and lhs_symbol is not None and sym_meanings_by_ph.get(lhs_symbol):
            eq_meaning = sym_meanings_by_ph[lhs_symbol]
            eq_audit.setdefault("equation_meaning_method", {})[latex] = (
                f"Phase-2 result replaced: LHS '{lhs}' matches symbol -> {eq_meaning}"
            )

        paper_dataset[f"arXiv:{paper_id}"][index]["meaning"] = eq_meaning

        # remember for the LHS-grouping pass below
        eq_lhs[eq] = lhs
        eq_matched[eq] = lhs_symbol is not None
        eq_meaning_by_eq[eq] = eq_meaning

        # store audit so far (relations are appended in pass 2)
        paper_dataset[f"arXiv:{paper_id}"][index]["audit-trail"] = eq_audit

        # context for relation-matching = meaning + every symbol meaning
        eq_context[eq] = eq_meaning or ""

    # ---------- PASS 1.5: equations with the SAME LHS share one meaning ----------
    # group equations by their LHS latex
    lhs_groups = {}
    for eq in equaitons:
        l = eq_lhs.get(eq)
        if l:
            lhs_groups.setdefault(l, []).append(eq)

    for l, group in lhs_groups.items():
        if len(group) < 2:
            continue  # only groups that actually share an LHS

        # option 2: winner = meaning of an eq whose LHS matched a symbol;
        # otherwise the first non-empty meaning in the group.
        winner = None
        for eq in group:
            if eq_matched.get(eq) and eq_meaning_by_eq.get(eq):
                winner = eq_meaning_by_eq[eq]
                break
        if winner is None:
            for eq in group:
                if eq_meaning_by_eq.get(eq):
                    winner = eq_meaning_by_eq[eq]
                    break
        if winner is None:
            continue  # nobody in the group had a meaning

        # assign the winning meaning to every equation in the group
        for eq in group:
            idx = equaitons.index(eq) + 1
            paper_dataset[f"arXiv:{paper_id}"][idx]["meaning"] = winner
            eq_context[eq] = winner or ""
            audits[eq].setdefault("equation_meaning_method", {})[eqn_mapping[eq]["latex"]] = (
                f"shares LHS '{l}' -> unified meaning: {winner}"
            )

    # ---------- PASS 2: relations (needs the FULL eq_context) ----------
    for i, eq in enumerate(equaitons):
        index = i + 1
        eq_audit = audits[eq]

        relations = get_relations(eq, equaitons, eq_to_syms, sym_mapping,
                                  eqn_mapping, eq_context=eq_context, audit=eq_audit)
        paper_dataset[f"arXiv:{paper_id}"][index]["relations"] = relations

    # ---------- PASS 3: Save this specific paper's JSON before moving to the next ----------
    with open(f"./results/with_audit/{paper_id}.json", "w") as file:
        json.dump(paper_dataset, file, indent=4) # Added indent=4 to make your JSON files beautiful and readable!
    break