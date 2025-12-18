import re
from transformers import pipeline
import torch

# 1. SETUP QA PIPELINE
device = 0 if torch.cuda.is_available() else -1
print("Loading QA Model...")
qa_pipeline = pipeline(
    "question-answering", 
    model="deepset/roberta-base-squad2", 
    device=device
)

def extract_metadata_final(description):
    results = {
        "algorithm": None,
        "gates": [], 
        "qubits": None
    }

    # --- PART A: ROBUST GATE DETECTION (REGEX) ---
    gate_patterns = [
        r"\b(cnot|cx|controlled-not)\b",
        r"\b(hadamard|h-gate|h gate)\b",
        r"\b(toffoli|ccx|ccn)\b",
        r"\b(pauli-x|x-gate|sigma-x)\b",
        r"\b(pauli-z|z-gate|phase-flip)\b",
        r"\b(rotation|r[xyz]|theta)\b",
        r"\b(swap|cz|controlled-z)\b",
        r"\b(measure|measurement)\b"
    ]
    
    found_gates = set()
    lower_desc = description.lower()
    
    for pat in gate_patterns:
        match = re.search(pat, lower_desc)
        if match:
            found_gates.add(match.group(1))
            
    if found_gates:
        results["gates"] = list(found_gates)

    # --- PART B: ROBUST ALGORITHM DETECTION (QA LOOP) ---
    algo_questions = [
        "What is the name of the protocol?",
        "What is the name of the quantum algorithm?",
        "Which quantum ansatz is used?", 
        "What acronym is mentioned?"
    ]
    
    best_algo = None
    best_score = 0.0
    
    for q in algo_questions:
        pred = qa_pipeline(question=q, context=description)
        
        # Filter 1: Length check (Algorithms are usually short names)
        if len(pred['answer'].split()) > 5:
            continue
            
        if pred['score'] > best_score:
            best_score = pred['score']
            best_algo = pred['answer']
    
    # Filter 2: Sanity Check (The "Fix")
    # If the "Algorithm" is just talking about gates/wires, reject it.
    if best_algo:
        bad_words = ["gate", "qubit", "circuit", "wire", "line", "diagram"]
        if any(w in best_algo.lower() for w in bad_words):
            best_algo = None  # Reject this answer

    if best_score > 0.1 and best_algo:
        results["algorithm"] = best_algo

    # --- PART C: QUBIT COUNT (HYBRID) ---
    # 1. Regex (Fastest)
    qubit_match = re.search(r"(\d+)\s*-?\s*qubits?", lower_desc)
    if qubit_match:
        results["qubits"] = qubit_match.group(1)
    else:
        # 2. QA Fallback
        pred_q = qa_pipeline(question="How many qubits?", context=description)
        if pred_q['score'] > 0.1 and len(pred_q['answer']) < 5:
             # Ensure it's actually a number
            if any(char.isdigit() for char in pred_q['answer']):
                results["qubits"] = pred_q['answer']

    return results

# --- FINAL VERIFICATION ---
descriptions = [
    "Figure 1: A schematic of the VQE ansatz.", 
    "The circuit uses CNOT gates and a Hadamard gate.",
    "A generic diagram of a process flow." 
]

print(f"\n{'ALGORITHM':<15} | {'GATES':<30} | {'QUBITS'}")
print("-" * 65)
import json

with open("output.json", "r") as file:
    json_file= json.load(file)

for desc in json_file['2509.04140_6.png']["description"]:
    data = extract_metadata_final(desc)
    
    algo = data['algorithm'] if data['algorithm'] else "None"
    gates = ", ".join(data['gates']) if data['gates'] else "None"
    qubits = data['qubits'] if data['qubits'] else "None"
    
    print(f"{algo:<15} | {gates:<30} | {qubits}")




print(json_file['2509.04140_6.png']["description"])