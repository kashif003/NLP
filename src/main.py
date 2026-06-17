from html_reader import HTMLReader
import time
paper_ids = [
# "2401.13724",
# "2408.07125",
# "2508.05295",
# "2510.12545",
"2410.08937"
]

start = time.time()
for paper_id in paper_ids:
    print("paper ID:",paper_id )
    reader = HTMLReader(paper_id)

    content = reader.file_content
    equattions= reader.finding_all_equations()
    #symbols = reader.get_all_paragraph_symbols()
    print(equattions)
    # full_context = reader.get_symbol_full_context()
    # for symbol, data in full_context.items():
    #     print(f"\nSymbol: {symbol}")
    #     print(f"In equations: {data['equations']}")
    #     print(f"All contexts:")
    #     for i, ctx in enumerate(data['contexts']):
    #         print(f"  [{i+1}] {ctx}")
print("done")
end  = time.time()
print(end-start)