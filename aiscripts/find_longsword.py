import pymupdf4llm
import os

DOC_PATH = "/home/ouar/projects/agents/X_capstone_projects/documents/DandDRuleset.pdf"

print(f"Converting PDF to markdown...")
md_text = pymupdf4llm.to_markdown(DOC_PATH)

start_idx = md_text.find("Longsword")
if start_idx != -1:
    snippet = md_text[max(0, start_idx - 1000):start_idx + 1000]
    print("\n--- Snippet around 'Longsword' ---")
    print(snippet)
else:
    print("'Longsword' not found.")
