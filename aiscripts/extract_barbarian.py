import pymupdf4llm
import os

DOC_PATH = "/home/ouar/projects/agents/X_capstone_projects/documents/DandDRuleset.pdf"
OUTPUT_PATH = "/home/ouar/projects/agents/aiscripts/barbarian_section.md"

print(f"Converting PDF to markdown...")
md_text = pymupdf4llm.to_markdown(DOC_PATH)

start_idx = md_text.find("## **Barbarian**")
if start_idx == -1:
    start_idx = md_text.find("# Barbarian")

if start_idx != -1:
    snippet = md_text[start_idx:start_idx + 15000]
    with open(OUTPUT_PATH, "w") as f:
        f.write(snippet)
    print(f"Saved snippet to {OUTPUT_PATH}")
else:
    print("Barbarian section not found.")
