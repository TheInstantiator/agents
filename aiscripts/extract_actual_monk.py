import pymupdf4llm
import os

DOC_PATH = "/home/ouar/projects/agents/X_capstone_projects/documents/DandDRuleset.pdf"
OUTPUT_PATH = "/home/ouar/projects/agents/aiscripts/monk_actual_content.md"

print(f"Converting PDF to markdown...")
md_text = pymupdf4llm.to_markdown(DOC_PATH)

# Extract from # Monk to somewhere after Monk Class Features
snippet = md_text[193000:210000]
with open(OUTPUT_PATH, "w") as f:
    f.write(snippet)
print(f"Saved snippet to {OUTPUT_PATH}")
