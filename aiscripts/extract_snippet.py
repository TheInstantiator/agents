import pymupdf4llm
import os

DOC_PATH = "/home/ouar/projects/agents/X_capstone_projects/documents/DandDRuleset.pdf"
OUTPUT_PATH = "/home/ouar/projects/agents/aiscripts/monk_section_snippet.md"

if not os.path.exists(DOC_PATH):
    print(f"PDF not found at {DOC_PATH}")
    exit(1)

print(f"Converting PDF {DOC_PATH} to markdown...")
try:
    md_text = pymupdf4llm.to_markdown(DOC_PATH)
    print(f"Conversion complete. Total length: {len(md_text)} characters.")
except Exception as e:
    print(f"Conversion failed: {e}")
    exit(1)

start_idx = md_text.find("# Monk")
if start_idx == -1:
    start_idx = md_text.find("## Monk")
    print("Found '## Monk'")
else:
    print("Found '# Monk'")

if start_idx != -1:
    snippet = md_text[start_idx:start_idx + 20000]
    with open(OUTPUT_PATH, "w") as f:
        f.write(snippet)
    print(f"Saved snippet to {OUTPUT_PATH}")
else:
    print("Monk section NOT found using headers. Searching for 'Monk' text...")
    start_idx = md_text.find("Monk")
    if start_idx != -1:
         snippet = md_text[max(0, start_idx-500):start_idx + 10000]
         with open(OUTPUT_PATH, "w") as f:
             f.write(snippet)
         print(f"Saved 'Monk' text snippet to {OUTPUT_PATH}")
    else:
        print("Saving first 10000 chars as fallback.")
        with open(OUTPUT_PATH, "w") as f:
            f.write(md_text[:10000])
        print(f"Saved fallback to {OUTPUT_PATH}")
