import pymupdf4llm
import os
import re

DOC_PATH = "/home/ouar/projects/agents/X_capstone_projects/documents/DandDRuleset.pdf"

if not os.path.exists(DOC_PATH):
    print(f"PDF not found at {DOC_PATH}")
    exit(1)

print(f"Converting PDF to markdown...")
md_text = pymupdf4llm.to_markdown(DOC_PATH)

print("\nSearching for 'Monk' headers:")
for match in re.finditer(r'^#+ .*Monk.*$', md_text, re.MULTILINE):
    print(f"Position {match.start()}: '{match.group()}'")
    # Print 100 chars after
    print(f"  Context: {md_text[match.end():match.end()+100].strip().replace('\n', ' ')}")

# Search for potential table markers
print("\nSearching for potential Monk table markers (Level 1, Level 2...):")
for match in re.finditer(r'Level\s+Proficiency\s+Bonus', md_text):
    print(f"Position {match.start()}: '{match.group()}'")
    print(f"  Context: {md_text[match.start():match.start()+200].strip().replace('\n', ' ')}")
