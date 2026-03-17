import re
import sys 
sys.path.append("/home/ouar/projects/agents/X_capstone_projects/DandDCapstone/")
from build_rag_db import extract_tables_from_markdown

chunk = """
Some text above.
Longsword 1d8 Slashing Versatile (1d10) Sap 3 lb. 15 GP  
Maul 2d6 Bludgeoning Heavy, Two-Handed Topple 10 lb. 10 GP  
Morningstar 1d8 Piercing     - Sap 4 lb. 15 GP  
Pike 1d10 Piercing Heavy, Reach, Two-Handed Push 18 lb. 5 GP  
Some text below.
"""

tables = extract_tables_from_markdown(chunk)
print(f"Tables found: {len(tables)}")
for i, t in enumerate(tables):
    print(f"--- Table {i+1} ---")
    print(t)
