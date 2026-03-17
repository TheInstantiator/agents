import re
import sys 
sys.path.append("/home/ouar/projects/agents/X_capstone_projects/DandDCapstone/")
from build_rag_db import extract_tables_from_markdown

chunk = """
_Martial Melee Weapons_

Battleaxe 1d8 Slashing Versatile (1d10) Topple 4 lb. 10 GP

Flail 1d8 Bludgeoning       - Sap 2 lb. 10 GP

Glaive 1d10 Slashing Heavy, Reach, Two-Handed Graze 6 lb. 20 GP

Greataxe 1d12 Slashing Heavy, Two-Handed Cleave 7 lb. 30 GP

Greatsword 2d6 Slashing Heavy, Two-Handed Graze 6 lb. 50 GP

Halberd 1d10 Slashing Heavy, Reach, Two-Handed Cleave 6 lb. 20 GP

Lance 1d10 Piercing Heavy, Reach, Two-Handed (unless mounted) Topple 6 lb. 10 GP

Longsword 1d8 Slashing Versatile (1d10) Sap 3 lb. 15 GP

Maul 2d6 Bludgeoning Heavy, Two-Handed Topple 10 lb. 10 GP

Morningstar 1d8 Piercing     - Sap 4 lb. 15 GP
"""

tables = extract_tables_from_markdown(chunk)
print(f"Tables found: {len(tables)}")
for i, t in enumerate(tables):
    print(f"--- Table {i+1} ---")
    print(t)
