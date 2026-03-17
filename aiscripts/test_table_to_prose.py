import asyncio
import sys 
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "X_capstone_projects"))
dnd_capstone_path = os.path.join(project_root, "DandDCapstone")

sys.path.append(dnd_capstone_path)
from build_rag_db import extract_tables_from_markdown, convert_table_to_prose_rows

sys.path.append(project_root)
from core_agent import CoreAgent

chunk = """
_Simple Melee Weapons_

Club 1d4 Bludgeoning Light Slow 2 lb. 1 SP

Dagger 1d4 Piercing Finesse, Light, Thrown (Range 20/60) Nick 1 lb. 2 GP

Greatclub 1d8 Bludgeoning Two-Handed Push 10 lb. 2 SP

Handaxe 1d6 Slashing Light, Thrown (Range 20/60) Vex 2 lb. 5 GP

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

async def main():
    tables = extract_tables_from_markdown(chunk)
    print(f"Tables found: {len(tables)}\n")
    
    # Initialize the LLM agent used by build_rag_db
    test_kwargs = CoreAgent.load_litellm_kwargs_from_config("phi-agent", config_path="config.json")
    test_kwargs["temperature"] = 0.0
    table_agent = CoreAgent(
        agent_id="table_processor_test",
        system_prompt="You are a data processing assistant.",
        litellm_kwargs=test_kwargs,
        one_shot=True
    )

    for i, t in enumerate(tables):
        print(f"--- Processing Table {i+1} ---")
        prose_rows = await convert_table_to_prose_rows(table_agent, t)
        print("\n--- Output Prose Rows ---")
        for row in prose_rows:
            print(f"- {row}")

if __name__ == "__main__":
    asyncio.run(main())
