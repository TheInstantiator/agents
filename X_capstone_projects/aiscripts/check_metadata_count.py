import sys
import os
import chromadb

# Setup paths
SCRIPT_DIR = "/home/ouar/projects/agents/X_capstone_projects/aiscripts"
DB_PATH = "/home/ouar/projects/agents/X_capstone_projects/chroma_db"

print("Connecting to ChromaDB...", flush=True)
client = chromadb.PersistentClient(path=DB_PATH)
collection = client.get_collection(name="dnd_rules_multi_v1")

# Fetch all metadata
results = collection.get(include=['metadatas'])
metadatas = results['metadatas']

count = 0
examples = []

print(f"Total chunks in DB: {len(metadatas)}", flush=True)

for m in metadatas:
    # Check all string values in the metadata dict
    found = False
    for key, value in m.items():
        if isinstance(value, str) and "Longsword".lower() in value.lower():
            found = True
            break
            
    if found:
        count += 1
        if len(examples) < 5:
            examples.append(m)

print(f"\n'Longsword' appears in the metadata of {count} chunks.")
print("\nExamples of metadata containing 'Longsword':")
for ex in examples:
    print(ex)
    print("-" * 40)
