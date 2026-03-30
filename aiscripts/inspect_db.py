import chromadb
import os
import sys

CAPSTONE_DIR = "/home/ouar/projects/agents/X_capstone_projects/DandDCapstone"
DB_DIR = os.path.join(CAPSTONE_DIR, "chroma_db")

client = chromadb.PersistentClient(path=DB_DIR)
collection = client.get_collection(name="dnd_rules_multi_v2")

def search_text(query_text, n=10):
    print(f"\n--- Searching for: '{query_text}' ---")
    # We don't have the encoder here, so we use the collection's built-in if it has one, 
    # but build_rag_db used a specific one. We'll just use get() with where if possible, 
    # or just list some chunks.
    results = collection.get(limit=n, include=['documents', 'metadatas'])
    for doc, meta in zip(results['documents'], results['metadatas']):
        print(f"[{meta.get('type')}] {doc[:200]}...")

def find_by_type(type_name, limit=20):
    print(f"\n--- Finding chunks of type: '{type_name}' ---")
    results = collection.get(where={"type": type_name}, limit=limit, include=['documents', 'metadatas'])
    for doc, meta in zip(results['documents'], results['metadatas']):
        print(f"[{meta.get('parent_id')}] {doc}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        find_by_type(sys.argv[1])
    else:
        # Default: look for table row prose related to Monk
        print("Looking for Monk related table rows...")
        results = collection.get(
            where={"type": "table_row_prose"},
            include=['documents', 'metadatas']
        )
        count = 0
        for doc, meta in zip(results['documents'], results['metadatas']):
            if "Monk" in doc or "Monk" in meta.get('parent_summary', ''):
                print(f"ID: {meta.get('parent_id')} | Doc: {doc}")
                count += 1
            if count > 20: break
        print(f"Found {count} matches.")
