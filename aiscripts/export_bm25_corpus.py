"""
Quick script to export the current ChromaDB collection to a bm25_corpus.pkl file.
This lets run_evals_routed.py use a real BM25 index without requiring a full re-index.

Run from the agents/ directory:
    uv run aiscripts/export_bm25_corpus.py
"""
import os
import sys
import pickle
import chromadb

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DANDD_DIR = os.path.join(SCRIPT_DIR, "..", "X_capstone_projects", "DandDCapstone")
DB_DIR = os.path.join(DANDD_DIR, "chroma_db")
OUTPUT_PATH = os.path.join(DANDD_DIR, "bm25_corpus.pkl")

print(f"Connecting to ChromaDB at {DB_DIR}...")
client = chromadb.PersistentClient(path=DB_DIR)
collection = client.get_collection(name="dnd_rules_multi_v2")

print("Fetching all documents from collection (this may take a moment)...")
all_data = collection.get(include=["documents", "metadatas"])

documents = all_data["documents"]
metadatas = all_data["metadatas"]
ids = all_data["ids"]

print(f"  -> Fetched {len(documents)} documents.")

bm25_corpus = {
    "documents": documents,
    "metadatas": metadatas,
    "ids": ids,
}

print(f"Writing BM25 corpus to {OUTPUT_PATH}...")
with open(OUTPUT_PATH, "wb") as f:
    pickle.dump(bm25_corpus, f)

print(f"✅ Done! BM25 corpus saved with {len(documents)} documents.")
print(f"   run_evals_routed.py will now pick this up automatically at startup.")
