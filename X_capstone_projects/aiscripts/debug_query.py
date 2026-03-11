"""
Debug script: Shows what chunks are returned for a specific query,
including the full parent chunk text pulled from SQLite.
Usage: uv run aiscripts/debug_query.py "your question here"
"""
import sys
import os
import sqlite3
import chromadb
from sentence_transformers import SentenceTransformer

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.join(SCRIPT_DIR, "..")  # aiscripts/ lives inside X_capstone_projects/
DB_PATH = os.path.join(PROJECT_DIR, "DandDCapstone", "chroma_db")
SQLITE_PATH = os.path.join(PROJECT_DIR, "DandDCapstone", "parent_chunks.db")

query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "What is the maximum number of spell levels a Rod of Absorption can absorb over its entire existence?"

K = 5

print(f"\n🔍 Query: {query}\n")
print("Loading encoder...")
encoder = SentenceTransformer("BAAI/bge-large-en-v1.5", device="cpu")

print("Connecting to ChromaDB...")
client = chromadb.PersistentClient(path=DB_PATH)
col = client.get_collection("dnd_rules_multi_v1")

print("Connecting to SQLite...")
parent_db = sqlite3.connect(SQLITE_PATH)
cursor = parent_db.cursor()

embedding = encoder.encode(query, normalize_embeddings=True).tolist()
results = col.query(query_embeddings=[embedding], n_results=K, include=["documents", "metadatas"])

print(f"\n{'='*60}")
print(f"TOP {K} RETRIEVED CHUNKS")
print(f"{'='*60}")

for rank, (doc, meta) in enumerate(zip(results['documents'][0], results['metadatas'][0])):
    chunk_id = results['ids'][0][rank]
    parent_id = "_".join(chunk_id.split("_")[:3])

    # Fetch parent
    cursor.execute("SELECT content FROM parent_chunks WHERE id=?", (parent_id,))
    parent_row = cursor.fetchone()
    parent_text = parent_row[0] if parent_row else "[Parent not found]"

    print(f"\n--- Rank {rank+1} | Type: {meta.get('type')} ---")
    print(f"Chunk ID : {chunk_id}")
    print(f"Parent ID: {parent_id}")
    print(f"\n[CHILD CHUNK TEXT]:\n{doc[:400]}")
    if meta.get("parent_summary"):
        print(f"\n[PARENT SUMMARY]: {meta.get('parent_summary')}")
    print(f"\n[FULL PARENT CHUNK] ({len(parent_text)} chars):\n{parent_text[:800]}...")
    print(f"\n{'─'*60}")

parent_db.close()
