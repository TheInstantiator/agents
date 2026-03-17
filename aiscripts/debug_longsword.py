import sys, os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import chromadb
import sqlite3
from sentence_transformers import SentenceTransformer

os.chdir("/home/ouar/projects/agents/X_capstone_projects")

DB_PATH = "DandDCapstone/chroma_db"
SQLITE_PATH = "DandDCapstone/parent_chunks.db"

client = chromadb.PersistentClient(path=DB_PATH)
collection = client.get_collection(name="dnd_rules_multi_v1")
parent_db = sqlite3.connect(SQLITE_PATH)
parent_cursor = parent_db.cursor()
encoder = SentenceTransformer("BAAI/bge-large-en-v1.5", device="cpu")

query = "How much does a Longsword cost and what damage does it deal?"
hyde = "A longsword is a martial melee weapon that costs 15 gp and deals 1d8 slashing damage on a hit. It has the versatile property, allowing it to be wielded with two hands for 1d10 damage."

# TEST 1: Raw query only
print("\n=== TEST 1: Raw Query Embedding ===")
emb = encoder.encode(query, normalize_embeddings=True).tolist()
results = collection.query(query_embeddings=[emb], n_results=5, include=['documents', 'metadatas'])
for i, (doc, meta) in enumerate(zip(results['documents'][0], results['metadatas'][0])):
    print(f"\n[Rank {i+1}] Type={meta.get('type')} | Topic={meta.get('parent_summary','')[:80]}")
    print(f"  Doc: {doc[:200]}")

# TEST 2: HyDE only
print("\n\n=== TEST 2: HyDE Only Embedding ===")
emb2 = encoder.encode(hyde, normalize_embeddings=True).tolist()
results2 = collection.query(query_embeddings=[emb2], n_results=5, include=['documents', 'metadatas'])
for i, (doc, meta) in enumerate(zip(results2['documents'][0], results2['metadatas'][0])):
    print(f"\n[Rank {i+1}] Type={meta.get('type')} | Topic={meta.get('parent_summary','')[:80]}")
    print(f"  Doc: {doc[:200]}")

# TEST 3: Hybrid (Query + HyDE)
print("\n\n=== TEST 3: Hybrid (Query + HyDE) Embedding ===")
search_text = f"Question: {query}\nRule Summary: {hyde}"
emb3 = encoder.encode(search_text, normalize_embeddings=True).tolist()
results3 = collection.query(query_embeddings=[emb3], n_results=5, include=['documents', 'metadatas'])
for i, (doc, meta) in enumerate(zip(results3['documents'][0], results3['metadatas'][0])):
    print(f"\n[Rank {i+1}] Type={meta.get('type')} | Topic={meta.get('parent_summary','')[:80]}")
    print(f"  Doc: {doc[:200]}")

# Check if the parent chunk DOES have the data for whichever rank hits
print("\n\n=== PARENT CHUNK CHECK for Rank 1 from Test 1 ===")
chunk_id = results['ids'][0][0]
parent_id = "_".join(chunk_id.split("_")[:3])
print(f"chunk_id: {chunk_id}  parent_id: {parent_id}")
parent_cursor.execute("SELECT content FROM parent_chunks WHERE id=?", (parent_id,))
row = parent_cursor.fetchone()
if row:
    print(row[0][:1500])
else:
    print("NO PARENT FOUND")
