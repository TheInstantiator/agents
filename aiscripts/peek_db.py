import sys
import os
import chromadb
from sentence_transformers import SentenceTransformer

DB_PATH = "/home/ouar/projects/agents/X_capstone_projects/DandDCapstone/chroma_db"
client = chromadb.PersistentClient(path=DB_PATH)
collection = client.get_collection(name="dnd_rules_multi_v1")

encoder = SentenceTransformer("BAAI/bge-large-en-v1.5", device="cpu")

search_text = "How much does a Longsword cost and what damage does it deal? A Longsword is a common melee weapon used by many adventurers. It costs 2 gold pieces and deals 1d8 slashing damage plus the wielder's Strength modifi..."
query_embedding = encoder.encode(search_text, normalize_embeddings=True).tolist()

print("--- Dense Top 10 ---")
results = collection.query(
    query_embeddings=[query_embedding],
    n_results=10,
    include=['documents', 'metadatas']
)
for i, d in enumerate(results['documents'][0]):
    print(f"Rank {i+1}: {d[:150].replace(chr(10), ' ')}")

print("\n--- Exact Match 'Longsword' Top 10 ---")
kw_res = collection.query(
    query_embeddings=[query_embedding],
    n_results=10,
    where_document={"$contains": "Longsword"},
    include=['documents', 'metadatas']
)
for i, (d, m) in enumerate(zip(kw_res['documents'][0], kw_res['metadatas'][0])):
    print(f"Rank {i+1}: {d[:150].replace(chr(10), ' ')} | Metadatas: {m.get('type')} {m.get('table_summary')}")

