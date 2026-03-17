import os
import chromadb
from sentence_transformers import SentenceTransformer

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "X_capstone_projects", "DandDCapstone", "chroma_db")
client = chromadb.PersistentClient(path=DB_PATH)
collection = client.get_collection(name="dnd_rules_multi_v2")

# We want to find the exact chunk representing the Longsword weapon table entry.
# Let's search by a query that should match it well: "Longsword 1d8 Slashing Versatile (1d10) Sap 3 lb. 15 GP"
encoder = SentenceTransformer("BAAI/bge-large-en-v1.5", device="cpu")
query = "Longsword 1d8 Slashing Versatile (1d10) Sap 3 lb. 15 GP"
query_emb = encoder.encode([query], normalize_embeddings=True).tolist()

results = collection.query(
    query_embeddings=query_emb,
    n_results=10
)

print(f"--- Top 10 Chunks for the Longsword Table Row ---\n")
for i in range(len(results['documents'][0])):
    doc = results['documents'][0][i]
    meta = results['metadatas'][0][i]
    dist = results['distances'][0][i]
    
    # We only care if it mentions "15 GP" or "1d8" to narrow it down
    if "15 gp" in doc.lower() or "1d8" in doc.lower() or "15 gp" in meta.get("table_row_prose", "").lower():
        print(f"Rank {i+1} (Distance: {dist:.4f})")
        print(f"ID: {results['ids'][0][i]}")
        print(f"Type: {meta.get('type')}")
        print(f"Parent Summary: {meta.get('parent_summary')}")
        print(f"--- Document Content (This is what is ENCODED by BGE & scored by BM25/CE) ---")
        print(doc)
        print(f"--- Metadata ---")
        for k, v in meta.items():
            if k == "table_content":
                print(f"  {k}: [Truncated, length {len(v)}]")
            else:
                print(f"  {k}: {v}")
        print("="*80 + "\n")
