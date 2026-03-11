import os
import sys
import chromadb
from pprint import pprint
from sentence_transformers import SentenceTransformer

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "DandDCapstone", "chroma_db")
print(f"Connecting to DB at: {DB_PATH}")

print("Loading Encoder BAAI/bge-large-en-v1.5...")
encoder = SentenceTransformer("BAAI/bge-large-en-v1.5")

client = chromadb.PersistentClient(path=DB_PATH)
try:
    collection = client.get_collection(name="dnd_rules_multi_v2")
    print(f"Found collection: dnd_rules_multi_v2. Total documents: {collection.count()}")
    
    # Query for "longsword"
    query_emb = encoder.encode(["longsword"], normalize_embeddings=True).tolist()
    
    results = collection.query(
        query_embeddings=query_emb,
        n_results=5
    )
    
    output_file = os.path.join(os.path.dirname(__file__), "longsword_metadata_chunks.txt")
    with open(output_file, "w") as f:
        f.write("--- TOP 5 RESULTS FOR 'longsword' ---\n")
        for i in range(len(results['documents'][0])):
            doc = results['documents'][0][i]
            meta = results['metadatas'][0][i]
            distance = results['distances'][0][i]
            
            f.write(f"\n[Rank {i+1}] (Score: {distance:.4f})\n")
            f.write(f"Parent Summary : {meta.get('parent_summary', 'N/A')}\n")
            f.write(f"Top Keywords   : {meta.get('top_keywords', 'N/A')}\n")
            f.write(f"Chunk Preview  : {doc}\n")
    print(f"Results written to {output_file}")
        
except Exception as e:
    print(f"Error: {e}")
