import os
import sys
import pickle
import chromadb
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

if len(sys.argv) < 2:
    query = "Heightened Spell"
else:
    query = " ".join(sys.argv[1:])

BM25_CORPUS_PATH = os.path.join(os.path.dirname(__file__), "..", "X_capstone_projects", "DandDCapstone", "bm25_corpus.pkl")
DB_PATH = os.path.join(os.path.dirname(__file__), "..", "X_capstone_projects", "DandDCapstone", "chroma_db")

print(f"Connecting to DB at: {DB_PATH}")
print("Loading Encoder BAAI/bge-large-en-v1.5...")
encoder = SentenceTransformer("BAAI/bge-large-en-v1.5", device="cpu")

print(f"Loading BM25 corpus from {BM25_CORPUS_PATH}...")
with open(BM25_CORPUS_PATH, "rb") as f:
    corpus = pickle.load(f)
bm25_docs = corpus["documents"]
bm25_metas = corpus["metadatas"]
bm25_ids = corpus["ids"]
tokenized_docs = [doc.lower().split() for doc in bm25_docs]
bm25_index = BM25Okapi(tokenized_docs)

client = chromadb.PersistentClient(path=DB_PATH)
try:
    collection = client.get_collection(name="dnd_rules_multi_v2")
    print(f"Found collection. Total documents: {collection.count()}")
    
    print(f"Querying for '{query}'...")
    
    # --- Dense Vector Retrieval ---
    query_emb = encoder.encode([query], normalize_embeddings=True).tolist()
    dense_results = collection.query(
        query_embeddings=query_emb,
        n_results=25
    )
    
    # --- BM25 Retrieval ---
    bm25_query_tokens = query.lower().split()
    bm25_scores = bm25_index.get_scores(bm25_query_tokens)
    top_bm25_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:25]
    
    print(f"\n=== TOP 5 DENSE ENCODING RESULTS FOR '{query}' ===")
    for i in range(5):
        doc = dense_results['documents'][0][i]
        dist = dense_results['distances'][0][i]
        print(f"[{i+1}] (Dist: {dist:.4f}) {doc[:150]}...")

    print(f"\n=== TOP 5 BM25 RESULTS FOR '{query}' ===")
    for i, idx in enumerate(top_bm25_indices[:5]):
        score = bm25_scores[idx]
        doc = bm25_docs[idx]
        print(f"[{i+1}] (Score: {score:.4f}) {doc[:150]}...")
        
except Exception as e:
    print(f"Error: {e}")
