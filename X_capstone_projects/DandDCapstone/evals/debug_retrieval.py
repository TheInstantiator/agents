import json
import asyncio
import chromadb
import sqlite3
import sys
import os
import pickle
import re

from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

# ========================= PATHS =========================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(SCRIPT_DIR, "..", ".."))

DB_PATH = os.path.join(os.path.dirname(SCRIPT_DIR), "chroma_db")
SQLITE_PATH = os.path.join(os.path.dirname(SCRIPT_DIR), "parent_chunks.db")
BM25_CORPUS_PATH = os.path.join(os.path.dirname(SCRIPT_DIR), "bm25_corpus.pkl")

# ========================= SETUP =========================
print("Initializing databases...")
client = chromadb.PersistentClient(path=DB_PATH)
collection = client.get_collection(name="dnd_rules_multi_v2")

encoder = SentenceTransformer("BAAI/bge-large-en-v1.5", device="cpu")

with open(BM25_CORPUS_PATH, "rb") as f:
    bm25_data = pickle.load(f)
bm25_corpus_docs = bm25_data["documents"]
bm25_corpus_metas = bm25_data["metadatas"]
bm25_corpus_ids = bm25_data["ids"]

_tokenized = [re.findall(r'\w+', (meta.get('parent_summary', '') + " " + doc).lower())
              for doc, meta in zip(bm25_corpus_docs, bm25_corpus_metas)]
bm25_index = BM25Okapi(_tokenized)

def reciprocal_rank_fusion(vector_results, bm25_results, k=25):
    scores = {}
    for top_k_ids in vector_results.get('ids', []):
        if not top_k_ids: continue
        for rank, cid in enumerate(top_k_ids):
            scores[cid] = scores.get(cid, 0) + 1.0 / (rank + 60)
    
    for top_k_ids in bm25_results.get('ids', []):
        if not top_k_ids: continue
        for rank, cid in enumerate(top_k_ids):
            scores[cid] = scores.get(cid, 0) + 1.0 / (rank + 60)
    
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]

def search_query(query: str):
    print(f"\n--- Searching for: {query} ---")
    
    # 1. Vector Search
    embeddings = encoder.encode([query], normalize_embeddings=True).tolist()
    vec_results = collection.query(
        query_embeddings=embeddings,
        n_results=10,
        include=['documents', 'metadatas']
    )
    
    print("\n--- Top 3 Vector Search Results ---")
    for i in range(min(3, len(vec_results['ids'][0]))):
        doc = vec_results['documents'][0][i]
        meta = vec_results['metadatas'][0][i]
        print(f"\nResult {i+1} (ID: {vec_results['ids'][0][i]}):")
        print(f"Type: {meta.get('type')}")
        print(f"Parent Summary: {meta.get('parent_summary')}")
        print(f"Content:\n{doc[:200]}...")
        
    # 2. BM25 Search
    tokens = re.findall(r'\w+', query.lower())
    print(f"\nBM25 Tokens: {tokens}")
    bm25_scores = bm25_index.get_scores(tokens)
    top_bm25_idx = sorted(range(len(bm25_scores)), key=lambda x: bm25_scores[x], reverse=True)[:10]
    
    print("\n--- Top 3 BM25 Search Results ---")
    bm25_res = {'ids': [[]]}
    for i, idx in enumerate(top_bm25_idx[:3]):
        cid = bm25_corpus_ids[idx]
        bm25_res['ids'][0].append(cid)
        doc = bm25_corpus_docs[idx]
        meta = bm25_corpus_metas[idx]
        print(f"\nResult {i+1} (ID: {cid}, Score: {bm25_scores[idx]:.2f}):")
        print(f"Type: {meta.get('type')}")
        print(f"Parent Summary: {meta.get('parent_summary')}")
        print(f"Content:\n{doc[:200]}...")

    # Build full BM25 results dict for RRF
    bm25_res_full = {
        'ids': [[bm25_corpus_ids[i] for i in top_bm25_idx]]
    }

    # 3. RRF Merge
    rrf_merged = reciprocal_rank_fusion(vec_results, bm25_res_full, k=5)
    print("\n--- Top 5 RRF Merged Results ---")
    for i, (cid, score) in enumerate(rrf_merged):
        try:
            corpus_idx = bm25_corpus_ids.index(cid)
            doc = bm25_corpus_docs[corpus_idx]
            meta = bm25_corpus_metas[corpus_idx]
        except ValueError:
            doc = "Unknown"
            meta = {}
        print(f"\nRank {i+1} (ID: {cid}, RRF Score: {score:.4f}):")
        print(f"Type: {meta.get('type')}")
        print(f"Parent Summary: {meta.get('parent_summary')}")
        print(f"Content:\n{doc[:200]}...")

if __name__ == '__main__':
    search_query("Death Saving Throws rule summary")
