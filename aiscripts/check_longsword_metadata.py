import os
import sys
import pickle
import chromadb
from pprint import pprint
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

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
    print(f"Found collection: dnd_rules_multi_v2. Total documents: {collection.count()}")
    
    query = "longsword"
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
    
    output_file = os.path.join(os.path.dirname(__file__), "longsword_metadata_chunks.txt")
    with open(output_file, "w") as f:
        f.write(f"=== TOP 25 DENSE ENCODING RESULTS FOR '{query}' ===\n\n")
        val_docs = dense_results['documents'][0]
        val_metas = dense_results['metadatas'][0]
        val_dists = dense_results['distances'][0]
        for i in range(len(val_docs)):
            doc = val_docs[i]
            meta = val_metas[i]
            dist = val_dists[i]
            f.write(f"[Dense Rank {i+1}] (Distance: {dist:.4f})\n")
            f.write(f"Document ID     : {dense_results['ids'][0][i]}\n")
            f.write(f"Type            : {meta.get('type', 'N/A')}\n")
            f.write(f"Parent Summary  : {meta.get('parent_summary', 'N/A')}\n")
            f.write(f"Table Summary   : {meta.get('table_summary', 'N/A')}\n")
            f.write(f"Chunk Preview   : {doc[:300].strip()}\n")
            if 'table_content' in meta:
                f.write(f"Raw Table Sample: {meta['table_content'][:100].replace('\n', ' ')}...\n")
            f.write("-" * 60 + "\n\n")

        f.write("\n\n" + "=" * 80 + "\n\n")
        f.write(f"=== TOP 25 BM25 RESULTS FOR '{query}' ===\n\n")
        for i, idx in enumerate(top_bm25_indices):
            score = bm25_scores[idx]
            doc = bm25_docs[idx]
            meta = bm25_metas[idx]
            f.write(f"[BM25 Rank {i+1}] (Score: {score:.4f})\n")
            f.write(f"Document ID     : {bm25_ids[idx]}\n")
            f.write(f"Type            : {meta.get('type', 'N/A')}\n")
            f.write(f"Parent Summary  : {meta.get('parent_summary', 'N/A')}\n")
            f.write(f"Table Summary   : {meta.get('table_summary', 'N/A')}\n")
            f.write(f"Chunk Preview   : {doc[:300].strip()}\n")
            if 'table_content' in meta:
                f.write(f"Raw Table Sample: {meta['table_content'][:100].replace('\n', ' ')}...\n")
            f.write("-" * 60 + "\n\n")

    print(f"Results written to {output_file}")
        
except Exception as e:
    print(f"Error: {e}")
