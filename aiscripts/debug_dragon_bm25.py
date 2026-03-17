import os
import sys
import pickle
import chromadb
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import re

def simple_tokenize(text: str):
    if not text: return []
    return re.findall(r'\w+', text.lower())

BM25_CORPUS_PATH = "/home/ouar/projects/agents/X_capstone_projects/DandDCapstone/bm25_corpus.pkl"

with open(BM25_CORPUS_PATH, "rb") as f:
    corpus = pickle.load(f)

docs = corpus["documents"]
ids = corpus["ids"]
tokenized = [simple_tokenize(d) for d in docs]
bm25 = BM25Okapi(tokenized)

query = "Adult Red Dragon Legendary Resistance"
query_tokens = simple_tokenize(query)
scores = bm25.get_scores(query_tokens)

top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:10]

print(f"Top 10 BM25 for: {query}")
for i, idx in enumerate(top_indices):
    print(f"[{i+1}] Score: {scores[idx]:.2f} | ID: {ids[idx]} | Preview: {docs[idx][:100].replace('\n', ' ')}")
    if "Red" in docs[idx]:
        print("    -> FOUND 'Red' in text")
    if "Adult" in docs[idx]:
        print("    -> FOUND 'Adult' in text")
