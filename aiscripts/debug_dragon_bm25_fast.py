import os
import pickle
import re
from rank_bm25 import BM25Okapi

def simple_tokenize(text: str):
    if not text: return []
    return re.findall(r'\w+', text.lower())

BM25_CORPUS_PATH = "/home/ouar/projects/agents/X_capstone_projects/DandDCapstone/bm25_corpus.pkl"

if not os.path.exists(BM25_CORPUS_PATH):
    print("Corpus not found")
    exit()

with open(BM25_CORPUS_PATH, "rb") as f:
    corpus = pickle.load(f)

docs = corpus["documents"]
ids = corpus["ids"]
print(f"Loaded {len(docs)} documents")

tokenized = [simple_tokenize(d) for d in docs]
bm25 = BM25Okapi(tokenized)

query = "Adult Red Dragon Legendary Resistance"
query_tokens = simple_tokenize(query)
scores = bm25.get_scores(query_tokens)

top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:10]

print(f"Top 10 BM25 for: {query}")
for i, idx in enumerate(top_indices):
    preview = docs[idx][:120].replace('\n', ' ')
    print(f"[{i+1}] Score: {scores[idx]:.2f} | ID: {ids[idx]} | Preview: {preview}")
