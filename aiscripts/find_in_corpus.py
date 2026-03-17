import pickle
import os

BM25_CORPUS_PATH = os.path.join(os.path.dirname(__file__), "..", "X_capstone_projects", "DandDCapstone", "bm25_corpus.pkl")

with open(BM25_CORPUS_PATH, "rb") as f:
    corpus = pickle.load(f)

docs = corpus["documents"]
found = False
for i, d in enumerate(docs):
    if "Heightened Spell" in d:
        print(f"Index {i}: {d[:100]}...")
        found = True

if not found:
    print("NOT FOUND IN CORPUS")
