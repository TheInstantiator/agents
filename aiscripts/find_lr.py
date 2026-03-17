import pickle, os

BM25_CORPUS_PATH = "/home/ouar/projects/agents/X_capstone_projects/DandDCapstone/bm25_corpus.pkl"
with open(BM25_CORPUS_PATH, "rb") as f:
    corpus = pickle.load(f)

docs = corpus["documents"]
ids = corpus["ids"]

for i, d in enumerate(docs):
    if "Legendary Resistance (3/Day, or 4/Day in Lair)" in d:
        print(f"Index {i} | ID: {ids[i]} | Preview: {d[:200].replace('\n', ' ')}")
