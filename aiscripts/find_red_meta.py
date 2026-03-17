import pickle, os
BM25_CORPUS_PATH = "/home/ouar/projects/agents/X_capstone_projects/DandDCapstone/bm25_corpus.pkl"
with open(BM25_CORPUS_PATH, "rb") as f: corpus = pickle.load(f)

for i, m in enumerate(corpus["metadatas"]):
    if "Red Dragon" in m.get("parent_summary", ""):
        print(f"Index {i} | Parent Summary: {m['parent_summary']}")
        break
