import pickle, os
BM25_CORPUS_PATH = "/home/ouar/projects/agents/X_capstone_projects/DandDCapstone/bm25_corpus.pkl"
with open(BM25_CORPUS_PATH, "rb") as f: corpus = pickle.load(f)
print(f"--- Child 0 ---\n{corpus['documents'][2511]}")
print(f"\n--- Child 1 ---\n{corpus['documents'][2512]}")
