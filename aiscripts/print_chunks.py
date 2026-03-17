import pickle, os
BM25_CORPUS_PATH = os.path.join(os.path.dirname(__file__), "..", "X_capstone_projects", "DandDCapstone", "bm25_corpus.pkl")
with open(BM25_CORPUS_PATH, "rb") as f: corpus = pickle.load(f)
print(f"--- 746 ---\n{corpus['documents'][746]}")
print(f"\n--- 747 ---\n{corpus['documents'][747]}")
