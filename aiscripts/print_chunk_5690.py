import pickle, os
BM25_CORPUS_PATH = "/home/ouar/projects/agents/X_capstone_projects/DandDCapstone/bm25_corpus.pkl"
with open(BM25_CORPUS_PATH, "rb") as f: corpus = pickle.load(f)
print(corpus["documents"][5690])
