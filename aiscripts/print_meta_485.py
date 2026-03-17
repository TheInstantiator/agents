import pickle, os
BM25_CORPUS_PATH = "/home/ouar/projects/agents/X_capstone_projects/DandDCapstone/bm25_corpus.pkl"
with open(BM25_CORPUS_PATH, "rb") as f: corpus = pickle.load(f)

for i, id_s in enumerate(corpus["ids"]):
    if "parent_485" in id_s:
        print(f"Index {i} | ID: {id_s} | Summary: {corpus['metadatas'][i].get('parent_summary')}")
        break
