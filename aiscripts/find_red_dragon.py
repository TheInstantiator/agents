import os
import pickle
import re

BM25_CORPUS_PATH = "/home/ouar/projects/agents/X_capstone_projects/DandDCapstone/bm25_corpus.pkl"

with open(BM25_CORPUS_PATH, "rb") as f:
    corpus = pickle.load(f)

docs = corpus["documents"]
ids = corpus["ids"]

for i, d in enumerate(docs):
    if "Adult Red Dragon" in d:
        print(f"Index {i} | ID: {ids[i]} | Content: {d[:200].replace('\n', ' ')}")
