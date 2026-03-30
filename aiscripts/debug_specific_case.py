import sys
import os
import json
import asyncio
import re
import chromadb
import sqlite3
import pickle
from typing import List, Dict, Tuple
from pydantic import BaseModel, Field

from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

# Set up paths relative to this script in aiscripts
AISCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
HOME_DIR = os.path.dirname(AISCRIPTS_DIR)
PROJECT_ROOT = os.path.join(HOME_DIR, "X_capstone_projects")
CAPSTONE_DIR = os.path.join(PROJECT_ROOT, "DandDCapstone")
EVALS_DIR = os.path.join(CAPSTONE_DIR, "evals")

sys.path.append(PROJECT_ROOT)
from core_agent import CoreAgent

# Paths from run_evals_pre_release.py
DB_PATH = os.path.join(CAPSTONE_DIR, "chroma_db")
SQLITE_PATH = os.path.join(CAPSTONE_DIR, "parent_chunks.db")
BM25_CORPUS_PATH = os.path.join(CAPSTONE_DIR, "bm25_corpus.pkl")

# Re-use models from run_evals_pre_release.py
class DecomposedQueries(BaseModel):
    thoughts: str
    sub_queries: list[str] = Field(default_factory=list)

class HyDEAnswer(BaseModel):
    thoughts: str
    hyde_text: str

class CritiqueResult(BaseModel):
    thoughts: str
    is_complete: bool
    missing_aspects: list[str] = Field(default_factory=list)
    follow_up_query: str = ""

class StandardAnswer(BaseModel):
    thoughts: str = Field(description="Internal reasoning process.")
    answer: str = Field(description="The concise, factual answer for the user.")

class RerankOutput(BaseModel):
    thoughts: str
    ranked_indices: list[int] = Field(description="List of integer indices ordered from most to least relevant.")

# Setup (mostly copied from run_evals_pre_release.py)
print("Initializing databases...")
client = chromadb.PersistentClient(path=DB_PATH)
collection = client.get_collection(name="dnd_rules_multi_v2")
parent_db = sqlite3.connect(SQLITE_PATH)
parent_cursor = parent_db.cursor()
encoder = SentenceTransformer("BAAI/bge-large-en-v1.5", device="cpu")

with open(BM25_CORPUS_PATH, "rb") as f:
    bm25_data = pickle.load(f)
bm25_corpus_docs = bm25_data["documents"]
bm25_corpus_metas = bm25_data["metadatas"]
bm25_corpus_ids = bm25_data["ids"]

_tokenized = [re.findall(r'\w+', (meta.get('parent_summary', '') + " " + doc).lower())
              for doc, meta in zip(bm25_corpus_docs, bm25_corpus_metas)]
bm25_index = BM25Okapi(_tokenized)

# Agents
phi_kwargs = CoreAgent.load_litellm_kwargs_from_config("phi-agent")
decomposer = CoreAgent(agent_id="decomposer", system_prompt="Break complex D&D questions into 2-4 simple, atomic sub-questions that together fully cover the original intent.", litellm_kwargs=phi_kwargs, response_model=DecomposedQueries, one_shot=True)
hyde_agent = CoreAgent(agent_id="hyde_agent", system_prompt="Generate a short, plausible excerpt from the official D&D rulebook that would perfectly answer the question.", litellm_kwargs=phi_kwargs, response_model=HyDEAnswer, one_shot=True)
critic = CoreAgent(agent_id="critic", system_prompt="You are a strict D&D rules auditor. Compare the Original Question and the provided Context with the Generated Answer. If the Context contains specific facts (like costs, damage types, or mechanics) that are missing from the Generated Answer, provide exactly ONE focused follow-up query to retrieve those specific missing facts. Do NOT use external knowledge not present in the Context.", litellm_kwargs=phi_kwargs, response_model=CritiqueResult, one_shot=True)
merger = CoreAgent(agent_id="merger", system_prompt="You are an expert editor. Combine the original answer and the supplemental answer into one clear, factual final answer. Do NOT include meta-commentary about needing more info, missing context, or being unable to find everything. Just state the facts you HAVE found.", litellm_kwargs=phi_kwargs, response_model=StandardAnswer, one_shot=True)
answer_agent = CoreAgent(agent_id="answer_generator", system_prompt="Answer strictly using the provided context from the D&D rulebook. Be precise and concise. Only provide the facts found in the context. Do NOT add phrases like 'I need more context' or 'further details may be needed'. If you found no information at all, say 'I don't know'.", litellm_kwargs=phi_kwargs, response_model=StandardAnswer, one_shot=True)
reranker = CoreAgent(agent_id="reranker", system_prompt="You are an expert search reranker. Rank the given search results by relevance to the question. Output a list of the integer indices representing the original position of each document, ordered from most relevant to least relevant.", litellm_kwargs=phi_kwargs, response_model=RerankOutput, one_shot=True)

def reciprocal_rank_fusion(vector_results: dict, bm25_results: dict, k: int = 25) -> List[Tuple]:
    scores = {}
    for top_k_ids in vector_results.get('ids', []):
        if not top_k_ids: continue
        for rank, cid in enumerate(top_k_ids):
            scores[cid] = scores.get(cid, 0) + 1.0 / (rank + 60)
    for top_k_ids in bm25_results.get('ids', []):
        if not top_k_ids: continue
        for rank, cid in enumerate(top_k_ids):
            scores[cid] = scores.get(cid, 0) + 1.0 / (rank + 60)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]

async def debug_question(original_query: str):
    print(f"\n{'='*20} DEBUGGING QUESTION {'='*20}")
    print(f"Query: {original_query}")

    # 1. Decomposition + HyDE
    print("\n--- [Decomposition + HyDE] ---")
    decomp = await decomposer.ask(original_query)
    hyde = await hyde_agent.ask(original_query)
    print(f"Sub-queries: {decomp.sub_queries}")
    print(f"HyDE Excerpt: {hyde.hyde_text[:200]}...")

    all_queries = [original_query] + decomp.sub_queries + [hyde.hyde_text]

    # 2. Vector Search
    print("\n--- [Vector Search] ---")
    all_embeddings = encoder.encode(all_queries, normalize_embeddings=True).tolist()
    vector_results = collection.query(query_embeddings=all_embeddings, n_results=20, include=['documents', 'metadatas'])
    print(f"Vector hits: {sum(len(x) for x in vector_results.get('ids', []))}")

    # 3. BM25 Search
    print("\n--- [BM25 Search] ---")
    bm25_tokens = re.findall(r'\w+', " ".join(all_queries).lower())
    bm25_scores = bm25_index.get_scores(bm25_tokens)
    top_bm25_idx = sorted(range(len(bm25_scores)), key=lambda x: bm25_scores[x], reverse=True)[:20]
    bm25_results = {
        'documents': [[bm25_corpus_docs[mi] for mi in top_bm25_idx]],
        'metadatas': [[bm25_corpus_metas[mi] for mi in top_bm25_idx]],
        'ids': [[bm25_corpus_ids[mi] for mi in top_bm25_idx]]
    }
    print(f"BM25 hits: {len(top_bm25_idx)}")

    # 4. RRF Merge
    rrf_merged = reciprocal_rank_fusion(vector_results, bm25_results, k=30)
    print(f"RRF merged to {len(rrf_merged)} chunks.")

    # 5. Reranking
    print("\n--- [Reranking] ---")
    rerank_prompt = f"Original Question: {original_query}\n\nRank these chunks from most to least relevant. Return only ordered indices (0-based):\n"
    rerank_lines = []
    for idx, (cid, _) in enumerate(rrf_merged):
        try:
            corpus_idx = bm25_corpus_ids.index(cid)
            doc = bm25_corpus_docs[corpus_idx]
        except ValueError:
            doc = "Context details unavailable for this chunk."
        rerank_lines.append(f"[{idx}] {doc[:300]}...")
    rerank_prompt += "\n".join(rerank_lines)
    
    try:
        reranked = await reranker.ask(rerank_prompt)
        print(f"Reranker thoughts: {reranked.thoughts}")
        print(f"Ranked Indices: {reranked.ranked_indices}")
    except Exception as e:
        print(f"Reranking FAILED: {e}")
        return

    valid_indices = [idx for idx in reranked.ranked_indices if 0 <= idx < len(rrf_merged)]
    top_indices = valid_indices[:12]

    # 6. Context Construction
    print("\n--- [Context Construction] ---")
    seen_parents = set()
    context_parts = []
    for rank_pos, idx in enumerate(top_indices):
        cid = rrf_merged[idx][0]
        meta_idx = bm25_corpus_ids.index(cid)
        meta = bm25_corpus_metas[meta_idx]
        doc = bm25_corpus_docs[meta_idx]
        
        parent_id = meta.get("parent_id", "")
        print(f"[Rank {rank_pos+1}] ID: {cid}, Parent: {parent_id}, Summary: {meta.get('parent_summary', 'N/A')}")
        
        chunk_context = f"[Retrieval Rank {rank_pos+1}]\nSection: {meta.get('parent_summary', 'N/A')}\nExact Excerpt: {doc}\n"
        if parent_id and parent_id not in seen_parents:
            seen_parents.add(parent_id)
            parent_cursor.execute("SELECT content FROM parent_chunks WHERE id=?", (parent_id,))
            row = parent_cursor.fetchone()
            if row and row[0]:
                chunk_context += f"\nBroader Section Context:\n{row[0][:1000]}\n"
        context_parts.append(chunk_context)

    context = "\n\n---\n\n".join(context_parts)

    # 7. Initial Answer
    print("\n--- [Initial Answer] ---")
    answer_obj = await answer_agent.ask(f"Question: {original_query}\n\nContext:\n{context}\n\nAnswer concisely using only the context.")
    print(f"Thoughts: {answer_obj.thoughts}")
    print(f"Answer: {answer_obj.answer}")

    # 8. Critique
    print("\n--- [Critique] ---")
    critique = await critic.ask(f"Original Question: {original_query}\n\nContext:\n{context}\n\nGenerated Answer: {answer_obj.answer}\n\nIs this answer complete?")
    print(f"Is Complete: {critique.is_complete}")
    print(f"Follow-up: {critique.follow_up_query}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python debug_specific_case.py 'Your question here' or index relative to dataset")
        sys.exit(1)
    
    query = sys.argv[1]
    # Check if it's an integer index + dataset
    if query.isdigit() and len(sys.argv) > 2:
        idx = int(query)
        dataset_file = sys.argv[2]
        dataset_path = os.path.join(EVALS_DIR, dataset_file)
        with open(dataset_path, 'r') as f:
            dataset = json.load(f)
        query = dataset[idx]['question']
        print(f"Loaded question from {dataset_file} index {idx}: {query}")

    asyncio.run(debug_question(query))
