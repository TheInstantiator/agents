import re
import json
import asyncio
import chromadb
import sqlite3
import sys
import os
import time
import pickle
from typing import List, Dict, Optional
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi

# Setup paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(SCRIPT_DIR, "..", ".."))

MODEL_NAME = "phi-agent"
ANSWER_MODEL_NAME = "phi-agent"

# Create logger
class Logger(object):
    def __init__(self, filename="eval_pre_release_output_log.txt"):
        self.terminal = sys.stdout
        self.log = open(filename, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

sys.stdout = Logger(os.path.join(SCRIPT_DIR, "eval_pre_release_output_log.txt"))
from core_agent import CoreAgent

DB_PATH = os.path.join(os.path.dirname(SCRIPT_DIR), "chroma_db")
SQLITE_PATH = os.path.join(os.path.dirname(SCRIPT_DIR), "parent_chunks.db")

DEBUG_MODE = True
DEBUG_LOG_PATH = os.path.join(SCRIPT_DIR, "eval_pre_release_debug_log.txt")
debug_log_file = open(DEBUG_LOG_PATH, "w") if DEBUG_MODE else None

def debug_print(*args, **kwargs):
    if debug_log_file:
        print(*args, file=debug_log_file, **kwargs)
        debug_log_file.flush()

# --- Models ---
class RouterDecision(BaseModel):
    scope: str = Field(description="'BROAD' or 'SPECIFIC'")
    category: str = Field(description="The D&D category, e.g., 'spells', 'monsters', 'rules', 'classes', 'items', 'lore', or 'other'")
    exact_match_keywords: list[str] = Field(default_factory=list, description="List of 1-3 highly specific exact phrases to search in the text (e.g., 'Adult Red Dragon'). Leave empty if none.")
    hyde_excerpt: str = Field(description="A fake 3-sentence excerpt from the official D&D rulebook that sounds like it would answer the player's question.")

class RewrittenQuery(BaseModel):
    thoughts: str = Field(description="Internal reasoning about clarifying the question.")
    rewritten_question: str = Field(description="The final rewritten question demanding specific base stats.")

class GeneratedAnswer(BaseModel):
    thoughts: str = Field(description="Internal reasoning to formulate the answer based strictly on the context.")
    final_answer: str = Field(description="The final concise answer to the question.")

class AuditResult(BaseModel):
    thoughts: str = Field(description="Reasoning about what information is still missing from the context.")
    is_complete: bool = Field(description="True if the provided context is sufficient to fully answer the question.")
    missing_info_keywords: list[str] = Field(default_factory=list, description="Keywords or phrases to search for specifically to fill the gaps.")

class AnswerEvalVerdict(BaseModel):
    thoughts: str = Field(description="Detailed reasoning comparing the Generated Answer to the Expected Answer.")
    is_correct: bool = Field(description="True if the generated answer is factually correct given the expected answer.")

# --- Global Setup ---
print("Initializing Vector Database connection...", flush=True)
client = chromadb.PersistentClient(path=DB_PATH)
try:
    collection = client.get_collection(name="dnd_rules_multi_v2")
except Exception as e:
    print(f"Error accessing collection: {e}. Has build_rag_db.py finished?", flush=True)
    exit()

print("Initializing SQLite (Parent Chunk Store)...", flush=True)
parent_db = sqlite3.connect(SQLITE_PATH)
parent_cursor = parent_db.cursor()

print("Loading Local Native Encoder (BAAI/bge-large-en-v1.5)...", flush=True)
encoder = SentenceTransformer("BAAI/bge-large-en-v1.5", device="cpu")

print("Loading Cross-Encoder Re-Ranker (ms-marco-MiniLM-L-6-v2)...", flush=True)
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device="cpu")

# --- BM25 Index ---
BM25_CORPUS_PATH = os.path.join(os.path.dirname(SCRIPT_DIR), "bm25_corpus.pkl")
bm25_index = None
bm25_corpus_docs = []
bm25_corpus_metas = []
bm25_corpus_ids = []

def simple_tokenize(text: str) -> List[str]:
    if not text: return []
    return re.findall(r'\w+', text.lower())

if os.path.exists(BM25_CORPUS_PATH):
    print("Loading BM25 corpus from disk...", flush=True)
    with open(BM25_CORPUS_PATH, "rb") as _f:
        _corpus = pickle.load(_f)
    bm25_corpus_docs = _corpus["documents"]
    bm25_corpus_metas = _corpus["metadatas"]
    bm25_corpus_ids = _corpus["ids"]
    _tokenised = []
    for doc, meta in zip(bm25_corpus_docs, bm25_corpus_metas):
        enriched_text = f"{meta.get('parent_summary', '')} {meta.get('table_summary', '')} {doc}"
        _tokenised.append(simple_tokenize(enriched_text))
    bm25_index = BM25Okapi(_tokenised)
else:
    print(f"  ⚠️  BM25 corpus not found.", flush=True)

phi_kwargs = CoreAgent.load_litellm_kwargs_from_config(MODEL_NAME)

router = CoreAgent(
    agent_id="master_router",
    system_prompt="You are the Master Routing AI for a D&D 5e offline rules engine. Transform natural language questions into search strategies.",
    litellm_kwargs=phi_kwargs,
    response_model=RouterDecision,
    one_shot=True
)

rephrase_agent = CoreAgent(
    agent_id="rephrase_agent",
    system_prompt="Rewrite player questions to explicitly target base, non-magical versions of items/monsters/rules.",
    litellm_kwargs=phi_kwargs,
    response_model=RewrittenQuery,
    one_shot=True
)

answer_agent = CoreAgent(
    agent_id="answer_generator",
    system_prompt="Answer player questions strictly using the provided context. If the answer is not in the text, say 'I don't know'.",
    litellm_kwargs=phi_kwargs,
    response_model=GeneratedAnswer,
    one_shot=True
)

audit_agent = CoreAgent(
    agent_id="audit_agent",
    system_prompt="Compare the context against the question. Identify missing facts. If facts are missing, suggest specific keywords for a second search.",
    litellm_kwargs=phi_kwargs,
    response_model=AuditResult,
    one_shot=True
)

answer_judge = CoreAgent(
    agent_id="answer_judge",
    system_prompt="You are an impartial, strict evaluator. You will be given a Question, an Expected Answer, and a Generated Answer. Your ONLY job is to determine if the Generated Answer contains the facts present in the Expected Answer. Treat the Expected Answer as the absolute, unquestionable truth. Do NOT use your own knowledge to correct the Expected Answer. If the Expected Answer says 18 gives +4, and the Generated Answer says +4, it is CORRECT.",
    litellm_kwargs=phi_kwargs,
    response_model=AnswerEvalVerdict,
    one_shot=True
)

def get_context_for_rank(results, rank: int, windowing: bool = False, aggregate_mode: bool = False) -> str:
    doc = results['documents'][0][rank]
    meta = results['metadatas'][0][rank]
    chunk_id = results['ids'][0][rank]
    parent_id = meta.get('parent_id')
    if not parent_id:
        parts = chunk_id.split("_")
        parent_id = "_".join(parts[:3])

    header_parts = []
    if meta.get('parent_summary'): header_parts.append(f"[Section Overview]: {meta['parent_summary']}")
    if meta.get('table_summary'): header_parts.append(f"[Table Focus]: {meta['table_summary']}")
    
    context = "\n".join(header_parts) + f"\n[Chunk {rank+1}]:\n{doc}"
    if meta.get('type') in ['table', 'table_row_prose'] and 'table_content' in meta:
        context += f"\n[Full Table]:\n{meta['table_content']}"

    # If we are building a massive prompt with 20 chunks, skip appending the 6000+ char parent text
    if aggregate_mode:
        return context

    if windowing:
        parent_parts = parent_id.split("_")
        if len(parent_parts) >= 3 and parent_parts[2].isdigit():
            idx = int(parent_parts[2])
            base_id = "_".join(parent_parts[:2])
            parent_ids = [f"{base_id}_{idx-1}", parent_id, f"{base_id}_{idx+1}"]
        else: parent_ids = [parent_id]
        
        full_text = []
        for pid in parent_ids:
            parent_cursor.execute("SELECT content FROM parent_chunks WHERE id=?", (pid,))
            row = parent_cursor.fetchone()
            if row: full_text.append(row[0][:1500]) # Slightly reduced to save tokens
        if full_text: context += f"\n[Windowed Context]:\n...\n" + "\n...\n".join(full_text)
    else:
        parent_cursor.execute("SELECT content FROM parent_chunks WHERE id=?", (parent_id,))
        row = parent_cursor.fetchone()
        if row: context += f"\n[Full Context]:\n{row[0][:1500]}"
    return context

async def hybrid_search(query: str, decision: RouterDecision, k: int):
    search_text = f"{query} {decision.hyde_excerpt}"
    query_embedding = encoder.encode(search_text, normalize_embeddings=True).tolist()
    
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k*3,
        include=['documents', 'metadatas']
    )
    docs = results['documents'][0]
    metas = results['metadatas'][0]
    ids = results['ids'][0]

    blacklist = {"cost", "damage", "ac", "stats", "hp", "hit", "points", "price", "modifier", "score", "level", "dc", "weight", "type", "size", "speed", "alignment", "languages", "properties", "standard", "base"}
    filtered_keywords = [kw for kw in decision.exact_match_keywords if kw.lower() not in blacklist and not kw.lower().startswith("level ")]

    exact_match_indices = set()
    existing_ids_set = set(ids)

    if bm25_index and filtered_keywords:
        bm25_query_tokens = simple_tokenize(" ".join(filtered_keywords))
        bm25_scores = bm25_index.get_scores(bm25_query_tokens)
        top_bm25_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:k * 3]

        for bm25_idx in top_bm25_indices:
            if bm25_scores[bm25_idx] <= 0: break
            doc_id = bm25_corpus_ids[bm25_idx]
            if doc_id not in existing_ids_set:
                docs.append(bm25_corpus_docs[bm25_idx])
                metas.append(bm25_corpus_metas[bm25_idx])
                ids.append(doc_id)
                exact_match_indices.add(len(docs) - 1)
                existing_ids_set.add(doc_id)
            else:
                try: exact_match_indices.add(ids.index(doc_id))
                except ValueError: pass

    pairs = [[query, doc] for doc in docs]
    scores = cross_encoder.predict(pairs).tolist()

    for idx in range(len(docs)):
        phrasal_count = 0
        doc_low = docs[idx].lower()
        for kw in filtered_keywords:
            if kw.lower() in doc_low: phrasal_count += 1
        
        if phrasal_count > 0:
            scores[idx] = (phrasal_count * 100.0) + (scores[idx] / 100.0)
        elif idx in exact_match_indices:
            scores[idx] = 50.0 + (scores[idx] / 100.0)

    scored = sorted(zip(scores, docs, metas, ids), key=lambda x: x[0], reverse=True)[:k]
    results['documents'][0] = [s[1] for s in scored]
    results['metadatas'][0] = [s[2] for s in scored]
    results['ids'][0] = [s[3] for s in scored]
    return results

async def run_pre_release_eval(dataset: List[Dict], dataset_name: str):
    print(f"\n--- PRE-RELEASE AGENTIC RAG EVALUATION: {dataset_name} ---")
    total_correct = 0

    for i, item in enumerate(dataset):
        query = item['question']
        expected_answer = item.get('expected_answer', '')
        print(f"\nQ{i+1}: {query}")

        # 1. ROUTE & DECOMPOSE
        decision = await router.ask(query)
        k = 10 if decision.scope == "SPECIFIC" else 20
        print(f"  [Router] Scope: {decision.scope} | Cat: {decision.category} | K: {k}")
        print(f"  [Router] Exact Match Keywords: {decision.exact_match_keywords}")
        print(f"  [Router] HyDE: {decision.hyde_excerpt[:150]}...")
        
        # 2. INITIAL RETRIEVAL
        results = await hybrid_search(query, decision, k)
        print(f"\n  [Search 1] Retrieved Top {len(results['documents'][0])} chunks.")
        
        context_parts = []
        for j in range(len(results['documents'][0])):
            ctx = get_context_for_rank(results, j, windowing=False, aggregate_mode=True)
            meta = results['metadatas'][0][j]
            topic = meta.get('parent_summary', meta.get('table_summary', ''))[:100]
            # Strip newlines for single-line printing
            topic = topic.replace('\n', ' ')
            print(f"    - Rank {j+1}: {topic} (+ Neighboring Windows)")
            context_parts.append(ctx)
        
        context = "\n\n".join(context_parts)

        # 3. INTERNAL AUDIT
        audit_prompt = f"Question: {query}\n\nContext:\n{context[:60000]}\n\nDoes this context contain EVERYTHING needed to answer the question? If not, what specifically is missing?"
        audit = await audit_agent.ask(audit_prompt)
        
        if not audit.is_complete and audit.missing_info_keywords:
            print(f"\n  [Audit] INCOMPLETE context detected.")
            print(f"  [Audit] Thoughts: {audit.thoughts}")
            print(f"  [Audit] Fetching missing info with keywords: {audit.missing_info_keywords}")
            # 4. REFINEMENT SEARCH
            refine_decision = RouterDecision(
                scope="SPECIFIC",
                category=decision.category,
                exact_match_keywords=audit.missing_info_keywords,
                hyde_excerpt=audit.thoughts
            )
            refine_results = await hybrid_search(query, refine_decision, 5)
            print(f"  [Search 2] Retrieved {len(refine_results['documents'][0])} supplementary chunks.")
            refine_context_parts = []
            for j in range(len(refine_results['documents'][0])):
                ctx = get_context_for_rank(refine_results, j, windowing=False, aggregate_mode=True)
                meta = refine_results['metadatas'][0][j]
                topic = meta.get('parent_summary', meta.get('table_summary', ''))[:100].replace('\n', ' ')
                print(f"    - Refined Rank {j+1}: {topic}")
                refine_context_parts.append(ctx)
            
            refine_context = "\n\n".join(refine_context_parts)
            context += "\n\n=== ADDITIONAL CONTEXT ===\n\n" + refine_context
        else:
            print(f"\n  [Audit] COMPLETE. Context has sufficient information.")

        # 5. REPHRASE & GENERATE
        # Include a strict warning about HyDE hallucination just like we did in routed evals
        ref_q_prompt = (
            f"Question: {query}\n"
            f"Keywords: {decision.exact_match_keywords}\n"
            f"HyDE Hint (DANGEROUS: DO NOT COPY FACTS OR NUMBERS FROM THIS): {decision.hyde_excerpt}"
        )
        ref_q = await rephrase_agent.ask(ref_q_prompt)
        print(f"\n  [Rephrase] Thoughts: {ref_q.thoughts}")
        print(f"  [Rephrase] Rewritten: {ref_q.rewritten_question}")
        
        gen_prompt = (
            f"Question: {query}\n"
            f"Clarification / Rephrased Focus: {ref_q.rewritten_question}\n\n"
            f"Context:\n{context[:100000]}\n\n"
            f"Answer the Question concisely based ONLY on the context. If it's not in the context, say 'I don't know'."
        )
        answer = await answer_agent.ask(gen_prompt)
        print(f"\n  [Generator] Thoughts: {answer.thoughts[:200]}...")
        print(f"  [Generated Answer]: {answer.final_answer}")

        # 6. FINAL JUDGE (Gen vs Golden)
        eval_prompt = f"Question: {query}\nExpected: {expected_answer}\nGenerated: {answer.final_answer}\nIs it correct?"
        verdict = await answer_judge.ask(eval_prompt)
        
        if verdict.is_correct:
            total_correct += 1
            print(f"  ✅ SUCCESS")
        else:
            print(f"  ❌ FAILED: {verdict.thoughts}")

    print(f"\nFinal Score: {total_correct}/{len(dataset)} ({(total_correct/len(dataset))*100:.1f}%)")

if __name__ == "__main__":
    from dotenv import load_dotenv
    os.chdir("/home/ouar/projects/agents/X_capstone_projects")
    load_dotenv(override=True)
    
    with open(os.path.join(SCRIPT_DIR, "golden_dataset_original.json"), 'r') as f:
        dataset = json.load(f)
    
    asyncio.run(run_pre_release_eval(dataset[:10], "Short Test Run")) # Limit to 10 for test
