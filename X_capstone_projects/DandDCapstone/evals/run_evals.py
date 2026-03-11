import json
import asyncio
import chromadb
import sqlite3
import sys
import os
import time
from typing import List, Dict
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

# Setup paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(SCRIPT_DIR, "..", ".."))

# Create logger
class Logger(object):
    def __init__(self, filename="eval_output_log.txt"):
        self.terminal = sys.stdout
        self.log = open(filename, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

sys.stdout = Logger(os.path.join(SCRIPT_DIR, "eval_output_log.txt"))
from core_agent import CoreAgent

DB_PATH = os.path.join(os.path.dirname(SCRIPT_DIR), "chroma_db")
SQLITE_PATH = os.path.join(os.path.dirname(SCRIPT_DIR), "parent_chunks.db")
DATASET_PATH = os.path.join(SCRIPT_DIR, "golden_dataset.json")

# --- LLM Judge Models ---
class JudgeVerdict(BaseModel):
    thoughts: str = Field(default="", description="Brief reasoning about whether the answer is present.")
    exact_quote: str = Field(default="", description="If the answer is found, copy the EXACT verbatim sentence from the chunk that contains the facts. If not found, output 'N/A'.")
    answer_found: bool = Field(description="True if the retrieved context contains enough info to answer the question.")
    rank_found: int = Field(description="1-indexed rank of the first chunk that contains the answer. Use -1 if not found.")

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

print("Initializing LLM Judge (phi-agent)...", flush=True)
phi_kwargs = CoreAgent.load_litellm_kwargs_from_config("phi-agent")
judge = CoreAgent(
    agent_id="eval_judge",
    system_prompt=(
        "You are a strict D&D rules evaluation judge. "
        "Your ONLY job is to determine whether the retrieved text chunks contain enough information "
        "to correctly answer the given question. Be strict — the chunk must actually contain the answer, "
        "not just be vaguely related. Return answer_found=True only if you're confident. "
        "CRITICAL: If you find the answer, you MUST extract the exact, verbatim quote from the text that proves it. "
        "Do not paraphrase. If you cannot find a verbatim quote that answers the question, you must set answer_found=False."
    ),
    litellm_kwargs=phi_kwargs,
    response_model=JudgeVerdict,
    one_shot=True  # No memory between questions — each eval is independent
)

def get_context_for_rank(results, rank: int, windowing: bool = False) -> str:
    """Builds the full context string for a given rank, including the parent chunk."""
    doc = results['documents'][0][rank]
    meta = results['metadatas'][0][rank]
    chunk_id = results['ids'][0][rank]

    # Start with the child chunk
    context = f"[Chunk {rank+1}]:\n{doc}"

    # Add raw table content if available
    if meta.get('type') == 'table' and 'table_content' in meta:
        context += f"\n[Raw Table]:\n{meta['table_content']}"

    # Fetch and append the parent chunk
    parent_id = "_".join(chunk_id.split("_")[:3])
    
    if windowing:
        # Identify preceding and succeeding internal chunk indices if possible
        parts = parent_id.split("_")
        if len(parts) >= 3 and parts[2].isdigit():
            idx = int(parts[2])
            base_id = "_".join(parts[:2])
            parent_ids = [f"{base_id}_{idx-1}", parent_id, f"{base_id}_{idx+1}"]
        else:
            parent_ids = [parent_id]

        full_text = []
        for pid in parent_ids:
            parent_cursor.execute("SELECT content FROM parent_chunks WHERE id=?", (pid,))
            row = parent_cursor.fetchone()
            if row:
                full_text.append(row[0][:2000])

        if full_text:
            context += f"\n[Windowed Section Context]:\n...\n" + "\n...\n".join(full_text)
    else:    
        parent_cursor.execute("SELECT content FROM parent_chunks WHERE id=?", (parent_id,))
        parent_row = parent_cursor.fetchone()
        if parent_row:
            context += f"\n[Full Section Context]:\n{parent_row[0][:2000]}"

    return context

async def evaluate_retrieval_llm(dataset: List[Dict], dataset_name: str, k: int = 10):
    """LLM-as-Judge evaluation: uses phi4 to determine if retrieved chunks answer the question."""
    print(f"\n--- Running LLM-as-Judge Evaluation (K={k}, Judge: phi-agent) ---")

    total_mrr = 0.0
    total_recall = 0
    total_windowed = 0
    num_queries = len(dataset)

    for i, item in enumerate(dataset):
        query = item['question']
        expected_answer = item.get('expected_answer', '')

        print(f"\nQ{i+1}: {query}")

        # 1. Embed and retrieve
        query_embedding = encoder.encode(query, normalize_embeddings=True).tolist()
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            include=['documents', 'metadatas']
        )

        # 3. Parallel Speculative Execution
        async def check_chunk(rank, is_window):
            context = get_context_for_rank(results, rank, windowing=is_window)
            if is_window:
                judge_prompt = (
                    f"Question: {query}\n"
                    f"Expected Answer: {expected_answer}\n\n"
                    f"Retrieved Expanded Window (from Rank {rank+1}):\n{context[:6000]}\n\n"
                    f"Does this expanded context contain enough information to correctly answer the question above? "
                    f"Set answer_found=true only if the answer is clearly present and complete. "
                    f"Set rank_found={rank+1} if found, otherwise -1."
                )
                topic_pre = "Window"
            else:
                judge_prompt = (
                    f"Question: {query}\n"
                    f"Expected Answer: {expected_answer}\n\n"
                    f"Retrieved Chunk (Rank {rank+1}):\n{context[:3000]}\n\n"
                    f"Does this specific chunk contain enough information to correctly answer the question above? "
                    f"Set answer_found=true only if the answer is clearly present here. "
                    f"Set rank_found={rank+1} if found, otherwise -1."
                )
                topic_pre = "Chunk"

            t_start = time.time()
            try:
                verdict = await judge.ask(judge_prompt)
            except Exception as e:
                print(f"  ⚠️  Judge error on {topic_pre} {rank+1}: {e}", flush=True)
                verdict = JudgeVerdict(thoughts="error", answer_found=False, rank_found=-1)
            
            elapsed = time.time() - t_start
            
            meta = results['metadatas'][0][rank]
            chunk_topic = meta.get('parent_summary', meta.get('table_summary', ''))[:80]
            status = f"✅ FOUND {'VIA WINDOWING' if is_window else 'HERE'}" if verdict.answer_found else f"❌ Not in this {topic_pre.lower()}"
            
            print(f"  [{topic_pre} {rank+1}] {status} | {elapsed:.1f}s", flush=True)
            if chunk_topic:
                print(f"    Topic: {chunk_topic}", flush=True)
            if verdict.thoughts:
                print(f"    Judge: {verdict.thoughts[:120]}", flush=True)
            if verdict.answer_found:
                print(f"    Judge Evidence: \"{verdict.exact_quote}\"", flush=True)
                
            return {"rank": rank, "is_window": is_window, "verdict": verdict}

        found_rank = -1
        used_windowing = False
        best_verdict = None

        # Sequentially check Chunk 1, Window 1, Chunk 2, Window 2, etc.
        # This prevents Ollama from queuing tasks and blocking early exits!
        for rank in range(min(k, len(results['documents'][0]))):
            # 1. Check Raw Chunk
            task_res = await check_chunk(rank, is_window=False)
            if task_res['verdict'].answer_found:
                found_rank = task_res['rank'] + 1
                used_windowing = task_res['is_window']
                best_verdict = task_res['verdict']
                break
                
            # 2. Check Windowed Chunk
            task_res = await check_chunk(rank, is_window=True)
            if task_res['verdict'].answer_found:
                found_rank = task_res['rank'] + 1
                used_windowing = task_res['is_window']
                best_verdict = task_res['verdict']
                break

        if found_rank != -1:
            mrr = 1.0 / found_rank
            total_mrr += mrr
            total_recall += 1
            if used_windowing:
                print(f"  ✅ SUCCESS (Windowing): Answer at rank {found_rank} (MRR+: {mrr:.2f})")
                print(f"     Reasoning: {best_verdict.thoughts[:120]}")
                total_windowed += 1
            else:
                print(f"  ✅ SUCCESS (Judge): Answer at rank {found_rank} (MRR+: {mrr:.2f})")
                print(f"     Reasoning: {best_verdict.thoughts[:120]}")
        else:
            print(f"  ❌ FAILURE (Judge): No valid chunk found in top {k} searches.", flush=True)

    # 5. Final Report
    avg_mrr = total_mrr / num_queries
    recall_rate = total_recall / num_queries

    print("\n" + "="*45)
    print(f"📊 EVALUATION METRICS: {dataset_name}")
    print("="*45)
    print(f"Total Queries   : {num_queries}")
    print(f"Judge Model     : phi-agent (ollama/phi4)")
    print(f"MRR+            : {avg_mrr:.4f}  (Target: ~1.0) *Window-Inclusive")
    print(f"Recall@{k}      : {recall_rate*100:.1f}%")
    print(f"Overall Success : ({recall_rate*100:.2f}%) {total_recall} / {num_queries} (including windowing)")
    print(f"Windowing Hits  : {total_windowed} fallback retrievals")
    print(f"Judge API Cost  : ${judge.cost:.4f} ({judge.tokens_used} tokens)")
    print("="*45)

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv(override=True)
    
    datasets = [
        "golden_dataset_original.json",
        "golden_dataset_gemini.json",
        "golden_dataset_grok.json",
        "golden_dataset_phi.json"
    ]
    
    for ds_name in datasets:
        ds_path = os.path.join(SCRIPT_DIR, ds_name)
        if os.path.exists(ds_path):
            log_msg = f"\n{'='*60}\n🚀 EVALUATING DATASET: {ds_name}\n{'='*60}\n"
            print(log_msg)
            with open("eval_output_log.txt", "a") as f:
                f.write(log_msg)
                
            with open(ds_path, 'r') as f:
                golden_dataset = json.load(f)
            asyncio.run(evaluate_retrieval_llm(golden_dataset, ds_name, k=10))
        else:
            log_msg = f"\n⚠️ Dataset not found: {ds_path}\n"
            print(log_msg)
            with open("eval_output_log.txt", "a") as f:
                f.write(log_msg)
