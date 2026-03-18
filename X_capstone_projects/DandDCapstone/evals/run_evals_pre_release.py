import re
import json
import asyncio
import chromadb
import sqlite3
import sys
import os
import pickle
from typing import List, Dict, Tuple
from pydantic import BaseModel, Field

from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

# ========================= PATHS =========================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(SCRIPT_DIR, "..", ".."))
from core_agent import CoreAgent

DB_PATH = os.path.join(os.path.dirname(SCRIPT_DIR), "chroma_db")
SQLITE_PATH = os.path.join(os.path.dirname(SCRIPT_DIR), "parent_chunks.db")
BM25_CORPUS_PATH = os.path.join(os.path.dirname(SCRIPT_DIR), "bm25_corpus.pkl")

# ========================= MODELS =========================
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

class AnswerEvalVerdict(BaseModel):
    thoughts: str = Field(description="Reasoning for the verdict.")
    is_correct: bool = Field(description="Whether the generated answer matches the expected answer factually.")

# ========================= SETUP =========================
print("Initializing databases...")
client = chromadb.PersistentClient(path=DB_PATH)
collection = client.get_collection(name="dnd_rules_multi_v2")

parent_db = sqlite3.connect(SQLITE_PATH)
parent_cursor = parent_db.cursor()

encoder = SentenceTransformer("BAAI/bge-large-en-v1.5", device="cpu")

# Load BM25
with open(BM25_CORPUS_PATH, "rb") as f:
    bm25_data = pickle.load(f)
bm25_corpus_docs = bm25_data["documents"]
bm25_corpus_metas = bm25_data["metadatas"]
bm25_corpus_ids = bm25_data["ids"]

_tokenized = [re.findall(r'\w+', (meta.get('parent_summary', '') + " " + doc).lower())
              for doc, meta in zip(bm25_corpus_docs, bm25_corpus_metas)]
bm25_index = BM25Okapi(_tokenized)

# ========================= AGENTS =========================
phi_kwargs = CoreAgent.load_litellm_kwargs_from_config("phi-agent")

decomposer = CoreAgent(
    agent_id="decomposer",
    system_prompt="Break complex D&D questions into 2-4 simple, atomic sub-questions that together fully cover the original intent.",
    litellm_kwargs=phi_kwargs,
    response_model=DecomposedQueries,
    one_shot=True
)

hyde_agent = CoreAgent(
    agent_id="hyde_agent",
    system_prompt="Generate a short, plausible excerpt from the official D&D rulebook that would perfectly answer the question.",
    litellm_kwargs=phi_kwargs,
    response_model=HyDEAnswer,
    one_shot=True
)

critic = CoreAgent(
    agent_id="critic",
    # system_prompt="You are a strict QA tester. Compare the Original Question with the Generated Answer. Only ask a follow-up query if information EXPLICITLY requested in the Original Question is missing. Do NOT ask for additional lore, stats, or mechanics that were not directly requested. If the question is fully answered, mark it as complete.",
    # system_prompt="You are a strict D&D rules auditor. Compare the original question with the generated answer. If anything is missing or incomplete (like missing costs, damage types, or specific mechanics mentioned in the rules), provide exactly ONE focused follow-up query to find that missing data.  If the question is good and based on your knowledge of D and D you could add some additional pertinent information to the answer to add interesting facts.  This is encouraged.",
    system_prompt="You are a strict D&D rules auditor. Compare the Original Question and the provided Context with the Generated Answer. If the Context contains specific facts (like costs, damage types, or mechanics) that are missing from the Generated Answer, provide exactly ONE focused follow-up query to retrieve those specific missing facts. Do NOT use external knowledge not present in the Context.",
    litellm_kwargs=phi_kwargs,
    response_model=CritiqueResult,
    one_shot=True
)

merger = CoreAgent(
    agent_id="merger",
    system_prompt="You are an expert editor. Combine the original answer and the supplemental answer into one clear, factual final answer. Do NOT include meta-commentary about needing more info, missing context, or being unable to find everything. Just state the facts you HAVE found.",
    litellm_kwargs=phi_kwargs,
    response_model=StandardAnswer,
    one_shot=True
)

answer_agent = CoreAgent(
    agent_id="answer_generator",
    system_prompt="Answer strictly using the provided context from the D&D rulebook. Be precise and concise. Only provide the facts found in the context. Do NOT add phrases like 'I need more context' or 'further details may be needed'. If you found no information at all, say 'I don't know'.",
    litellm_kwargs=phi_kwargs,
    response_model=StandardAnswer,
    one_shot=True
)

answer_judge = CoreAgent(
    agent_id="answer_judge",
    system_prompt="You are an impartial judge. Compare the Generated Answer to the Expected Answer. Determine if the Generated Answer is factually correct. You should mark it as correct if it matches the Expected Answer OR if it accurately answers the question based on the provided Context (which may contain updated rules compared to the Expected Answer).",
    litellm_kwargs=phi_kwargs,
    response_model=AnswerEvalVerdict,
    one_shot=True
)

reranker = CoreAgent(
    agent_id="reranker",
    system_prompt="You are an expert search reranker. Rank the given search results by relevance to the question. Output a list of the integer indices representing the original position of each document, ordered from most relevant to least relevant.",
    litellm_kwargs=phi_kwargs,
    response_model=RerankOutput,
    one_shot=True
)

# ========================= HELPERS =========================
def reciprocal_rank_fusion(vector_results: dict, bm25_results: dict, k: int = 25) -> List[Tuple]:
    scores = {}
    # Vector results (could be multiple lists if multiple sub-queries)
    for top_k_ids in vector_results.get('ids', []):
        if not top_k_ids: continue
        for rank, cid in enumerate(top_k_ids):
            scores[cid] = scores.get(cid, 0) + 1.0 / (rank + 60)
    
    # BM25 results
    for top_k_ids in bm25_results.get('ids', []):
        if not top_k_ids: continue
        for rank, cid in enumerate(top_k_ids):
            scores[cid] = scores.get(cid, 0) + 1.0 / (rank + 60)
    
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]


class MultiLogger:
    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()

    def flush(self):
        for f in self.files:
            f.flush()

# ========================= MAIN EVAL =========================
async def run_pre_release_eval(dataset: List[Dict], dataset_name: str, debug_log_file):
    print(f"\n=== PRE-RELEASE EVALUATION: {dataset_name} ===\n")
    total_correct = 0
    total_questions = len(dataset)
    
    # Store failure details for debug log
    for i, item in enumerate(dataset):
        try:
            original_query = item['question']
            expected_answer = item.get('expected_answer', '')

            print(f"\nQ{i+1}: {original_query}")
            print(f"Expected Answer: {expected_answer}\n")

            # 1. Decomposition + HyDE
            decomp = await decomposer.ask(original_query)
            hyde = await hyde_agent.ask(original_query)
            
            print(f"  [Decomposer] Generated {len(decomp.sub_queries)} sub-queries: {decomp.sub_queries}")
            print(f"  [HyDE] Excerpt: {hyde.hyde_text[:150]}...")

            all_queries = [original_query] + decomp.sub_queries + [hyde.hyde_text]

            # 2. Vector Search using specific encoder
            all_embeddings = encoder.encode(all_queries, normalize_embeddings=True).tolist()
            vector_results = collection.query(
                query_embeddings=all_embeddings,
                n_results=20,
                include=['documents', 'metadatas']
            )

            # 3. BM25 Search
            bm25_tokens = re.findall(r'\w+', " ".join(all_queries).lower())
            bm25_scores = bm25_index.get_scores(bm25_tokens)
            top_bm25_idx = sorted(range(len(bm25_scores)), key=lambda x: bm25_scores[x], reverse=True)[:20]

            bm25_results = {
                'documents': [[bm25_corpus_docs[mi] for mi in top_bm25_idx]],
                'metadatas': [[bm25_corpus_metas[mi] for mi in top_bm25_idx]],
                'ids': [[bm25_corpus_ids[mi] for mi in top_bm25_idx]]
            }

            # 4. RRF Merge
            rrf_merged = reciprocal_rank_fusion(vector_results, bm25_results, k=30)
            
            total_vec = sum(len(x) for x in vector_results.get('ids', []))
            total_bm25 = sum(len(x) for x in bm25_results.get('ids', []))
            print(f"  [Retrieval] Vector returned {total_vec} hits. BM25 returned {total_bm25} hits. RRF merged to top {len(rrf_merged)} chunks.")

            # 5. LLM Reranking (phi-4)
            rerank_prompt = f"Original Question: {original_query}\n\nRank these chunks from most to least relevant. Return only ordered indices (0-based):\n"
            rerank_lines = []
            for idx, (cid, _) in enumerate(rrf_merged):
                try:
                    corpus_idx = bm25_corpus_ids.index(cid)
                    doc = bm25_corpus_docs[corpus_idx]
                except ValueError:
                    doc = "Context details unavailable for this chunk."
                rerank_lines.append(f"[{idx}] {doc[:500]}...")
                
            rerank_prompt += "\n".join(rerank_lines)
            
            reranked = await reranker.ask(rerank_prompt)
            # Filter indices to ensure they are within the valid range of rrf_merged
            valid_indices = [idx for idx in reranked.ranked_indices if 0 <= idx < len(rrf_merged)]
            top_indices = valid_indices[:12]
            print(f"  [Reranker] Filtered and kept the top {len(top_indices)} chunks for context window.")

            # 6. Window Expansion with Deduplication
            seen_parents = set()
            context_parts = []
            for rank_pos, idx in enumerate(top_indices):
                cid = rrf_merged[idx][0]
                # Find corresponding metadata
                try:
                    meta_idx = bm25_corpus_ids.index(cid)
                    meta = bm25_corpus_metas[meta_idx]
                    doc = bm25_corpus_docs[meta_idx]
                except ValueError:
                    continue
                
                parent_id = meta.get("parent_id", "")
                
                chunk_context = f"[Retrieval Rank {rank_pos+1}]\n"
                if meta.get('parent_summary'): chunk_context += f"Section: {meta['parent_summary']}\n"
                if meta.get('table_summary'): chunk_context += f"Table Focus: {meta['table_summary']}\n"
                
                # Strongly highlight the precise chunk that ranked highly
                chunk_context += f"Exact Excerpt: {doc}\n"
                
                # Append full parent section only once per parent_id to prevent redundant token explosion
                if parent_id and parent_id not in seen_parents:
                    seen_parents.add(parent_id)
                    parent_cursor.execute("SELECT content FROM parent_chunks WHERE id=?", (parent_id,))
                    row = parent_cursor.fetchone()
                    if row and row[0]:
                        chunk_context += f"\nBroader Section Context:\n{row[0][:1500]}\n"
                        
                context_parts.append(chunk_context)

            context = "\n\n---\n\n".join(context_parts)

            # 7. Generate Initial Answer
            answer_obj = await answer_agent.ask(
                f"Question: {original_query}\n\nContext:\n{context}\n\nAnswer concisely using only the context."
            )
            initial_thoughts = answer_obj.thoughts
            initial_answer_text = answer_obj.answer

            # 8. Critic + Single Refinement Pass
            critique = await critic.ask(
                f"Original Question: {original_query}\n\nContext:\n{context}\n\nGenerated Answer: {initial_answer_text}\n\nIs this answer complete based on the Question and Context?"
            )

            final_thoughts = initial_thoughts
            final_answer_text = initial_answer_text

            if not critique.is_complete and critique.follow_up_query.strip():
                YELLOW = "\033[0;33m"
                RESET = "\033[0m"
                print(f"  {YELLOW}[Critic Thoughts]: {critique.thoughts}{RESET}")
                print(f"  {YELLOW}[Critic Follow-up]: {critique.follow_up_query}{RESET}")

                # One additional retrieval
                refine_embeddings = encoder.encode([critique.follow_up_query], normalize_embeddings=True).tolist()
                refine_vector = collection.query(
                    query_embeddings=refine_embeddings,
                    n_results=10,
                    include=['documents', 'metadatas']
                )

                refine_context_parts = []
                for r_idx in range(len(refine_vector.get('ids', [[]])[0][:6])):
                    doc = refine_vector['documents'][0][r_idx]
                    meta = refine_vector['metadatas'][0][r_idx]
                    refine_context_parts.append(f"Excerpt:\n{doc}\nSection Overview: {meta.get('parent_summary', '')}\n")
                
                refine_context = "\n\n".join(refine_context_parts)

                supplemental_obj = await answer_agent.ask(
                    f"Original Question: {original_query}\nFollow-up: {critique.follow_up_query}\n\nContext:\n{refine_context}\n\nProvide the missing information."
                )

                # Merge old and new
                merged = await merger.ask(
                    f"Original Answer: {initial_answer_text}\n\nNew Information: {supplemental_obj.answer}\n\nCombine them into one clear, complete final answer."
                )
                final_thoughts = merged.thoughts
                final_answer_text = merged.answer

            # Visual formatting for the final answer
            GREEN_BOLD = "\033[1;32m"
            CYAN = "\033[0;36m"
            RESET = "\033[0m"
            
            print(f"\n{CYAN}THOUGHTS:{RESET}")
            print(f"{CYAN}{final_thoughts}{RESET}")
            print(f"\n{GREEN_BOLD}{'='*60}")
            print(f"FINAL ANSWER:")
            print(f"{final_answer_text}")
            print(f"{'='*60}{RESET}\n")

            # 9. Final Judge
            verdict = await answer_judge.ask(
                f"Question: {original_query}\nContext:\n{context}\n\nExpected Answer: {expected_answer}\nGenerated Answer: {final_answer_text}\nIs it correct?"
            )

            if verdict.is_correct:
                total_correct += 1
                print(f"  ✅ Correct\n  [Judge]: {verdict.thoughts}")
            else:
                print(f"  ❌ Incorrect\n  [Judge]: {verdict.thoughts}")
                # Log failure to debug log
                debug_info = f"Dataset: {dataset_name}\nQuestion: {original_query}\nExpected: {expected_answer}\nGenerated: {final_answer_text}\nJudge Thoughts: {verdict.thoughts}\n"
                debug_log_file.write(f"{'#'*40}\n{debug_info}\n")
                debug_log_file.flush()

        except Exception as e:
            print(f"  💥 ERROR processing question {i+1}: {e}")
            query_text = item.get('question', 'Unknown') if isinstance(item, dict) else 'Unknown'
            debug_log_file.write(f"{'#'*40}\nERROR on Question {i+1}: {query_text}\nException: {e}\n")
            debug_log_file.flush()
            continue

    score_pct = (total_correct / total_questions * 100) if total_questions > 0 else 0
    print(f"\nFinal Score for {dataset_name}: {total_correct}/{total_questions} ({score_pct:.1f}%)")
    return total_correct, total_questions

if __name__ == "__main__":
    FILES_TO_RUN = [
        "golden_dataset_original.json",
        "golden_dataset_gemini.json",
        "golden_dataset_grok.json",
        "golden_dataset_phi.json"
    ]
    
    OUTPUT_LOG_PATH = os.path.join(SCRIPT_DIR, "eval_pre_release_output_log.txt")
    SUM_LOG_PATH = os.path.join(SCRIPT_DIR, "eval_pre_release_sum_log.txt")
    DEBUG_LOG_PATH = os.path.join(SCRIPT_DIR, "eval_pre_release_debug_log.txt")
    
    # Open files
    out_f = open(OUTPUT_LOG_PATH, "w")
    sum_f = open(SUM_LOG_PATH, "w")
    debug_f = open(DEBUG_LOG_PATH, "w")
    
    # Redirect stdout to both console and output log
    original_stdout = sys.stdout
    sys.stdout = MultiLogger(original_stdout, out_f)
    
    try:
        sum_f.write("=== EVALUATION SUMMARY ===\n\n")
        
        for filename in FILES_TO_RUN:
            file_path = os.path.join(SCRIPT_DIR, filename)
            if not os.path.exists(file_path):
                print(f"⚠️ Warning: File {filename} not found. Skipping.")
                continue
                
            with open(file_path, 'r') as f:
                dataset = json.load(f)
            
            # Run the eval
            correct, total = asyncio.run(run_pre_release_eval(dataset, filename, debug_f))
            
            # Write to summary log
            pct = (correct / total * 100) if total > 0 else 0
            sum_f.write(f"{filename}: {correct}/{total} ({pct:.1f}%)\n")
            sum_f.flush()
            
        print("\nAll evaluations complete.")
        
    finally:
        sys.stdout = original_stdout
        out_f.close()
        sum_f.close()
        debug_f.close()