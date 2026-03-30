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

class QuestionClassification(BaseModel):
    thoughts: str
    category: str = Field(description="Must be 'General' or 'Specific'.")

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
    system_prompt="You are a D&D rules expert. Break complex questions into 2-4 atomic sub-queries. For questions about class features, attributes, or costs, ensure one sub-query specifically targets the relevant table or level-progression section. Keep your sub-queries purely factual and literal; do not assume the user's question contains a mistake.",
    litellm_kwargs=phi_kwargs,
    response_model=DecomposedQueries,
    one_shot=True
)

hyde_agent = CoreAgent(
    agent_id="hyde_agent",
    system_prompt="Generate a short, plausible excerpt from an official D&D rulebook. This excerpt should look like a rules section containing the facts needed to answer the question.",
    litellm_kwargs=phi_kwargs,
    response_model=HyDEAnswer,
    one_shot=True
)

critic = CoreAgent(
    agent_id="critic",
    system_prompt="You are a strict D&D rules auditor. Compare the Original Question and the provided Context with the Generated Answer. If the Context contains specific facts (like numbers, dice types, or costs) that are missing from the Answer, provide ONE focused follow-up query. Do NOT use any external D&D knowledge. If the answer is already fully supported by the Context, mark it as complete.",
    litellm_kwargs=phi_kwargs,
    response_model=CritiqueResult,
    one_shot=True
)

merger = CoreAgent(
    agent_id="merger",
    system_prompt="You are an expert editor. Combine the original answer and the supplemental answer into one clear, factual final answer. Do NOT add meta-commentary. Ensure the final result is strictly grounded in the provided facts.",
    litellm_kwargs=phi_kwargs,
    response_model=StandardAnswer,
    one_shot=True
)

answer_agent = CoreAgent(
    agent_id="answer_generator",
    system_prompt="Answer the question using ONLY the provided context. Do not use your own knowledge. Provide your answer as a concise, exact, and complete sentence rather than a single word or number. Be incredibly precise: do not mix up table rows, do not alter 'start' vs 'end' of turn timings, and do not perform math unless explicitly instructed by the text. Quote the text directly when determining specific effects or limits. If the context lacks the answer, say 'I don't know'.",
    litellm_kwargs=phi_kwargs,
    response_model=StandardAnswer,
    one_shot=True
)

answer_judge = CoreAgent(
    agent_id="answer_judge",
    system_prompt="You are an impartial evaluator. Compare the Generated Answer to the Expected Answer. Determine if the Generated Answer contains the core factual information required by the Expected Answer. DO NOT fail an answer just because it doesn't show the mathematical reasoning, or because it answers with a stark number (like '5' instead of '5 hit points'). If the final factual conclusion matches, mark it as true.",
    litellm_kwargs=phi_kwargs,
    response_model=AnswerEvalVerdict,
    one_shot=True
)

reranker = CoreAgent(
    agent_id="reranker",
    system_prompt="You are an expert search reranker. Rank the given search results by relevance to the question. You MUST output a list of exactly 12 integer indices representing the original position of the most relevant documents, ordered from best to worst.",
    litellm_kwargs=phi_kwargs,
    response_model=RerankOutput,
    one_shot=True
)

classifier = CoreAgent(
    agent_id="classifier",
    system_prompt="Classify the user's D&D question as either 'General' or 'Specific'. A 'Specific' question asks for a defined rule, stat, class ability, or cost (e.g. 'What is the gold cost of a Longsword?', 'How does Fireball work?'). A 'General' question is broad, open-ended, and requires explanation of multiple systems (e.g. 'How do I play D&D?', 'How does combat work?'). You MUST output valid JSON with exactly two distinct fields: 'thoughts' (your reasoning) and 'category' (strictly the exact string 'General' or 'Specific'). Do not create any extra fields.",
    litellm_kwargs=phi_kwargs,
    response_model=QuestionClassification,
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
async def execute_full_retrieval_pipeline(query: str, log_prefix="  ") -> str:
    # 1. Decomposition + HyDE
    decomp = await decomposer.ask(query)
    hyde = await hyde_agent.ask(query)
    
    print(f"{log_prefix}[Decomposer] Generated {len(decomp.sub_queries)} sub-queries: {decomp.sub_queries}")
    print(f"{log_prefix}[HyDE] Excerpt: {hyde.hyde_text[:150]}...")

    all_queries = [query] + decomp.sub_queries + [hyde.hyde_text]

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
    print(f"{log_prefix}[Retrieval] Vector returned {total_vec} hits. BM25 returned {total_bm25} hits. RRF merged to top {len(rrf_merged)} chunks.")

    # 5. LLM Reranking (phi-4)
    rerank_prompt = f"Original Question: {query}\n\nRank these chunks from most to least relevant. Return only ordered indices (0-based):\n"
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
    valid_indices = [idx for idx in reranked.ranked_indices if 0 <= idx < len(rrf_merged)]
    top_indices = valid_indices[:12]
    print(f"{log_prefix}[Reranker] Filtered and kept the top {len(top_indices)} chunks for context window.")

    # 6. Window Expansion with Deduplication
    seen_parents = set()
    context_parts = []
    for rank_pos, idx in enumerate(top_indices):
        cid = rrf_merged[idx][0]
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
        
        chunk_context += f"Exact Excerpt: {doc}\n"
        
        if parent_id and parent_id not in seen_parents:
            seen_parents.add(parent_id)
            parent_cursor.execute("SELECT content FROM parent_chunks WHERE id=?", (parent_id,))
            row = parent_cursor.fetchone()
            if row and row[0]:
                chunk_context += f"\nBroader Section Context:\n{row[0][:1500]}\n"
                
        context_parts.append(chunk_context)

    return "\n\n---\n\n".join(context_parts)

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

            # 0. Classify Question
            classification = await classifier.ask(original_query)
            is_general = classification.category.strip().lower() == "general"
            print(f"  [Classifier] Question categorized as: {classification.category} (Passes allowed: {3 if is_general else 1})")

            # 1-6. Full Retrieval Pipeline (Decomposer -> HyDE -> Dual Search -> RRF -> Rerank -> Expand Context)
            context = await execute_full_retrieval_pipeline(original_query, log_prefix="  ")

            # 7. Generate Initial Answer
            answer_obj = await answer_agent.ask(
                f"Question: {original_query}\n\nContext:\n{context}\n\nAnswer concisely using only the context."
            )
            initial_thoughts = answer_obj.thoughts
            initial_answer_text = answer_obj.answer

            final_thoughts = initial_thoughts
            final_answer_text = initial_answer_text

            # 8. Critic + Refinement Pass (Multi-Pass for General)
            max_passes = 3 if is_general else 1
            current_pass = 0
            
            while current_pass < max_passes:
                critique = await critic.ask(
                    f"Original Question: {original_query}\n\nExisting Context:\n{context}\n\nGenerated Answer: {final_answer_text}\n\nIs this answer robust and complete based on the Question and Context? If it is a broad question, are there major D&D rules completely missing from the explanation?"
                )

                if critique.is_complete or not critique.follow_up_query.strip():
                    break # Answer is good enough!

                YELLOW = "\033[0;33m"
                RESET = "\033[0m"
                print(f"  {YELLOW}[Critic Pass {current_pass+1}/{max_passes} Thoughts]: {critique.thoughts}{RESET}")
                print(f"  {YELLOW}[Critic Pass {current_pass+1}/{max_passes} Follow-up]: {critique.follow_up_query}{RESET}")

                # Full second retrieval pass using the complete multi-agent pipeline
                print(f"  {YELLOW}[Refinement] Running full retrieval pipeline for follow-up query...{RESET}")
                refine_context = await execute_full_retrieval_pipeline(critique.follow_up_query, log_prefix="    ")

                supplemental_obj = await answer_agent.ask(
                    f"Original Question: {original_query}\nFollow-up: {critique.follow_up_query}\n\nContext:\n{refine_context}\n\nProvide the missing information."
                )

                # Merge old and new
                merged = await merger.ask(
                    f"Original Answer: {final_answer_text}\n\nNew Information: {supplemental_obj.answer}\n\nCombine them into one clear, complete final answer."
                )
                final_thoughts = merged.thoughts
                final_answer_text = merged.answer
                
                # Append the new context into the running total context so the critic knows what we've already found
                context += "\n\n---\n\n" + refine_context
                
                current_pass += 1

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
                f"Question: {original_query}\n\nExpected Answer: {expected_answer}\nGenerated Answer: {final_answer_text}\nIs the Generated Answer factually correct based on the Expected Answer?"
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