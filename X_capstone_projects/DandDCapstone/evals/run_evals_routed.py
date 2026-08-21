import re
import json
import asyncio
import chromadb
import sqlite3
import sys
import os
import time
import pickle
from typing import List, Dict
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi

# Setup paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(SCRIPT_DIR, ".."))
sys.path.append(os.path.join(SCRIPT_DIR, "..", ".."))
from rag_config import COLLECTION_NAME, EMBED_MODEL, DB_DIR, SQLITE_DB_PATH, BM25_CORPUS_PATH, get_rag_device

MODEL_NAME = "phi-agent"
ANSWER_MODEL_NAME = "phi-agent"

# Create logger
class Logger(object):
    def __init__(self, filename="eval_routed_output_log.txt"):
        self.terminal = sys.stdout
        self.log = open(filename, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

sys.stdout = Logger(os.path.join(SCRIPT_DIR, "eval_routed_output_log.txt"))
from core_agent import CoreAgent

DB_PATH = DB_DIR
SQLITE_PATH = SQLITE_DB_PATH

DEBUG_MODE = True
DEBUG_LOG_PATH = os.path.join(SCRIPT_DIR, "eval_routed_debug_log.txt")
debug_log_file = open(DEBUG_LOG_PATH, "w") if DEBUG_MODE else None

def debug_print(*args, **kwargs):
    if debug_log_file:
        print(*args, file=debug_log_file, **kwargs)
        debug_log_file.flush()

# --- Models ---
class JudgeVerdict(BaseModel):
    thoughts: str = Field(default="", description="Brief reasoning about whether the answer is present.")
    exact_quote: str = Field(default="", description="If the answer is found, copy the EXACT verbatim sentence from the chunk that contains the facts. If not found, output 'N/A'.")
    answer_found: bool = Field(description="True if the retrieved context contains enough info to answer the question.")
    rank_found: int = Field(description="1-indexed rank of the first chunk that contains the answer. Use -1 if not found.")

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

class AnswerEvalVerdict(BaseModel):
    thoughts: str = Field(description="Detailed reasoning comparing the Generated Answer to the Expected Answer.")
    is_correct: bool = Field(description="True if the generated answer is factually correct given the expected answer.")

# --- Global Setup ---
print("Initializing Vector Database connection...", flush=True)
client = chromadb.PersistentClient(path=DB_PATH)
try:
    collection = client.get_collection(name=COLLECTION_NAME)
except Exception as e:
    print(f"Error accessing collection: {e}. Has build_rag_db.py finished?", flush=True)
    exit()

print("Initializing SQLite (Parent Chunk Store)...", flush=True)
parent_db = sqlite3.connect(SQLITE_PATH)
parent_cursor = parent_db.cursor()

print(f"Loading Local Native Encoder ({EMBED_MODEL})...", flush=True)
encoder = SentenceTransformer(EMBED_MODEL, device=get_rag_device())

print("Loading Cross-Encoder Re-Ranker (ms-marco-MiniLM-L-6-v2)...", flush=True)
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device="cpu")

# ── BM25 INDEX ────────────────────────────────────────────────────────────────
# Load the pre-built corpus from build_rag_db.py and create a real BM25 index.
# This ensures tabular rows (which score poorly with dense vectors) get proper
# term-frequency-based ranking when the router provides exact keywords.

bm25_index = None
bm25_corpus_docs = []
bm25_corpus_metas = []
bm25_corpus_ids = []

def simple_tokenize(text: str) -> List[str]:
    """Robust tokenizer that strips punctuation/Markdown to improve BM25 matching for bolded terms."""
    if not text:
        return []
    return re.findall(r'\w+', text.lower())

if os.path.exists(BM25_CORPUS_PATH):
    print("Loading BM25 corpus from disk...", flush=True)
    with open(BM25_CORPUS_PATH, "rb") as _f:
        _corpus = pickle.load(_f)
    bm25_corpus_docs = _corpus["documents"]
    bm25_corpus_metas = _corpus["metadatas"]
    bm25_corpus_ids = _corpus["ids"]
    # Improved tokenization: prepend summaries to the text so BM25 knows the monster/section name
    # even if it's not in the small child chunk! (Headless Chunk problem fix)
    _tokenised = []
    for doc, meta in zip(bm25_corpus_docs, bm25_corpus_metas):
        enriched_text = f"{meta.get('parent_summary', '')} {meta.get('table_summary', '')} {doc}"
        _tokenised.append(simple_tokenize(enriched_text))
    
    bm25_index = BM25Okapi(_tokenised)
    print(f"  BM25 index built with keyword-enrichment: {len(bm25_corpus_docs)} documents.", flush=True)
else:
    print(f"  ⚠️  BM25 corpus not found at {BM25_CORPUS_PATH}. Run build_rag_db.py first. Falling back to boolean scan.", flush=True)

print(f"Initializing LLM Judge and Master Router ({MODEL_NAME})...", flush=True)
phi_kwargs = CoreAgent.load_litellm_kwargs_from_config(MODEL_NAME)
ONE_SHOT_JUDGE = phi_kwargs.pop("one_shot_judge", False)  # Read + remove (not a litellm param)
print(f"Judge mode: {'ONE-SHOT (all chunks at once)' if ONE_SHOT_JUDGE else 'INCREMENTAL (memory-accumulating)'}")

# Same proven judge as run_evals.py
judge = CoreAgent(
    agent_id="eval_judge",
    system_prompt=(
        "You are a strict D&D rules evaluation judge. "
        "Your ONLY job is to determine whether the retrieved text chunks contain enough information "
        "to correctly answer the given question. "
        "The chunk MUST contain the specific facts required by the question, but it is perfectly acceptable (and often expected) if it contains other surrounding context or information. "
        "Return answer_found=True if the core facts are present verbatim. "
        "CRITICAL: If you find the answer, you MUST extract the exact, verbatim quote from the text that proves it. "
        "Do not paraphrase. If you cannot find a verbatim quote that answers the question, you must set answer_found=False."
    ),
    litellm_kwargs=phi_kwargs,
    response_model=JudgeVerdict,
    one_shot=ONE_SHOT_JUDGE
)

router = CoreAgent(
    agent_id="master_router",
    system_prompt=(
        "You are the Master Routing AI for a D&D 5e offline rules engine. \n"
        "Your goal is to transform a player's natural language question into a high-precision search strategy.\n\n"
        "Return a JSON object with these fields:\n"
        "1. \"scope\": 'BROAD' for general concepts (e.g., 'How to DM'), 'SPECIFIC' for lookups (e.g., 'Red Dragon AC').\n"
        "2. \"category\": One of: 'spells', 'monsters', 'rules', 'classes', 'items', 'lore', 'other'.\n"
        "3. \"exact_match_keywords\": A list of up to 3 highly specific proper nouns, names, or exact phrases (e.g., 'Adult Red Dragon', 'Longsword', 'Fireball'). The text MUST literally contain these words exactly. CRITICAL: DO NOT include generic properties like 'cost', 'damage', 'AC', or 'stats'. Leave empty if there isn't a specific noun.\n"
        "4. \"hyde_excerpt\": A 3-sentence fake excerpt from the D&D rulebook that expands slang/abbreviations to their formal names (e.g. DM -> Dungeon Master, T-Rex -> Tyrannosaurus Rex).\n"
        "   - Sentence 1: Define the formal subject.\n"
        "   - Sentence 2: Mention the specific stat or rule requested.\n"
        "   - Sentence 3: Provide a technical example using official D&D terminology.\n\n"
        "Example Input: \"What's the T-Rex's AC?\"\n"
        "Example Output JSON:\n"
        "{\n"
        "  \"scope\": \"SPECIFIC\",\n"
        "  \"category\": \"monsters\",\n"
        "  \"exact_match_keywords\": [\"Tyrannosaurus Rex\", \"Armor Class\"],\n"
        "  \"hyde_excerpt\": \"A Tyrannosaurus Rex is a huge beast often encountered in jungles. Its Armor Class is determined by its thick, tough hide. This high AC makes the Tyrannosaurus Rex difficult to hit with non-magical weapons.\"\n"
        "}"
    ),
    litellm_kwargs=phi_kwargs,
    response_model=RouterDecision,
    one_shot=True
)

print(f"Initializing Rephrase Agent ({MODEL_NAME})...", flush=True)
rephrase_agent = CoreAgent(
    agent_id="rephrase_agent",
    system_prompt=(
        "You are an expert D&D rules interpreter.\n"
        "Your task is to rewrite user questions to be highly precise and target the 'Standard/Base' ruleset.\n"
        "RELEVANCY RULE: You will be given a 'Hypothetical Answer' from a search router. \n"
        "WARNING: The Hypothetical Answer often contains hallucinations or incorrect facts. \n"
        "DO NOT copy facts (like costs, dice, or rules) from the Hypothetical Answer into the Rewritten Question unless they were in the Original Question.\n"
        "Only use the Hypothetical Answer to understand the 'topic' or 'category' of the search.\n"
        "Make the rewritten question clear and technically accurate, but do not add invented details."
    ),
    litellm_kwargs=phi_kwargs,
    response_model=RewrittenQuery,
    one_shot=True
)

print(f"Initializing Generation Agent ({ANSWER_MODEL_NAME})...", flush=True)
try:
    answer_kwargs = CoreAgent.load_litellm_kwargs_from_config(ANSWER_MODEL_NAME)
except KeyError:
    print(f"Could not load {ANSWER_MODEL_NAME}, defaulting to {MODEL_NAME}")
    answer_kwargs = phi_kwargs

answer_agent = CoreAgent(
    agent_id="answer_generator",
    system_prompt=(
        "You are an expert D&D 5e assistant answering player questions based strictly on the provided text chunks. "
        "Formulate a concise and accurate answer. If the answer is not in the text, say 'I don't know'."
    ),
    litellm_kwargs=answer_kwargs,
    response_model=GeneratedAnswer,
    one_shot=True
)

answer_judge = CoreAgent(
    agent_id="answer_judge",
    system_prompt=(
        "You are an impartial evaluator. You will be given a User Question, an Expected Golden Answer, and a Generated Answer. "
        "Your job is to determine if the Generated Answer is factually correct and equivalent to the Golden Answer. "
        "Ignore formatting or phrasing differences; focus purely on the facts. "
        "CRITICAL: Be a lenient evaluator regarding additional context. "
        "If the Generated Answer is more detailed or provides more surrounding facts than the Golden Answer, DO NOT penalize it, as long as all facts required by the Golden Answer are present and accurate. "
        "Only fail if the Generated Answer contradicts the Golden Answer or misses a key fact required by the Golden Answer."
    ),
    litellm_kwargs=phi_kwargs,
    response_model=AnswerEvalVerdict,
    one_shot=True
)

def get_context_for_rank(results, rank: int, windowing: bool = False) -> str:
    """Builds the full context string for a given rank, including parent summaries and raw tables."""
    doc = results['documents'][0][rank]
    meta = results['metadatas'][0][rank]
    chunk_id = results['ids'][0][rank]
    
    # Robust parent_id extraction (handles {hash}_parent_{idx} prefixes)
    parent_id = meta.get('parent_id')
    if not parent_id:
        parts = chunk_id.split("_")
        parent_id = "_".join(parts[:3])

    # 1. Build Semantic Header from Agentic Summaries
    header_parts = []
    if meta.get('parent_summary'):
        header_parts.append(f"[Section Overview]: {meta['parent_summary']}")
    if meta.get('table_summary'):
        header_parts.append(f"[Table Focus]: {meta['table_summary']}")
    
    header_str = "\n".join(header_parts)
    context = f"{header_str}\n[Chunk {rank+1}] (ID: {chunk_id}):\n{doc}"

    # 2. Raw Table Fallback
    # If the hit is a prose row OR a raw table chunk, include the full raw table for holistic context
    if meta.get('type') in ['table', 'table_row_prose'] and 'table_content' in meta:
        context += f"\n[Full Raw Table Data]:\n{meta['table_content']}"

    # 3. Windowing (surrounding parent chunks)
    if windowing:
        parent_parts = parent_id.split("_")
        if len(parent_parts) >= 3 and parent_parts[2].isdigit():
            idx = int(parent_parts[2])
            base_id = "_".join(parent_parts[:2])
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
        # Standard Parent Retrieval
        parent_cursor.execute("SELECT content FROM parent_chunks WHERE id=?", (parent_id,))
        parent_row = parent_cursor.fetchone()
        if parent_row:
            context += f"\n[Full Section Context]:\n{parent_row[0][:2000]}"

    return context

async def evaluate_retrieval_llm(dataset: List[Dict], dataset_name: str):
    """LLM-as-Judge evaluation with Master Router for retrieval, chunk-by-chunk judgment."""
    print(f"\n--- Running LLM-as-Judge Evaluation w/ Master Router (Judge: {MODEL_NAME}) ---")

    total_mrr = 0.0
    total_recall = 0
    total_gen_success = 0
    total_windowed = 0
    num_queries = len(dataset)

    for i, item in enumerate(dataset):
        query = item['question']
        expected_answer = item.get('expected_answer', '')

        print(f"\nQ{i+1}: {query}")
        print(f"  A: {expected_answer}")

        # CLEAR AGENT MEMORY between questions
        # CLEAR AGENT MEMORY between questions
        judge.working_memory.clear()
        router.working_memory.clear()
        answer_agent.working_memory.clear()
        answer_judge.working_memory.clear()

        # 0. Master Router pass
        t_router = time.time()
        try:
            decision = await router.ask(query)
            router_time = time.time() - t_router
            print(f"  [Router] Scope: {decision.scope} | Cat: {decision.category} | {router_time:.1f}s")
            print(f"  [Router] Exact Matches: {decision.exact_match_keywords}")
            print(f"  [Router] HyDE: {decision.hyde_excerpt[:150]}...")

            k = 80 if decision.scope.upper() == "BROAD" else 20
        except Exception as e:
            print(f"  ⚠️  Router error: {e}")
            decision = RouterDecision(scope="SPECIFIC", category="rules", exact_match_keywords=[], hyde_excerpt=query)
            k = 20

        # 1. Hybrid Search Construction (Dense Vector) 
        # Using the natural language query and the formalized HyDE excerpt together
        search_text = f"{query} {decision.hyde_excerpt}"
        
        query_embedding = encoder.encode(search_text, normalize_embeddings=True).tolist()

        # We retrieve a wider net (k*3) to give the Cross-Encoder more options to rank
        k_fetch = k * 3
        
        # ── TRACK A: Semantic Dense Search ───────────────────────────
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=k_fetch,
            include=['documents', 'metadatas']
        )
        
        docs = results['documents'][0]
        metas = results['metadatas'][0]
        ids = results['ids'][0]

        # ── TRACK B: Real BM25 Search ──────────────────────────────────────────
        # Hard-filter common/generic words the LLM sometimes adds as keywords
        # This prevents noisy BM25 hits from burying dense results.
        blacklist = {
            "cost", "damage", "ac", "stats", "hp", "hit", "points", "price", 
            "modifier", "score", "level", "dc", "weight", "type", "size", 
            "speed", "alignment", "languages", "properties", "standard", "base"
        }
        filtered_keywords = [
            kw for kw in decision.exact_match_keywords
            if kw.lower() not in blacklist and not kw.lower().startswith("level ")
        ]

        # Track which candidate indices came from BM25 (to hard-override CE score for them)
        exact_match_indices = set()
        existing_ids_set = set(ids)

        if bm25_index and filtered_keywords:
            # Build a combined BM25 query from all filtered keywords using the robust tokenizer
            bm25_query_tokens = simple_tokenize(" ".join(filtered_keywords))
            bm25_scores = bm25_index.get_scores(bm25_query_tokens)

            # Get the top k*3 BM25 results
            top_bm25_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:k * 3]

            debug_print(f"  [BM25] Top 5 scores: { [(bm25_corpus_docs[i][:60], round(bm25_scores[i], 2)) for i in top_bm25_indices[:5]] }")

            for bm25_idx in top_bm25_indices:
                if bm25_scores[bm25_idx] <= 0:
                    break  # No more useful BM25 hits
                doc_id = bm25_corpus_ids[bm25_idx]
                if doc_id not in existing_ids_set:
                    docs.append(bm25_corpus_docs[bm25_idx])
                    metas.append(bm25_corpus_metas[bm25_idx])
                    ids.append(doc_id)
                    exact_match_indices.add(len(docs) - 1)
                    existing_ids_set.add(doc_id)
                else:
                    # Already in pool from dense search — still mark for override
                    try:
                        existing_pos = ids.index(doc_id)
                        exact_match_indices.add(existing_pos)
                    except ValueError:
                        pass
        elif filtered_keywords:
            # Fallback: boolean scan if BM25 index not available
            all_data = collection.get(include=['documents', 'metadatas'])
            for kw in filtered_keywords:
                low_kw = kw.lower()
                for doc_f, meta_f, id_f in zip(all_data['documents'], all_data['metadatas'], all_data['ids']):
                    if low_kw in doc_f.lower() and id_f not in existing_ids_set:
                        docs.append(doc_f)
                        metas.append(meta_f)
                        ids.append(id_f)
                        exact_match_indices.add(len(docs) - 1)
                        existing_ids_set.add(id_f)

        # ── Cross-Encoder Re-Ranking ───────────────────────────────────────────
        # The cross-encoder scores every candidate against the query.
        # We then HARD OVERRIDE the score for BM25 exact matches so they can never
        # be buried by the CE's prose-bias against short tabular rows.
        t_rerank = time.time()
        pairs = [[query, doc] for doc in docs]
        scores = cross_encoder.predict(pairs)
        scores = scores.tolist() if hasattr(scores, 'tolist') else list(scores)

        debug_print(f"  [DEBUG] exact_match_indices: {exact_match_indices}")
        for idx in range(len(docs)):
            doc_text = docs[idx].lower()
            meta = metas[idx]
            
            # ── PHRASAL BOOST ──
            # Count how many of the Router's specific keywords/phrases match literally.
            # A chunk with BOTH "Adult Red Dragon" and "Legendary Resistance" 
            # should score higher than one with just "Legendary Resistance".
            phrasal_match_count = 0
            for kw in filtered_keywords:
                low_kw = kw.lower()
                if (low_kw in doc_text or 
                    low_kw in meta.get('parent_summary', '').lower() or
                    low_kw in meta.get('table_summary', '').lower()):
                    phrasal_match_count += 1

            if phrasal_match_count > 0:
                # Tiered Phrasal Boost: (100 * count) ensures multi-keyword matches win.
                old_score = scores[idx]
                scores[idx] = (phrasal_match_count * 100.0) + (old_score / 100.0)
                debug_print(f"  [DEBUG] PHRASAL hit x{phrasal_match_count} chunk {idx} ({meta.get('type')}). CE: {old_score:.2f} -> Boosted: {scores[idx]:.2f}")
            elif idx in exact_match_indices:
                # Mid tier: BM25 Token match (bag of words, no full phrasal match)
                old_score = scores[idx]
                scores[idx] = 50.0 + (old_score / 100.0)
                debug_print(f"  [DEBUG] TOKEN hit chunk {idx} ({meta.get('type')}). CE: {old_score:.2f} -> Boosted: {scores[idx]:.2f}")

        # Sort by score descending and keep top k
        scored_results = sorted(zip(scores, docs, metas, ids), key=lambda x: x[0], reverse=True)
        top_results = scored_results[:k]

        # Re-pack into Chroma results format
        results['documents'][0] = [item[1] for item in top_results]
        results['metadatas'][0] = [item[2] for item in top_results]
        results['ids'][0] = [item[3] for item in top_results]

        rerank_time = time.time() - t_rerank
        debug_print(f"  [ReRanker] Graded {len(docs)} docs, kept top {k} | {rerank_time:.1f}s")

        # 2. Judge evaluation — two modes controlled by ONE_SHOT_JUDGE in config.json
        #    ONE_SHOT (ollama): dump all K chunks in a single call — fast, no per-chunk overhead
        #    INCREMENTAL (paid APIs): accumulate memory across chunks — cheap via prompt caching
        found_rank = -1
        used_windowing = False
        best_verdict = None

        if ONE_SHOT_JUDGE:
            # ── ONE-SHOT MODE ──────────────────────────────────────────────────
            # Concatenate every retrieved chunk + its narrative window into one big prompt
            num_chunks = len(results['documents'][0])
            all_narratives = []
            for rank in range(min(k, num_chunks)):
                narrative = get_context_for_rank(results, rank, windowing=True)
                all_narratives.append(f"--- Result Rank {rank+1} ---\n{narrative}")
            
            combined = "\n\n".join(all_narratives)

            judge.working_memory.clear()
            one_shot_prompt = (
                f"Question: {query}\n"
                f"Expected Answer: {expected_answer}\n\n"
                f"Below are the Top {num_chunks} retrieved results (each with surrounding context):\n\n"
                f"{combined[:120000]}\n\n"
                f"Instructions:\n"
                f"- Do these results COLLECTIVELY contain the specific facts to answer the question?\n"
                f"- The answer might be spread across multiple ranks.\n"
                f"- Set answer_found=true if the combined text has the answer.\n"
                f"- Set rank_found to the rank (1-{num_chunks}) where the fact first appears."
            )
            t_start = time.time()
            debug_print(f"  [One-Shot] Judging all {num_chunks} windowed narratives...")
            try:
                best_verdict = await judge.ask(one_shot_prompt)
            except Exception as e:
                print(f"  ⚠️  One-shot judge error: {e}", flush=True)
                best_verdict = JudgeVerdict(thoughts="error", answer_found=False, rank_found=-1)
            elapsed = time.time() - t_start
            status = "✅ FOUND" if best_verdict.answer_found else "❌ Not found"
            print(f"  [One-Shot] {status} | {elapsed:.1f}s", flush=True)
            if best_verdict.thoughts:
                print(f"    Judge: {best_verdict.thoughts[:1000]}", flush=True)
            if best_verdict.answer_found:
                print(f"    Judge Evidence: \"{best_verdict.exact_quote}\"", flush=True)
                found_rank = best_verdict.rank_found if best_verdict.rank_found > 0 else 1


        else:
            # ── INCREMENTAL MODE (memory-accumulating) ─────────────────────────
            # Evaluates chunk-by-chunk; judge memory persists within the question
            # so split answers are caught when combined evidence becomes complete.
            async def check_chunk(rank, is_window):
                # NOTE: do NOT clear judge memory here — it accumulates across chunks
                # within this question so split answers (e.g. list spread over 2 chunks)
                # are caught when the combined evidence becomes complete.
                context = get_context_for_rank(results, rank, windowing=is_window)
                chunk_limited = context[:6000] if is_window else context[:3000]
                
                judge_prompt = (
                    f"Question: {query}\n"
                    f"Expected Answer: {expected_answer}\n\n"
                    f"New Context (Rank {rank+1}, {'Window' if is_window else 'Chunk'}):\n{chunk_limited}\n\n"
                    f"Consider this new context AND all previous chunks you have evaluated for this question.\n"
                    f"Does the TOTAL context provide the SPECIFIC FACTS required to match the Expected Answer?\n"
                    f"The answer may be split across multiple chunks. Only set answer_found=true if the combined information handles the full requirement.\n"
                    f"DO NOT use outside knowledge. The facts MUST be in the provided text.\n"
                    f"Set answer_found=true ONLY if the facts are found. Set rank_found={rank+1} if found, otherwise -1."
                )
                topic_pre = "Window" if is_window else "Chunk"

                t_start = time.time()
                try:
                    verdict = await judge.ask(judge_prompt)
                except Exception as e:
                    print(f"  ⚠️  Judge error on {topic_pre} {rank+1}: {e}", flush=True)
                    verdict = JudgeVerdict(thoughts="error", answer_found=False, rank_found=-1)

                elapsed = time.time() - t_start
                meta = results['metadatas'][0][rank]
                chunk_topic = meta.get('parent_summary', meta.get('table_summary', ''))[:80]
                raw_doc = results['documents'][0][rank][:100].replace('\n', ' ')
                status = f"✅ FOUND {'VIA WINDOWING' if is_window else 'HERE'}" if verdict.answer_found else f"❌ Not in this {topic_pre.lower()}"

                print(f"  [{topic_pre} {rank+1}] {status} | {elapsed:.1f}s", flush=True)
                if chunk_topic:
                    print(f"    Topic: {chunk_topic}", flush=True)
                print(f"    Doc:   {raw_doc}", flush=True)
                if verdict.thoughts:
                    print(f"    Judge: {verdict.thoughts[:1000]}", flush=True)
                if verdict.answer_found:
                    print(f"    Judge Evidence: \"{verdict.exact_quote}\"", flush=True)

                return {"rank": rank, "is_window": is_window, "verdict": verdict}

            for rank in range(min(k, len(results['documents'][0]))):
                task_res = await check_chunk(rank, is_window=True)
                if task_res['verdict'].answer_found:
                    found_rank = task_res['rank'] + 1
                    used_windowing = True
                    best_verdict = task_res['verdict']
                    break


        if found_rank != -1:
            mrr = 1.0 / found_rank
            total_mrr += mrr
            total_recall += 1
            if used_windowing:
                print(f"  ✅ RETRIEVAL SUCCESS (Windowing): Answer at rank {found_rank} (MRR+: {mrr:.2f})")
                total_windowed += 1
            else:
                print(f"  ✅ RETRIEVAL SUCCESS (Judge): Answer at rank {found_rank} (MRR+: {mrr:.2f})")
            debug_print(f"     Reasoning: {best_verdict.thoughts[:1000]}")
        else:
            print(f"  ❌ RETRIEVAL FAILURE (Judge): No valid chunk found in top {k} searches.", flush=True)

        # --- NEW PHASE: ANSWER GENERATION ---
        # 1) Rephrase the query to target base entities
        rephrase_agent.working_memory.clear()
        rephrase_prompt = (
            f"Original Question: {query}\n\n"
            f"Router Metadata:\n"
            f"  Scope: {decision.scope} | Category: {decision.category}\n"
            f"  Keywords: {decision.exact_match_keywords}\n"
            f"  Potential Topic Context (DANGEROUS: May contain lies): {decision.hyde_excerpt}\n\n"
            f"Rewrite the Original Question to clearly ask for the base/standard properties. \n"
            f"STRICT RULE: Do not include ANY specific numbers, costs, or mechanics from the 'Potential Topic Context' in your rewrite. \n"
            f"Only use that context to understand what the user is likely referring to (e.g. if they say 'T-Rex', the context helps you know they mean 'Tyrannosaurus Rex')."
        )
        t_ref = time.time()
        try:
            ref_resp = await rephrase_agent.ask(rephrase_prompt)
            if isinstance(ref_resp, str):
                ref_resp = RewrittenQuery(thoughts="Fallback", rewritten_question=ref_resp)
        except Exception as e:
            print(f"  ⚠️  Rephrase error: {e}", flush=True)
            ref_resp = RewrittenQuery(thoughts="error", rewritten_question=query)
        elapsed_ref = time.time() - t_ref
        print(f"  [Rephraser ({MODEL_NAME})] | {elapsed_ref:.1f}s", flush=True)
        print(f"    Rewritten Q: {ref_resp.rewritten_question}", flush=True)

        # 2) Give the generator only the relevant context that actually contained the answer
        if found_rank != -1:
            generation_context = get_context_for_rank(results, found_rank - 1, windowing=used_windowing)
        else:
            # Fallback to all chunks if judge completely failed, just in case
            all_contexts = [get_context_for_rank(results, r, windowing=True) for r in range(min(k, len(results['documents'][0])))]
            generation_context = "\n\n".join(all_contexts)
            
        gen_prompt = (
            f"Original Question: {query}\n"
            f"Rewritten Clarification: {ref_resp.rewritten_question}\n"
            f"Router Scope: {decision.scope} | Category: {decision.category}\n"
            f"Router Keys: {decision.exact_match_keywords}\n"
            f"Hypothetical Answer: {decision.hyde_excerpt}\n\n"
            f"Context:\n{generation_context[:120000]}\n\n"
            f"Instructions:\n"
            f"- Answer the question strictly using the provided context.\n"
            f"- The hypothetical answer is just a guide, not the actual answer.  Notice its content and search accordingly.\n"
            f"- Focus on answering the Rewritten Clarification, which ensures you target the standard, non-magical base version of any item or creature.\n"
            f"- You should search for things with the specific keywords in the 'Router Keys'. Your answer MUST pertain specifically to the exact 'Router Keys'.\n"
            f"- Use the 'Hypothetical Answer' as a guide for what a good answer might look like, but base your facts ONLY on the Context provided.\n"
            f"- Do not include external knowledge.\n"
            f"- If the base item/monster from the Router Keys is NOT explicitly defined with its base stats in the context, you MUST put 'I don't know' in the final_answer field. Do NOT substitute a magical variant.\n"
            f"- DO NOT BE CHATTY. Write only the factual answer or 'I don't know'.\n"
        )
        
        t_gen = time.time()
        try:
            gen_response = await answer_agent.ask(gen_prompt)
        except Exception as e:
            print(f"  ⚠️  Answer generation error: {e}", flush=True)
            gen_response = GeneratedAnswer(thoughts="error", final_answer="Error during generation")
        elapsed_gen = time.time() - t_gen
        
        print(f"  [Generation ({ANSWER_MODEL_NAME})] | {elapsed_gen:.1f}s", flush=True)
        print(f"    Answer: {gen_response.final_answer.strip()}", flush=True)
        
        # --- PHASE: ANSWER EVALUATION ---
        eval_prompt = (
            f"Question: {query}\n"
            f"Expected Golden Answer: {expected_answer}\n"
            f"Generated Answer: {gen_response.final_answer}\n\n"
            f"Is the generated answer factually correct according to the golden answer?"
        )
        t_eval = time.time()
        try:
            eval_response = await answer_judge.ask(eval_prompt)
        except Exception as e:
            print(f"  ⚠️  Answer evaluation error: {e}", flush=True)
            eval_response = AnswerEvalVerdict(thoughts="error", is_correct=False)
        elapsed_eval = time.time() - t_eval
        
        if eval_response.is_correct:
            total_gen_success += 1
            print(f"  ✅ GEN SUCCESS (Judge) | {elapsed_eval:.1f}s", flush=True)
        else:
            print(f"  ❌ GEN FAILED (Judge) | {elapsed_eval:.1f}s", flush=True)
            print(f"    Gen Eval Thoughts: {eval_response.thoughts}", flush=True)

    # Final Report
    avg_mrr = total_mrr / num_queries
    recall_rate = total_recall / num_queries
    gen_success_rate = total_gen_success / num_queries

    print("\n" + "="*45)
    print(f"📊 ROUTED EVALUATION METRICS: {dataset_name}")
    print("="*45)
    print(f"Total Queries   : {num_queries}")
    print(f"Judge Model     : {MODEL_NAME}")
    print(f"Answer Model    : {ANSWER_MODEL_NAME}")
    print(f"Retrieval MRR+  : {avg_mrr:.4f}  (Target: ~1.0)")
    print(f"Retrieval Succ  : ({recall_rate*100:.2f}%) {total_recall} / {num_queries}")
    print(f"End-to-End Gen  : ({gen_success_rate*100:.2f}%) {total_gen_success} / {num_queries}")
    print(f"Windowing Hits  : {total_windowed} fallback retrievals")
    print(f"Judge API Cost  : ${judge.cost + answer_judge.cost:.4f} ({judge.tokens_used + answer_judge.tokens_used} tokens)")
    print(f"Answer API Cost : ${answer_agent.cost:.4f} ({answer_agent.tokens_used} tokens)")
    print(f"Router+Rephrase Cost: ${router.cost + rephrase_agent.cost:.4f} ({router.tokens_used + rephrase_agent.tokens_used} tokens)")
    print("="*45)

if __name__ == "__main__":
    from dotenv import load_dotenv
    os.chdir("/home/ouar/projects/agents/X_capstone_projects")
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
            log_msg = f"\n{'='*60}\n🚀 ROUTED EVALUATING DATASET: {ds_name}\n{'='*60}\n"
            print(log_msg)
            with open(ds_path, 'r') as f:
                golden_dataset = json.load(f)
            asyncio.run(evaluate_retrieval_llm(golden_dataset, ds_name))
        else:
            print(f"\n⚠️ Dataset not found: {ds_path}\n")
