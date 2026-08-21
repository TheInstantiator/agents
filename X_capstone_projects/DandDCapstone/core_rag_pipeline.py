import re
import json
import asyncio
import chromadb
import sqlite3
import sys
import os
import pickle
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, AsyncGenerator, Optional
from pydantic import BaseModel, Field

from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(SCRIPT_DIR, ".."))
from core_agent import CoreAgent
from rag_config import (
    DB_DIR,
    SQLITE_DB_PATH,
    BM25_CORPUS_PATH,
    COLLECTION_NAME,
    EMBED_MODEL,
    RERANK_MODEL,
    RRF_CANDIDATES,
    CE_TOP_K,
    CE_SCORE_FLOOR,
    CE_MIN_CHUNKS,
    CE_AMBIGUOUS_GAP,
    MAX_CHILDREN_PER_PARENT,
    TABLE_CE_BOOST,
    TABLE_SKIP_EXPAND_CE,
    MIN_TABLE_ROWS,
    PARENT_WINDOW_BEFORE,
    PARENT_WINDOW_AFTER,
    PARENT_WINDOW_CAP,
    MAX_PASSES_GENERAL,
    MAX_PASSES_SPECIFIC,
    FOLLOWUP_OVERLAP_ABORT,
    TABLE_TYPES,
    get_rag_device,
    get_ce_device,
    prefers_tables,
)

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

class QuestionClassification(BaseModel):
    thoughts: str
    category: str = Field(description="Must be 'General' or 'Specific'.")

class RouterDecision(BaseModel):
    thoughts: str
    needs_swarm: bool = Field(description="True if the prompt needs a new database search. False if the answer is entirely within the chat history.")
    cached_answer: str = Field(description="The direct answer if needs_swarm is False. Otherwise empty string.")
    rewritten_query: str = Field(description="If needs_swarm is True, rewrite the User Prompt so that it is a highly specific, standalone D&D question that includes ALL relevant context from the Chat History (e.g. adding the specific Class, Monster, or Spell being discussed). If the prompt is already highly specific, return it unchanged.")


@dataclass
class RetrievalResult:
    context: str
    parent_ids: set = field(default_factory=set)
    ce_top_score: Optional[float] = None


# ========================= SETUP =========================
def init_rag_system():
    client = chromadb.PersistentClient(path=DB_DIR)
    try:
        collection = client.get_collection(name=COLLECTION_NAME)
    except Exception:
        collection = client.get_collection(name="dnd_rules_multi_v2")

    parent_db = sqlite3.connect(SQLITE_DB_PATH, check_same_thread=False)
    parent_cursor = parent_db.cursor()

    device = get_rag_device()
    encoder = SentenceTransformer(EMBED_MODEL, device=device)

    try:
        cross_encoder = CrossEncoder(RERANK_MODEL, device=get_ce_device())
    except Exception as e:
        print(f"Cross-encoder load failed ({e}). Falling back to LLM rerank.")
        cross_encoder = None

    with open(BM25_CORPUS_PATH, "rb") as f:
        bm25_data = pickle.load(f)
    bm25_corpus_docs = bm25_data["documents"]
    bm25_corpus_metas = bm25_data["metadatas"]
    bm25_corpus_ids = bm25_data["ids"]
    bm25_id_to_idx = {cid: i for i, cid in enumerate(bm25_corpus_ids)}

    _tokenized = [
        re.findall(r"\w+", f"{meta.get('header_path', '')} {doc}".lower())
        for doc, meta in zip(bm25_corpus_docs, bm25_corpus_metas)
    ]
    bm25_index = BM25Okapi(_tokenized)

    return (
        collection, parent_cursor, encoder, bm25_index,
        bm25_corpus_docs, bm25_corpus_metas, bm25_corpus_ids,
        cross_encoder, bm25_id_to_idx,
    )

print("Initializing local AI databases for Streamlit...")
(
    collection, parent_cursor, encoder, bm25_index,
    bm25_corpus_docs, bm25_corpus_metas, bm25_corpus_ids,
    cross_encoder, bm25_id_to_idx,
) = init_rag_system()

# ========================= AGENTS CONFIG =========================
# Assign a specific LLM from your config.json to each agent role here to mix and match!

# Decomposer: Breaks down complex questions into atomic sub-queries. Needs good structure and reasoning.
decomposer_kwargs = CoreAgent.load_litellm_kwargs_from_config("gemma-agent")

# HyDE: Generates hypothetical textbook answers to enhance semantic search. Needs creativity.
hyde_kwargs       = CoreAgent.load_litellm_kwargs_from_config("gemma-agent")

# Critic: Audits the final answer against the strict context to find missing facts. Needs extreme precision.
critic_kwargs     = CoreAgent.load_litellm_kwargs_from_config("phi-agent")

# Merger: Edits and combines text into clean Markdown. Needs good formatting and layout skills.
merger_kwargs     = CoreAgent.load_litellm_kwargs_from_config("gemma-agent")

# Answer Generator: Synthesizes the exact answer from raw context chunks. Needs strict instruction following.
answer_kwargs     = CoreAgent.load_litellm_kwargs_from_config("gemma-agent")

# Classifier: Quickly labels questions as General vs Specific. Best with a small, lightning-fast model.
classifier_kwargs = CoreAgent.load_litellm_kwargs_from_config("gemma-agent")

# Reranker: Ranks chunks based on relevance. Heaviest prompt. Needs a massive context window and high logic.
reranker_kwargs   = CoreAgent.load_litellm_kwargs_from_config("gemma-agent")

# Router: Analyzes chat history to see if the user is asking a follow-up. Needs conversational awareness.
router_kwargs     = CoreAgent.load_litellm_kwargs_from_config("gemma-agent")

# ========================= AGENT INSTANCES =========================
decomposer = CoreAgent(
    agent_id="decomposer",
    system_prompt="You are a D&D rules expert. Break complex questions into 2-4 atomic sub-queries. For questions about class features, attributes, or costs, ensure one sub-query specifically targets the relevant table or level-progression section. Keep your sub-queries purely factual and literal; do not assume the user's question contains a mistake. You MUST output valid JSON with exactly two fields: 'thoughts' and 'sub_queries'.",
    litellm_kwargs=decomposer_kwargs,
    response_model=DecomposedQueries,
    one_shot=True
)

hyde_agent = CoreAgent(
    agent_id="hyde_agent",
    system_prompt="Generate a short, plausible excerpt from an official D&D rulebook. This excerpt should look like a rules section containing the facts needed to answer the question. CRITICAL: Keep both 'thoughts' and 'hyde_text' incredibly concise (2-3 sentences maximum). Do not write an entire page, and do not use repetitive emojis. You MUST output valid JSON with exactly two fields: 'thoughts' and 'hyde_text'.",
    litellm_kwargs=hyde_kwargs,
    response_model=HyDEAnswer,
    one_shot=True
)

critic = CoreAgent(
    agent_id="critic",
    system_prompt="You are a strict D&D rules auditor. Compare the Original Question and the provided Context with the Generated Answer. If the Context contains specific facts (like numbers, dice types, or costs) that are missing from the Answer, provide ONE focused follow-up query. Do NOT use any external D&D knowledge. If the answer is already fully supported by the Context, mark it as complete. You MUST output valid JSON with exactly four fields: 'thoughts', 'is_complete', 'missing_aspects', and 'follow_up_query'.",
    litellm_kwargs=critic_kwargs,
    response_model=CritiqueResult,
    one_shot=True
)

merger = CoreAgent(
    agent_id="merger",
    system_prompt="You are an expert editor. Combine the original answer and the supplemental answer into one clear, factual final answer. Use Markdown formatting (bullet points, bold text, line breaks) to make the final answer highly readable. Do NOT add meta-commentary (e.g. 'according to the context' or 'the text mentions'). If the supplemental answer does not contain new facts or says 'I don't know', ignore it entirely and just return the original answer.",
    litellm_kwargs=merger_kwargs,
    response_model=StandardAnswer,
    one_shot=True
)

answer_agent = CoreAgent(
    agent_id="answer_generator",
    system_prompt="Answer the question using ONLY the provided context. Do not use your own knowledge. Provide your answer concisely, but ALWAYS begin your answer with a self-contained introductory sentence that explicitly restates the subject of the question (e.g. 'Here is the information regarding the Owlbear:'). If your answer involves a list, progression, or multiple items, you MUST format it using Markdown bullet points and line breaks for readability—do not output a giant text blob. Be incredibly precise: do not mix up table rows, do not alter 'start' vs 'end' of turn timings, and do not perform math unless explicitly instructed by the text. Quote the text directly when determining specific effects or limits. If the context lacks the answer, say 'I don't know'. If the context is thin or contradictory, say so explicitly and list which fact is missing. Do not fill gaps.",
    litellm_kwargs=answer_kwargs,
    response_model=StandardAnswer,
    one_shot=True
)

classifier = CoreAgent(
    agent_id="classifier",
    system_prompt="Classify the user's D&D question as either 'General' or 'Specific'. A 'Specific' question asks for a defined rule, stat, class ability, or cost (e.g. 'What is the gold cost of a Longsword?', 'How does Fireball work?'). A 'General' question is broad, open-ended, and requires explanation of multiple systems (e.g. 'How do I play D&D?', 'How does combat work?'). You MUST output valid JSON with exactly two distinct fields: 'thoughts' (your reasoning) and 'category' (strictly the exact string 'General' or 'Specific'). Do not create any extra fields.",
    litellm_kwargs=classifier_kwargs,
    response_model=QuestionClassification,
    one_shot=True
)

reranker = CoreAgent(
    agent_id="reranker",
    system_prompt="You are an expert search reranker. Rank the given search results by relevance to the question. Output a list of the integer indices representing the original position of each document, ordered from most relevant to least relevant. CRITICAL: Keep your 'thoughts' field incredibly concise (1-2 sentences maximum). Do not write an essay and do not evaluate every chunk. Just output the array. You MUST output valid JSON with exactly two fields: 'thoughts' and 'ranked_indices'.",
    litellm_kwargs=reranker_kwargs,
    response_model=RerankOutput,
    one_shot=True
)

lazy_router = CoreAgent(
    agent_id="router",
    system_prompt="You are a conversational routing agent. Review the Chat History and the User Prompt. If the prompt can be answered instantly using ONLY the facts explicitly stated in the Chat History, set 'needs_swarm' to false and provide the 'cached_answer'. Ensure 'cached_answer' is a self-contained, grammatically complete sentence that explicitly restates the subject of the query. If the prompt requires searching the D&D rulebook for new facts, set 'needs_swarm' to true. CRITICAL: If setting needs_swarm to true, you MUST look at the Chat History to see what Class, Spell, Monster, Topic, etc., the user was currently talking about, and rewrite the User Prompt to explicitly include that context. Only do this if you feel this is a followup question in need of clarification.",
    litellm_kwargs=router_kwargs,
    response_model=RouterDecision,
    one_shot=True
)

def reset_agent_metrics():
    for agent in [classifier, decomposer, hyde_agent, reranker, answer_agent, critic, merger, lazy_router]:
        agent.tokens_used = 0
        agent.cost = 0.0

def get_agent_metrics():
    agents = [classifier, decomposer, hyde_agent, reranker, answer_agent, critic, merger, lazy_router]
    tokens = sum(agent.tokens_used for agent in agents)
    cost = sum(agent.cost for agent in agents)
    
    breakdown = {}
    mapping = {
        "classifier": "Classifier",
        "decomposer": "Decomposer",
        "hyde_agent": "HYDE",
        "reranker": "Reranker",
        "answer_generator": "Answer",
        "critic": "Critic",
        "merger": "Merger",
        "router": "Router"
    }
    for agent in agents:
        mod = getattr(agent, "config_model_name", "unknown")
        if mod not in breakdown:
            breakdown[mod] = {"tokens": 0, "cost": 0.0, "roles": []}
        breakdown[mod]["tokens"] += agent.tokens_used
        breakdown[mod]["cost"] += agent.cost
        
        role = mapping.get(agent.agent_id, agent.agent_id.capitalize())
        if role not in breakdown[mod]["roles"]:
            breakdown[mod]["roles"].append(role)
            
    return tokens, cost, breakdown

# ========================= HELPERS =========================
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


def _lookup(cid: str) -> Tuple[str, dict]:
    idx = bm25_id_to_idx.get(cid)
    if idx is None:
        return "", {}
    return bm25_corpus_docs[idx], bm25_corpus_metas[idx]


def _raw_text(doc: str, meta: dict) -> str:
    return meta.get("raw_text") or doc


def _token_overlap(a: str, b: str) -> float:
    ta = set(re.findall(r"\w+", a.lower()))
    tb = set(re.findall(r"\w+", b.lower()))
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _snap_window(parent_text: str, start: int, raw_len: int) -> str:
    lo = max(0, start - PARENT_WINDOW_BEFORE)
    hi = min(len(parent_text), start + raw_len + PARENT_WINDOW_AFTER)
    if lo > 0:
        nl = parent_text.rfind("\n", 0, lo)
        if nl != -1:
            lo = nl + 1
    if hi < len(parent_text):
        nl = parent_text.find("\n", hi)
        if nl != -1:
            hi = nl
    window = parent_text[lo:hi]
    if len(window) > PARENT_WINDOW_CAP:
        mid = start + raw_len // 2
        half = PARENT_WINDOW_CAP // 2
        lo2 = max(0, mid - half)
        hi2 = min(len(parent_text), lo2 + PARENT_WINDOW_CAP)
        window = parent_text[lo2:hi2]
    return window


def _merge_parent_windows(parent_text: str, spans: List[Tuple[int, int]]) -> str:
    ranges = []
    for start, end in spans:
        lo = max(0, start - PARENT_WINDOW_BEFORE)
        hi = min(len(parent_text), end + PARENT_WINDOW_AFTER)
        ranges.append([lo, hi])
    ranges.sort()
    merged = []
    for lo, hi in ranges:
        if not merged or lo > merged[-1][1]:
            merged.append([lo, hi])
        else:
            merged[-1][1] = max(merged[-1][1], hi)
    pieces = []
    for lo, hi in merged:
        if lo > 0:
            nl = parent_text.rfind("\n", 0, lo)
            if nl != -1:
                lo = nl + 1
        if hi < len(parent_text):
            nl = parent_text.find("\n", hi)
            if nl != -1:
                hi = nl
        pieces.append(parent_text[lo:hi])
    text = "\n".join(pieces)
    cap = PARENT_WINDOW_CAP * max(1, len(merged))
    return text[:cap]


def _cross_encoder_rank(query: str, rrf_merged: List[Tuple]) -> List[Tuple[str, float]]:
    kept = []
    pairs = []
    for cid, _ in rrf_merged:
        doc, meta = _lookup(cid)
        raw = _raw_text(doc, meta)
        if not raw:
            continue
        kept.append(cid)
        pairs.append((query, raw))
    if not pairs:
        return []
    if cross_encoder is None:
        return [(cid, 0.0) for cid in kept]
    scores = cross_encoder.predict(pairs)
    ranked = list(zip(kept, [float(s) for s in scores]))
    ranked.sort(key=lambda x: x[1], reverse=True)
    return ranked


def _apply_ce_floor(ranked: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
    above = [x for x in ranked if x[1] >= CE_SCORE_FLOOR]
    if len(above) < CE_MIN_CHUNKS:
        return ranked[:max(CE_MIN_CHUNKS, min(len(ranked), CE_TOP_K * 2))]
    return above


def _apply_table_boost(query: str, ranked: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
    if not prefers_tables(query):
        return ranked
    boosted = []
    for cid, score in ranked:
        _, meta = _lookup(cid)
        if meta.get("type") in TABLE_TYPES:
            score = score * TABLE_CE_BOOST
        boosted.append((cid, score))
    boosted.sort(key=lambda x: x[1], reverse=True)
    return boosted


def _diversity_cap(ranked: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
    counts = {}
    out = []
    for cid, score in ranked:
        _, meta = _lookup(cid)
        pid = meta.get("parent_id") or cid
        if counts.get(pid, 0) >= MAX_CHILDREN_PER_PARENT:
            continue
        counts[pid] = counts.get(pid, 0) + 1
        out.append((cid, score))
        if len(out) >= CE_TOP_K:
            break
    return out


def _ensure_table_rows(selected: List[Tuple[str, float]], ranked: List[Tuple[str, float]], query: str) -> List[Tuple[str, float]]:
    if not prefers_tables(query):
        return selected
    table_n = sum(1 for cid, _ in selected if _lookup(cid)[1].get("type") in TABLE_TYPES)
    if table_n >= MIN_TABLE_ROWS:
        return selected
    selected_ids = {cid for cid, _ in selected}
    extras = []
    for cid, score in ranked:
        if cid in selected_ids:
            continue
        if _lookup(cid)[1].get("type") in TABLE_TYPES:
            extras.append((cid, score))
            if len(extras) >= MIN_TABLE_ROWS - table_n:
                break
    if not extras:
        return selected
    out = list(selected)
    for extra in extras:
        replaced = False
        for i in range(len(out) - 1, -1, -1):
            if _lookup(out[i][0])[1].get("type") not in TABLE_TYPES:
                out[i] = extra
                replaced = True
                break
        if not replaced:
            out.append(extra)
    out.sort(key=lambda x: x[1], reverse=True)
    return out[:CE_TOP_K]


def _build_context(selected: List[Tuple[str, float]]) -> Tuple[str, set]:
    spans_by_parent = {}
    for cid, ce_score in selected:
        doc, meta = _lookup(cid)
        pid = meta.get("parent_id", "")
        if not pid:
            continue
        raw = _raw_text(doc, meta)
        if meta.get("type") == "table_row_prose" and ce_score >= TABLE_SKIP_EXPAND_CE:
            continue
        start = int(meta.get("child_char_start") or 0)
        spans_by_parent.setdefault(pid, []).append((start, start + len(raw)))

    merged_text = {}
    for pid, spans in spans_by_parent.items():
        parent_cursor.execute("SELECT content FROM parent_chunks WHERE id=?", (pid,))
        row = parent_cursor.fetchone()
        if not row or not row[0]:
            continue
        parent_text = row[0]
        any_start = any(
            "child_char_start" in _lookup(cid)[1]
            for cid, _ in selected
            if _lookup(cid)[1].get("parent_id") == pid
        )
        if any_start:
            merged_text[pid] = _merge_parent_windows(parent_text, spans)
        else:
            merged_text[pid] = parent_text[:1500]

    seen_parents = set()
    parent_ids = set()
    context_parts = []
    for rank_pos, (cid, _) in enumerate(selected):
        doc, meta = _lookup(cid)
        if not doc and not meta:
            continue
        pid = meta.get("parent_id", "")
        raw = _raw_text(doc, meta)
        chunk_context = f"[Retrieval Rank {rank_pos+1}]\n"
        header_bits = []
        if meta.get("header_path"):
            header_bits.append(meta["header_path"])
        if meta.get("parent_summary"):
            header_bits.append(meta["parent_summary"])
        if header_bits:
            chunk_context += f"Section: {' | '.join(header_bits)}\n"
        if meta.get("table_summary"):
            chunk_context += f"Table Focus: {meta['table_summary']}\n"
        chunk_context += f"Exact Excerpt: {raw}\n"
        if pid:
            parent_ids.add(pid)
            if pid in merged_text and pid not in seen_parents:
                seen_parents.add(pid)
                chunk_context += f"\nBroader Section Context:\n{merged_text[pid]}\n"
        context_parts.append(chunk_context)
    return "\n\n---\n\n".join(context_parts), parent_ids

# ========================= MAIN PIPELINE =========================
async def execute_full_retrieval_pipeline(query: str, yield_event=None, is_general: bool = False) -> RetrievalResult:
    if yield_event:
        await yield_event({"type": "status", "message": f"Expanding query using Decomposer & HyDE..."})

    decomp = await decomposer.ask(query)
    hyde = await hyde_agent.ask(query)

    if yield_event:
        await yield_event({
            "type": "expansion",
            "sub_queries": decomp.sub_queries,
            "hyde_text": hyde.hyde_text
        })

    all_queries = [query] + decomp.sub_queries + [hyde.hyde_text]

    if yield_event:
        await yield_event({"type": "status", "message": "Executing Vector Semantic & BM25 Keyword Search..."})

    all_embeddings = encoder.encode(all_queries, normalize_embeddings=True).tolist()
    vector_results = collection.query(query_embeddings=all_embeddings, n_results=20, include=['documents', 'metadatas'])

    bm25_tokens = re.findall(r'\w+', " ".join(all_queries).lower())
    bm25_scores = bm25_index.get_scores(bm25_tokens)
    top_bm25_idx = sorted(range(len(bm25_scores)), key=lambda x: bm25_scores[x], reverse=True)[:20]

    bm25_results = {
        'documents': [[bm25_corpus_docs[mi] for mi in top_bm25_idx]],
        'metadatas': [[bm25_corpus_metas[mi] for mi in top_bm25_idx]],
        'ids': [[bm25_corpus_ids[mi] for mi in top_bm25_idx]]
    }

    rrf_merged = reciprocal_rank_fusion(vector_results, bm25_results, k=RRF_CANDIDATES)

    if yield_event:
        await yield_event({"type": "status", "message": f"Retrieved {len(rrf_merged)} RRF chunks. Cross-encoder reranking..."})

    ranked = _cross_encoder_rank(query, rrf_merged)
    ranked = _apply_table_boost(query, ranked)
    ranked = _apply_ce_floor(ranked)
    selected = _diversity_cap(ranked)
    selected = _ensure_table_rows(selected, ranked, query)
    ce_top = selected[0][1] if selected else None

    use_llm_rerank = False
    if selected and is_general and cross_encoder is not None:
        tail = selected[min(len(selected), CE_TOP_K) - 1][1]
        if (selected[0][1] - tail) < CE_AMBIGUOUS_GAP:
            use_llm_rerank = True
    if selected and cross_encoder is None:
        use_llm_rerank = True

    if use_llm_rerank:
        if yield_event:
            model_name = reranker_kwargs.get("model", "local").split("/")[-1]
            await yield_event({"type": "status", "message": f"Ambiguous set — asking {model_name} to rerank..."})
        pool = selected if selected else ranked[:RRF_CANDIDATES]
        rerank_prompt = f"Original Question: {query}\n\nRank these chunks from most to least relevant. Return only ordered indices (0-based):\n"
        rerank_lines = []
        for idx, (cid, _) in enumerate(pool):
            doc, meta = _lookup(cid)
            rerank_lines.append(f"[{idx}] {_raw_text(doc, meta)[:500]}...")
        rerank_prompt += "\n".join(rerank_lines)
        reranked = await reranker.ask(rerank_prompt)
        valid_indices = [idx for idx in reranked.ranked_indices if 0 <= idx < len(pool)]
        if valid_indices:
            selected = [pool[idx] for idx in valid_indices[:CE_TOP_K]]

    if yield_event:
        await yield_event({"type": "status", "message": f"Parsing parent windows for top {len(selected)} chunks..."})

    context, parent_ids = _build_context(selected)
    return RetrievalResult(context=context, parent_ids=parent_ids, ce_top_score=ce_top)

async def answer_dnd_question(question: str):
    """Async generator that yields JSON events for a Streamlit UI"""
    yield {"type": "status", "message": "Classifying query..."}
    classification = await classifier.ask(question)
    is_general = classification.category.strip().lower() == "general"
    max_passes = MAX_PASSES_GENERAL if is_general else MAX_PASSES_SPECIFIC
    yield {"type": "classification", "category": classification.category, "max_passes": max_passes}

    # 1. Pipeline
    queue = asyncio.Queue()
    async def yield_callback(data):
        await queue.put(data)
        
    # Start retrieval as a background task
    retrieval_task = asyncio.create_task(
        execute_full_retrieval_pipeline(question, yield_event=yield_callback, is_general=is_general)
    )

    while not retrieval_task.done() or not queue.empty():
        try:
            event = await asyncio.wait_for(queue.get(), timeout=0.1)
            yield event
        except asyncio.TimeoutError:
            pass

    retrieval = retrieval_task.result()
    context = retrieval.context
    seen_parent_ids = set(retrieval.parent_ids)
    ce_top_score = retrieval.ce_top_score
    yield {"type": "status", "message": "Drafting initial answer based on context window..."}

    if is_general:
        answer_instruction = "This is a General question. Provide a highly detailed, comprehensive explanation covering all aspects of the rules found in the context. Stay within 16k tokens."
    else:
        answer_instruction = "This is a Specific question. Give a highly specific, extremely concise answer focusing only on the exact core facts requested."

    answer_obj = await answer_agent.ask(
        f"Question: {question}\n\nContext:\n{context}\n\n{answer_instruction}"
    )

    final_answer_text = answer_obj.answer
    yield {"type": "initial_answer", "answer": final_answer_text, "thoughts": answer_obj.thoughts}

    if "i don't know" in final_answer_text.lower() and (
        ce_top_score is None or ce_top_score < CE_SCORE_FLOOR
    ):
        yield {"type": "status", "message": "Context too weak for a grounded answer. Stopping critic loop."}
        yield {"type": "done", "final_answer": "The retrieved rules do not contain enough information to answer this."}
        return

    current_pass = 0
    previous_follow_ups = []
    refine_context = ""
    while current_pass < max_passes:
        yield {"type": "status", "message": f"Critic Pass {current_pass+1}/{max_passes} evaluating for missing facts..."}

        if current_pass == 0:
            critic_prompt = f"Original Question: {question}\n\nExisting Context:\n{context}\n\nGenerated Answer: {final_answer_text}\n\nIs this answer robust and complete based on the Question and Context? If it is a broad question, are there major D&D rules completely missing from the explanation?"
        else:
            critic_prompt = f"Original Question: {question}\n\nPrevious Follow-up Request: {previous_follow_ups[-1]}\n\nNew Supplemental Context:\n{refine_context}\n\nMerged Answer so Far: {final_answer_text}\n\nEvaluate if the New Supplemental Context successfully supplied the missing facts you requested. Does the Merged Answer now feel complete? If facts are STILL missing, provide ONE focused follow-up query. Otherwise, mark as complete."

        critique = await critic.ask(critic_prompt)

        yield {
            "type": "critic_eval",
            "pass": current_pass+1,
            "thoughts": critique.thoughts,
            "follow_up": critique.follow_up_query,
            "is_complete": critique.is_complete
        }

        if critique.is_complete or not critique.follow_up_query.strip():
            break

        follow_up = critique.follow_up_query.strip()
        follow_norm = " ".join(follow_up.lower().split())
        if any(" ".join(prev.lower().split()) == follow_norm for prev in previous_follow_ups):
            yield {"type": "status", "message": "Critic is looping on the same missing facts. Aborting research."}
            break
        if any(_token_overlap(follow_up, prev) > FOLLOWUP_OVERLAP_ABORT for prev in previous_follow_ups):
            yield {"type": "status", "message": "Critic follow-up overlaps a prior query. Aborting research."}
            break

        yield {"type": "status", "message": f"Pass {current_pass+1} Retrieval: Fetching full pipeline context..."}

        refine_task = asyncio.create_task(
            execute_full_retrieval_pipeline(follow_up, yield_event=yield_callback, is_general=is_general)
        )
        while not refine_task.done() or not queue.empty():
            try:
                event = await asyncio.wait_for(queue.get(), timeout=0.1)
                yield event
            except asyncio.TimeoutError:
                pass
        refine_result = refine_task.result()
        refine_context = refine_result.context
        new_parents = refine_result.parent_ids - seen_parent_ids
        if not new_parents:
            yield {"type": "status", "message": "Follow-up retrieval added no new sections. Stopping critic."}
            break
        seen_parent_ids.update(refine_result.parent_ids)

        yield {"type": "status", "message": f"Pass {current_pass+1}: Extracting precise supplemental info..."}
        supplemental_obj = await answer_agent.ask(
            f"Original Question: {question}\nFollow-up: {follow_up}\n\nContext:\n{refine_context}\n\nProvide the missing information."
        )

        yield {"type": "status", "message": f"Pass {current_pass+1}: Merger Agent stitching answers..."}
        merged = await merger.ask(
            f"Original Answer: {final_answer_text}\n\nNew Information: {supplemental_obj.answer}\n\nCombine them into one clear, complete final answer."
        )
        final_answer_text = merged.answer

        yield {"type": "merge_update", "pass": current_pass+1, "answer": final_answer_text}

        previous_follow_ups.append(follow_up)
        current_pass += 1

    yield {"type": "done", "final_answer": final_answer_text}

async def route_query_with_history(query: str, chat_history: str):
    """Intercepts queries using conversational memory before hitting vector search"""
    reset_agent_metrics()
    yield {"type": "status", "message": "Lazy Router evaluating chat history..."}
    prompt = f"Chat History:\n{chat_history}\n\nUser Prompt: {query}"
    decision = await lazy_router.ask(prompt)
    
    if not decision.needs_swarm:
        tokens, cost, breakdown = get_agent_metrics()
        yield {"type": "done_from_cache", "final_answer": decision.cached_answer, "thoughts": decision.thoughts, "tokens": tokens, "cost": cost, "breakdown": breakdown}
    else:
        # Route to full swarm with rewritten isolated query
        standalone_query = decision.rewritten_query if decision.rewritten_query else query
        if standalone_query != query:
            yield {"type": "status", "message": f"Router Contextualized Query: {standalone_query}"}
        
        async for event in answer_dnd_question(standalone_query):
            if event["type"] == "done":
                tokens, cost, breakdown = get_agent_metrics()
                event["tokens"] = tokens
                event["cost"] = cost
                event["breakdown"] = breakdown
            yield event

if __name__ == "__main__":
    # Test script for CLI debugging
    async def test():
        async for event in route_query_with_history("What is Fireball?", ""):
            print(event["type"])
    asyncio.run(test())
