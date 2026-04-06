import re
import json
import asyncio
import chromadb
import sqlite3
import sys
import os
import pickle
from typing import List, Dict, Tuple, AsyncGenerator
from pydantic import BaseModel, Field

from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

# ========================= PATHS =========================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(SCRIPT_DIR, ".."))
from core_agent import CoreAgent

DB_PATH = os.path.join(SCRIPT_DIR, "chroma_db")
SQLITE_PATH = os.path.join(SCRIPT_DIR, "parent_chunks.db")
BM25_CORPUS_PATH = os.path.join(SCRIPT_DIR, "bm25_corpus.pkl")

# ========================= SETTINGS =========================
MAX_PASSES_GENERAL = 8
MAX_PASSES_SPECIFIC = 3

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

# ========================= SETUP =========================
def init_rag_system():
    client = chromadb.PersistentClient(path=DB_PATH)
    collection = client.get_collection(name="dnd_rules_multi_v2")

    parent_db = sqlite3.connect(SQLITE_PATH, check_same_thread=False)
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
    
    return collection, parent_cursor, encoder, bm25_index, bm25_corpus_docs, bm25_corpus_metas, bm25_corpus_ids

print("Initializing local AI databases for Streamlit...")
collection, parent_cursor, encoder, bm25_index, bm25_corpus_docs, bm25_corpus_metas, bm25_corpus_ids = init_rag_system()

# ========================= AGENTS CONFIG =========================
# Assign a specific LLM from your config.json to each agent role here to mix and match!

# Decomposer: Breaks down complex questions into atomic sub-queries. Needs good structure and reasoning.
decomposer_kwargs = CoreAgent.load_litellm_kwargs_from_config("e-gemma-agent")

# HyDE: Generates hypothetical textbook answers to enhance semantic search. Needs creativity.
hyde_kwargs       = CoreAgent.load_litellm_kwargs_from_config("e-gemma-agent")

# Critic: Audits the final answer against the strict context to find missing facts. Needs extreme precision.
critic_kwargs     = CoreAgent.load_litellm_kwargs_from_config("phi-agent")

# Merger: Edits and combines text into clean Markdown. Needs good formatting and layout skills.
merger_kwargs     = CoreAgent.load_litellm_kwargs_from_config("e-gemma-agent")

# Answer Generator: Synthesizes the exact answer from raw context chunks. Needs strict instruction following.
answer_kwargs     = CoreAgent.load_litellm_kwargs_from_config("e-gemma-agent")

# Classifier: Quickly labels questions as General vs Specific. Best with a small, lightning-fast model.
classifier_kwargs = CoreAgent.load_litellm_kwargs_from_config("e-gemma-agent")

# Reranker: Ranks chunks based on relevance. Heaviest prompt. Needs a massive context window and high logic.
reranker_kwargs   = CoreAgent.load_litellm_kwargs_from_config("e-gemma-agent")

# Router: Analyzes chat history to see if the user is asking a follow-up. Needs conversational awareness.
router_kwargs     = CoreAgent.load_litellm_kwargs_from_config("e-gemma-agent")

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
    system_prompt="Answer the question using ONLY the provided context. Do not use your own knowledge. Provide your answer concisely, but ALWAYS begin your answer with a self-contained introductory sentence that explicitly restates the subject of the question (e.g. 'Here is the information regarding the Owlbear:'). If your answer involves a list, progression, or multiple items, you MUST format it using Markdown bullet points and line breaks for readability—do not output a giant text blob. Be incredibly precise: do not mix up table rows, do not alter 'start' vs 'end' of turn timings, and do not perform math unless explicitly instructed by the text. Quote the text directly when determining specific effects or limits. If the context lacks the answer, say 'I don't know'.",
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

# ========================= MAIN PIPELINE =========================
async def execute_full_retrieval_pipeline(query: str, yield_event=None):
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

    rrf_merged = reciprocal_rank_fusion(vector_results, bm25_results, k=30)
    
    if yield_event:
        model_name = reranker_kwargs.get("model", "local").split("/")[-1]
        await yield_event({"type": "status", "message": f"Retrieved {len(rrf_merged)} RRF chunks. Asking {model_name} to rerank..."})

    rerank_prompt = f"Original Question: {query}\n\nRank these chunks from most to least relevant. Return only ordered indices (0-based):\n"
    rerank_lines = []
    for idx, (cid, _) in enumerate(rrf_merged):
        try:
            corpus_idx = bm25_corpus_ids.index(cid)
            doc = bm25_corpus_docs[corpus_idx]
        except ValueError:
            doc = ""
        rerank_lines.append(f"[{idx}] {doc[:500]}...")
        
    rerank_prompt += "\n".join(rerank_lines)
    
    reranked = await reranker.ask(rerank_prompt)
    valid_indices = [idx for idx in reranked.ranked_indices if 0 <= idx < len(rrf_merged)]
    top_indices = valid_indices[:12]
    
    if yield_event:
        await yield_event({"type": "status", "message": f"Parsing full rulebook parent context for top {len(top_indices)} chunks..."})

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
    retrieval_task = asyncio.create_task(execute_full_retrieval_pipeline(question, yield_event=yield_callback))
    
    while not retrieval_task.done() or not queue.empty():
        try:
            # Poll the queue to pass status UI updates upstream
            event = await asyncio.wait_for(queue.get(), timeout=0.1)
            yield event
        except asyncio.TimeoutError:
            pass

    context = retrieval_task.result()
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

    # Critic + Refinement Pass
    # max_passes is already set above during classification
    current_pass = 0
    
    previous_follow_up = ""
    while current_pass < max_passes:
        yield {"type": "status", "message": f"Critic Pass {current_pass+1}/{max_passes} evaluating for missing facts..."}
        
        if current_pass == 0:
            critic_prompt = f"Original Question: {question}\n\nExisting Context:\n{context}\n\nGenerated Answer: {final_answer_text}\n\nIs this answer robust and complete based on the Question and Context? If it is a broad question, are there major D&D rules completely missing from the explanation?"
        else:
            critic_prompt = f"Original Question: {question}\n\nPrevious Follow-up Request: {previous_follow_up}\n\nNew Supplemental Context:\n{refine_context}\n\nMerged Answer so Far: {final_answer_text}\n\nEvaluate if the New Supplemental Context successfully supplied the missing facts you requested. Does the Merged Answer now feel complete? If facts are STILL missing, provide ONE focused follow-up query. Otherwise, mark as complete."
            
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
            
        if critique.follow_up_query.strip().lower() == previous_follow_up.strip().lower():
            yield {"type": "status", "message": "Critic is looping on the same missing facts. Aborting research."}
            break

        yield {"type": "status", "message": f"Pass {current_pass+1} Retrieval: Fetching full pipeline context..."}
        
        refine_task = asyncio.create_task(execute_full_retrieval_pipeline(critique.follow_up_query, yield_event=yield_callback))
        while not refine_task.done() or not queue.empty():
            try:
                event = await asyncio.wait_for(queue.get(), timeout=0.1)
                yield event
            except asyncio.TimeoutError:
                pass
        refine_context = refine_task.result()

        yield {"type": "status", "message": f"Pass {current_pass+1}: Extracting precise supplemental info..."}
        supplemental_obj = await answer_agent.ask(
            f"Original Question: {question}\nFollow-up: {critique.follow_up_query}\n\nContext:\n{refine_context}\n\nProvide the missing information."
        )

        yield {"type": "status", "message": f"Pass {current_pass+1}: Merger Agent stitching answers..."}
        merged = await merger.ask(
            f"Original Answer: {final_answer_text}\n\nNew Information: {supplemental_obj.answer}\n\nCombine them into one clear, complete final answer."
        )
        final_answer_text = merged.answer
        
        yield {"type": "merge_update", "pass": current_pass+1, "answer": final_answer_text}
        
        # Pass the context sliding variables forward instead of blowing up the context payload
        previous_follow_up = critique.follow_up_query
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
