import os
import sys
import hashlib
import sqlite3
import re
import asyncio
import pickle
import chromadb
import pymupdf4llm
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from tqdm import tqdm
import torch

# ========================= PATHS =========================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(SCRIPT_DIR, ".."))
from core_agent import CoreAgent

DOC_PATH = os.path.join(SCRIPT_DIR, "..", "documents", "DandDRuleset.pdf")
DB_DIR = os.path.join(SCRIPT_DIR, "chroma_db")
SQLITE_DB_PATH = os.path.join(SCRIPT_DIR, "parent_chunks.db")
BM25_CORPUS_PATH = os.path.join(SCRIPT_DIR, "bm25_corpus.pkl")

# ========================= HELPERS =========================
def calculate_md5(file_path: str) -> str:
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

# ========================= PYDANTIC MODELS =========================
class TableSummary(BaseModel):
    thoughts: str = Field(default="", description="Internal reasoning.")
    summary: str = Field(description="A 1-sentence summary.")

class ParentSummary(BaseModel):
    title: str = Field(description="A short descriptive title for this section.")
    summary: str = Field(description="A 1-2 sentence overview of what this section explains.")

class TableRowProse(BaseModel):
    thoughts: str = Field(default="", description="Internal reasoning about how to convert these table rows to prose.")
    prose_rows: list[str] = Field(description="One natural-language sentence per data row.")

# ========================= LLM HELPERS =========================
async def summarize_content(agent: CoreAgent, text: str, mode: str = "table") -> BaseModel:
    if mode == "table":
        prompt = f"Summarize what this data table lists in 1 sentence:\n\n{text[:2000]}"
        response_model = TableSummary
    else:
        prompt = f"Summarize the core rules/concepts in this D&D rulebook snippet in 1-2 sentences. Be very concise:\n\n{text[:3000]}"
        response_model = ParentSummary
        
    old_model = agent.response_model
    agent.response_model = response_model
    try:
        return await agent.ask(prompt)
    finally:
        agent.response_model = old_model

# ========================= TABLE PROCESSING =========================
def parse_markdown_table_to_prose(table_text: str) -> list[str]:
    """Fast programmatic conversion for standard markdown tables."""
    lines = [l.strip() for l in table_text.strip().split('\n') if l.strip()]
    if len(lines) < 3 or '|' not in lines[0]:
        return []
    
    headers = [h.strip() for h in lines[0].split('|') if h.strip()]
    if not headers:
        return []

    prose_rows = []
    for line in lines[1:]:
        if '|' not in line:
            continue
        values = [v.strip() for v in line.split('|') if v.strip()]
        if not values or all(set(v) <= {'-', ':', ' '} for v in values):
            continue
        
        pairs = [f"the {header} is {value}" 
                for header, value in zip(headers, values) 
                if value and value not in {'-', '—'}]
        
        if pairs:
            sentence = ", ".join(pairs)
            prose_rows.append(sentence[0].upper() + sentence[1:] + ".")
    
    return prose_rows

async def convert_table_to_prose_rows(agent: CoreAgent, table_text: str) -> list[str]:
    """Convert table rows to prose. Try fast parsing first, fallback to LLM."""
    prose_rows = parse_markdown_table_to_prose(table_text)
    if prose_rows:
        return prose_rows

    # LLM fallback for ghost/complex tables
    old_model = agent.response_model
    agent.response_model = TableRowProse
    try:
        prompt = (
            f"Convert each DATA ROW in this D&D table into a single natural-language sentence.\n"
            f"Include all important values. Skip headers and separators.\n\n"
            f"Table:\n{table_text[:3000]}"
        )
        response = await agent.ask(prompt)
        return response.prose_rows if response.prose_rows else []
    finally:
        agent.response_model = old_model

def extract_tables_from_markdown(text: str) -> list[str]:
    """Detect both standard markdown tables and ghost tables."""
    lines = text.split('\n')
    tables = []
    current_table = []
    empty_count = 0

    for line in lines:
        cleaned = line.strip()
        is_md_table = line.count('|') >= 2
        is_ghost_row = bool(re.search(r'^\d+.*\s+[+−-]\d+$', cleaned)) or \
                       bool(re.search(r'\d+(?:d\d+)?', cleaned) and re.search(r'\b(GP|SP|CP)\b', cleaned, re.I))

        if is_md_table or is_ghost_row:
            current_table.append(line)
            empty_count = 0
        elif cleaned == "" and current_table:
            empty_count += 1
            if empty_count > 1:
                if len([l for l in current_table if l.strip()]) > 2:
                    tables.append('\n'.join(current_table))
                current_table = []
                empty_count = 0
            else:
                current_table.append(line)
        else:
            if current_table and len([l for l in current_table if l.strip()]) > 2:
                tables.append('\n'.join(current_table))
            current_table = []
            empty_count = 0

    if current_table and len([l for l in current_table if l.strip()]) > 2:
        tables.append('\n'.join(current_table))
        
    return tables

# ========================= MAIN PROCESSING =========================
SEM = asyncio.Semaphore(1)

async def process_parent_chunk_async(p_idx: int, p_text: str, current_hash: str, 
                                     table_agent: CoreAgent, child_splitter):
    """Process one parent chunk: summarize, handle tables, create small children."""
    p_id = f"{current_hash}_parent_{p_idx}"
    
    async with SEM:
        # Parent summary
        try:
            p_review = await summarize_content(table_agent, p_text, mode="parent")
            p_summary = f"{p_review.title}: {p_review.summary}"
        except Exception:
            p_summary = "D&D Rules Section"

        # Handle tables
        tables = extract_tables_from_markdown(p_text)
        chunk_docs = []
        chunk_metas = []
        chunk_ids = []

        for t_idx, table in enumerate(tables):
            try:
                t_review = await summarize_content(table_agent, table, mode="table")
                t_summary = t_review.summary
            except Exception:
                t_summary = "A table containing data."

            prose_rows = await convert_table_to_prose_rows(table_agent, table)

            if prose_rows:
                for r_idx, prose in enumerate(prose_rows):
                    if not prose.strip(): continue
                    r_id = f"{p_id}_table_{t_idx}_row_{r_idx}"
                    chunk_docs.append(prose)
                    chunk_metas.append({
                        "source": "DandDRuleset.pdf",
                        "doc_hash": current_hash,
                        "parent_id": p_id,
                        "type": "table_row_prose",
                        "parent_summary": p_summary,
                        "table_summary": t_summary,
                        "table_content": table,
                    })
                    chunk_ids.append(r_id)
            else:
                t_id = f"{p_id}_table_{t_idx}"
                chunk_docs.append(t_summary)
                chunk_metas.append({
                    "source": "DandDRuleset.pdf",
                    "doc_hash": current_hash,
                    "parent_id": p_id,
                    "type": "table",
                    "parent_summary": p_summary,
                    "table_summary": t_summary,
                    "table_content": table,
                })
                chunk_ids.append(t_id)

    # Create small child text chunks
    child_chunks = child_splitter.split_text(p_text)
    for c_idx, child in enumerate(child_chunks):
        if not child.strip(): continue
        c_id = f"{p_id}_child_{c_idx}"
        chunk_docs.append(child)
        chunk_metas.append({
            "source": "DandDRuleset.pdf",
            "doc_hash": current_hash,
            "parent_id": p_id,
            "type": "text",
            "parent_summary": p_summary,
        })
        chunk_ids.append(c_id)
        
    return p_id, p_text, chunk_docs, chunk_metas, chunk_ids


async def build_database():
    if not os.path.exists(DOC_PATH):
        print(f"❌ Error: Could not find document at {DOC_PATH}")
        return

    print("0. Initializing ChromaDB...")
    client = chromadb.PersistentClient(path=DB_DIR)
    collection = client.get_or_create_collection(
        name="dnd_rules_multi_v2", metadata={"hnsw:space": "cosine"}
    )

    current_hash = calculate_md5(DOC_PATH)

    # Skip if already ingested
    try:
        if collection.get(where={"doc_hash": current_hash}, limit=1)["ids"]:
            print(f"✅ Document already ingested (hash: {current_hash[:8]}...). Skipping.")
            return
    except Exception:
        pass

    # SQLite setup
    conn = sqlite3.connect(SQLITE_DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS parent_chunks 
                 (id TEXT PRIMARY KEY, doc_hash TEXT, content TEXT)''')
    conn.commit()

    print("\n1. Extracting Markdown from PDF...")
    content = pymupdf4llm.to_markdown(DOC_PATH)

    print("\n2. Hierarchical Chunking...")
    md_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "Header 1"), ("##", "Header 2"), ("###", "Header 3")],
        strip_headers=False
    )
    md_docs = md_splitter.split_text(content)
    parent_chunks_text = [doc.page_content for doc in md_docs if doc.page_content.strip()]

    # Split oversized parents
    oversized_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=300)
    final_parents = []
    for chunk in parent_chunks_text:
        if len(chunk) > 4000:
            final_parents.extend(oversized_splitter.split_text(chunk))
        else:
            final_parents.append(chunk)
    parent_chunks_text = final_parents

    print(f"   -> Created {len(parent_chunks_text)} parent chunks.")

    child_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)

    documents, metadatas, ids = [], [], []

    litellm_kwargs = CoreAgent.load_litellm_kwargs_from_config("phi-agent")
    table_agent = CoreAgent(
        agent_id="table_summarizer",
        system_prompt="You are a data assistant. Summarize markdown tables into logical prose concisely.",
        litellm_kwargs=litellm_kwargs,
        one_shot=True,
        response_model=TableSummary
    )

    print("\nProcessing parent chunks (with table conversion)...")
    tasks = [process_parent_chunk_async(p_idx, p_text, current_hash, table_agent, child_splitter)
             for p_idx, p_text in enumerate(parent_chunks_text)]

    for f in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Parent Chunks"):
        p_id, p_text, chunk_docs, chunk_metas, chunk_ids = await f
        
        c.execute("INSERT OR REPLACE INTO parent_chunks (id, doc_hash, content) VALUES (?, ?, ?)",
                  (p_id, current_hash, p_text))
        
        documents.extend(chunk_docs)
        metadatas.extend(chunk_metas)
        ids.extend(chunk_ids)

    conn.commit()
    conn.close()

    print(f"\nCreated {len(documents)} embeddable chunks.")

    # Embedding & Storage
    print("\n3. Loading encoder (BAAI/bge-large-en-v1.5)...")
    device = os.getenv("RAG_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
    encoder = SentenceTransformer("BAAI/bge-large-en-v1.5", device=device)

    print("4. Embedding...")
    embeddings = encoder.encode(documents, normalize_embeddings=True, show_progress_bar=True)

    print("5. Storing in ChromaDB...")
    batch_size = 5000
    for i in range(0, len(documents), batch_size):
        end = min(i + batch_size, len(documents))
        collection.upsert(
            documents=documents[i:end],
            metadatas=metadatas[i:end],
            embeddings=embeddings[i:end].tolist(),
            ids=ids[i:end]
        )

    print(f"\n✅ Ingestion complete! Stored in {DB_DIR}")

    # Save BM25 corpus
    print(f"\n6. Saving BM25 corpus to {BM25_CORPUS_PATH}...")
    with open(BM25_CORPUS_PATH, "wb") as f:
        pickle.dump({"documents": documents, "metadatas": metadatas, "ids": ids}, f)
    print("✅ BM25 corpus saved.")

if __name__ == "__main__":
    asyncio.run(build_database())