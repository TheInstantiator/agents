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


# Get absolute paths to safely locate the document and DB directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(SCRIPT_DIR, ".."))
from core_agent import CoreAgent

DOC_PATH = os.path.join(SCRIPT_DIR, "..", "documents", "DandDRuleset.pdf")
DB_DIR = os.path.join(SCRIPT_DIR, "chroma_db")
SQLITE_DB_PATH = os.path.join(SCRIPT_DIR, "parent_chunks.db")

def calculate_md5(file_path: str) -> str:
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

class TableSummary(BaseModel):
    thoughts: str = Field(default="", description="Internal reasoning.")
    summary: str = Field(description="A 1-sentence summary.")

class ParentSummary(BaseModel):
    title: str = Field(description="A short descriptive title for this section.")
    summary: str = Field(description="A 1-2 sentence overview of what this section explains.")

class TableRowProse(BaseModel):
    thoughts: str = Field(default="", description="Internal reasoning about how to convert these table rows to prose.")
    prose_rows: list[str] = Field(description="One natural-language sentence per data row, e.g. 'The Longsword costs 15 gp, deals 1d8 slashing damage, and has the Versatile property.'")

async def summarize_content(agent: CoreAgent, text: str, mode: str = "table") -> BaseModel:
    if mode == "table":
        prompt = f"Summarize what this data table lists in 1 sentence:\n\n{text[:2000]}"
        response_model = TableSummary
    else:
        prompt = f"Summarize the core rules/concepts in this D&D rulebook snippet in 1-2 sentences. Be very concise:\n\n{text[:3000]}"
        response_model = ParentSummary
        
    # Temporary swap response model for this call
    old_model = agent.response_model
    agent.response_model = response_model
    try:
        response = await agent.ask(prompt)
        return response
    finally:
        agent.response_model = old_model


# ── PROGRAMMATIC TABLE-TO-PROSE ─────────────────────────────────────────────
def parse_markdown_table_to_prose(table_text: str) -> list[str]:
    """Attempt to programmatically convert a standard Markdown table (with pipes)
    into one natural-language sentence per data row.
    
    Returns an empty list if the table is not a standard pipe-delimited table
    (i.e. it's a 'ghost table') so the caller can fall back to the LLM."""
    lines = [l.strip() for l in table_text.strip().split('\n') if l.strip()]
    
    # Must have at least header + separator + 1 data row, and contain pipes
    if len(lines) < 3 or '|' not in lines[0]:
        return []
    
    # Parse header row
    headers = [h.strip() for h in lines[0].split('|') if h.strip()]
    if not headers:
        return []
    
    # Skip separator row(s) — rows that are all dashes / colons / pipes
    data_start = 1
    for i in range(1, len(lines)):
        cleaned = lines[i].replace('|', '').replace('-', '').replace(':', '').strip()
        if cleaned == '' or all(c in '-|: ' for c in lines[i]):
            data_start = i + 1
        else:
            break
    
    prose_rows = []
    for line in lines[data_start:]:
        if '|' not in line:
            continue
        values = [v.strip() for v in line.split('|') if v.strip()]
        if not values:
            continue
        # Skip rows that look like separators
        if all(set(v) <= {'-', ':', ' '} for v in values):
            continue
        
        # Build a sentence by zipping headers with values
        pairs = []
        for header, value in zip(headers, values):
            if value and value != '-' and value != '—':
                pairs.append(f"the {header} is {value}")
        
        if pairs:
            # Capitalise the first word and join with commas
            sentence = ", ".join(pairs)
            sentence = sentence[0].upper() + sentence[1:] + "."
            prose_rows.append(sentence)
    
    return prose_rows


async def convert_table_to_prose_rows(agent: CoreAgent, table_text: str) -> list[str]:
    """Convert each data row in a table to a natural language sentence.
    
    Strategy:
    1. Try programmatic parsing first (free, instant, deterministic).
    2. Fall back to LLM only for ghost tables or if programmatic parse fails.
    """
    # 1. Attempt programmatic extraction for standard Markdown tables
    prose_rows = parse_markdown_table_to_prose(table_text)
    if prose_rows:
        return prose_rows
    
    # 2. Fallback: use the LLM for ghost tables or non-standard formats
    old_model = agent.response_model
    agent.response_model = TableRowProse
    try:
        prompt = (
            f"Convert each DATA ROW in this D&D table into a single natural-language sentence.\n"
            f"Include all column values (name, cost, damage, weight, properties, etc.) in the sentence.\n"
            f"Skip header rows and separator rows (e.g. rows with only dashes).\n"
            f"Example output: ['The Longsword costs 15 gp, deals 1d8 slashing damage, weighs 3 lbs, and has the Versatile (1d10) property.']\n\n"
            f"Table:\n{table_text[:3000]}"
        )
        response = await agent.ask(prompt)
        return response.prose_rows if response.prose_rows else []
    except Exception as e:
        print(f"  ⚠️  Prose conversion failed: {e}")
        return []
    finally:
        agent.response_model = old_model

def extract_tables_from_markdown(text: str) -> list[str]:
    """Extends detection to 'Ghost Tables' and handles empty lines between rows."""
    lines = text.split('\n')
    tables = []
    current_table = []
    empty_line_count = 0
    
    for line in lines:
        cleaned = line.strip()
        # 1. Catch Standard Markdown Tables (| header |)
        is_md_table = line.count('|') >= 2
        
        # 2. Catch "Ghost Tables" (Aligned patterns like '10–11  +0')
        is_ghost_row = bool(re.search(r'^\d+.*\s+[+−-]\d+$', cleaned)) or \
                       bool(re.search(r'^\d+–\d+\s+[+−-]\d+$', cleaned))
        
        if is_md_table or is_ghost_row:
            current_table.append(line)
            empty_line_count = 0
        elif cleaned == "" and current_table:
            # Allow up to 1 empty line between rows (common in some PDF rips)
            empty_line_count += 1
            if empty_line_count > 1:
                # Check if we have at least 2 non-empty lines
                if len([l for l in current_table if l.strip()]) > 2:
                    tables.append('\n'.join(current_table))
                current_table = []
                empty_line_count = 0
            else:
                current_table.append(line)
        else:
            if current_table and len([l for l in current_table if l.strip()]) > 2:
                tables.append('\n'.join(current_table))
            current_table = []
            empty_line_count = 0
            
    if current_table and len([l for l in current_table if l.strip()]) > 2:
        tables.append('\n'.join(current_table))
        
    return tables

# --- Parallel Processing Core ---
# Semaphore controls how many LLM calls run concurrently.
# Ollama defaults to sequential (OLLAMA_NUM_PARALLEL=1).
# For mobile 4090 (16GB VRAM) + phi4, 2 is the safe max.
# Set OLLAMA_NUM_PARALLEL=2 in your env and restart ollama to match.
SEM = asyncio.Semaphore(2)

async def process_parent_chunk_async(p_idx, p_text, current_hash, table_agent, child_splitter):
    """Worker function for parallel processing of a single parent chunk."""
    p_id = f"{current_hash}_parent_{p_idx}"
    
    async with SEM:
        # 1. Agentic Parent Summary (The "Semantic context")
        try:
            p_review = await summarize_content(table_agent, p_text, mode="parent")
            p_summary = f"{p_review.title}: {p_review.summary}"
        except Exception:
            p_summary = "D&D Rules Section"

        # 2. Extract and summarize ANY tables
        tables = extract_tables_from_markdown(p_text)
        chunk_docs = []
        chunk_metas = []
        chunk_ids = []
        
        for t_idx, table in enumerate(tables):
            # 2a. Get a 1-sentence overview summary (for metadata / fallback)
            try:
                t_review = await summarize_content(table_agent, table, mode="table")
                t_summary = t_review.summary
            except Exception:
                t_summary = "A table containing data."

            # 2b. Convert every data row to dense prose for embedding
            # This prevents the Cross-Encoder and dense vector from penalising
            # dry tabular formatting (e.g. "Longsword 1d8 Slashing 15 GP").
            prose_rows = await convert_table_to_prose_rows(table_agent, table)

            if prose_rows:
                # Embed each prose row as its own child chunk
                for r_idx, prose in enumerate(prose_rows):
                    if not prose.strip():
                        continue
                    r_id = f"{p_id}_table_{t_idx}_row_{r_idx}"
                    chunk_docs.append(prose)
                    chunk_metas.append({
                        "source": "DandDRuleset.pdf",
                        "doc_hash": current_hash,
                        "type": "table_row_prose",
                        "table_content": table,          # raw table kept for context retrieval
                        "table_summary": t_summary,
                        "parent_summary": p_summary,
                    })
                    chunk_ids.append(r_id)
            else:
                # Fallback: embed the summary if prose conversion failed
                t_id = f"{p_id}_table_{t_idx}"
                chunk_docs.append(t_summary)
                chunk_metas.append({
                    "source": "DandDRuleset.pdf",
                    "doc_hash": current_hash,
                    "type": "table",
                    "table_content": table,
                    "parent_summary": p_summary,
                })
                chunk_ids.append(t_id)

    # (Outside semaphore for LLM, but inside function)
    # 3. Create Child Chunks
    child_chunks = child_splitter.split_text(p_text)
    for c_idx, child in enumerate(child_chunks):
        if not child.strip(): continue
        c_id = f"{p_id}_child_{c_idx}"
        chunk_docs.append(child)
        chunk_metas.append({
            "source": "DandDRuleset.pdf",
            "doc_hash": current_hash,
            "type": "text",
            "parent_summary": p_summary,
        })
        chunk_ids.append(c_id)
        
    return p_id, p_text, chunk_docs, chunk_metas, chunk_ids

async def build_database():
    """Main function to rip the PDF, chunk it hierarchically, extract tables, and embed."""
    if not os.path.exists(DOC_PATH):
        print(f"❌ Error: Could not find document at {DOC_PATH}")
        return

    print("0. Initializing Vector Database connection...")
    client = chromadb.PersistentClient(path=DB_DIR)
    # Using a new collection name since the schema fundamentally changed 
    collection = client.get_or_create_collection(
        name="dnd_rules_multi_v2",
        metadata={"hnsw:space": "cosine"}
    )
    
    current_hash = calculate_md5(DOC_PATH)
    
    try:
        existing_docs = collection.get(where={"doc_hash": current_hash}, limit=1)
        if existing_docs and existing_docs.get('ids') and len(existing_docs['ids']) > 0:
            print(f"✅ Document already ingested (MD5: {current_hash[:8]}). Skipping to save time & compute!")
            return
    except Exception:
        pass 
        
    print("0.5 Initializing SQLite Storage Engine for Parent Chunks...")
    conn = sqlite3.connect(SQLITE_DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS parent_chunks 
                 (id TEXT PRIMARY KEY, doc_hash TEXT, content TEXT)''')
    conn.commit()

    print("\n1. Extracting text from PDF (Using pymupdf4llm for Markdown formatting)...")
    content = pymupdf4llm.to_markdown(DOC_PATH)
    
    print("\n2. Hierarchical Chunking (Markdown Header Splitting → Child Chunks + Table Extraction)...")
    
    # ── PARENT SPLITTING: MarkdownHeaderTextSplitter ───────────────────────
    # Splits by structural headers instead of arbitrary character counts.
    # This produces logically coherent parent chunks that respect section boundaries.
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]
    md_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
        strip_headers=False,  # Keep headers in the text for context
    )
    md_header_docs = md_splitter.split_text(content)
    
    # The MarkdownHeaderTextSplitter returns Document objects with .page_content and .metadata.
    # Extract the text from each for our pipeline.
    parent_chunks_text = [doc.page_content for doc in md_header_docs if doc.page_content.strip()]
    print(f"   -> Markdown header splitter produced {len(parent_chunks_text)} structurally logical Parent Chunks.")
    
    # Some header-split sections may be enormous (e.g. a chapter with no sub-headers).
    # We sub-split any parent chunk > 4000 chars to keep them manageable for the LLM summariser.
    MAX_PARENT_SIZE = 4000
    oversized_splitter = RecursiveCharacterTextSplitter(
        chunk_size=MAX_PARENT_SIZE, chunk_overlap=300, separators=["\n\n", "\n", " ", ""]
    )
    final_parent_chunks = []
    for chunk in parent_chunks_text:
        if len(chunk) > MAX_PARENT_SIZE:
            sub_chunks = oversized_splitter.split_text(chunk)
            final_parent_chunks.extend(sub_chunks)
        else:
            final_parent_chunks.append(chunk)
    
    parent_chunks_text = final_parent_chunks
    print(f"   -> After sub-splitting oversized sections: {len(parent_chunks_text)} Parent Chunks.")
    
    # ── CHILD SPLITTING: Tiny 300-char chunks for ChromaDB precision ──────
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50, separators=["\n\n", "\n", " ", ""])
    
    documents = [] # Will hold text to embed (either child chunks or table prose rows)
    metadatas = [] 
    ids = []
    
    # Initialize Core Agent for table summarization (Using Phi for speed so we don't wait hours)
    litellm_kwargs = CoreAgent.load_litellm_kwargs_from_config("phi-agent") 
    table_agent = CoreAgent(
        agent_id="table_summarizer",
        system_prompt="You are a data assistant. Your job is to concisely summarize markdown tables into logical prose. Do not list every item, just explain what the table is used for.",
        litellm_kwargs=litellm_kwargs,
        one_shot=True,
        response_model=TableSummary
    )

    print(f"\nProcessing {len(parent_chunks_text)} Parent Chunks (Generating Summaries & Tiny Chunks in Parallel)...")
    
    # Run all parent chunk processing tasks in parallel
    tasks = [
        process_parent_chunk_async(p_idx, p_text, current_hash, table_agent, child_splitter)
        for p_idx, p_text in enumerate(parent_chunks_text)
    ]
    
    # We use asyncio.as_completed to show progress as they finish
    all_results = []
    for f in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Parent Chunks"):
        p_id, p_text, chunk_docs, chunk_metas, chunk_ids = await f
        
        # 1. Store parent chunk in SQLite
        c.execute("INSERT OR REPLACE INTO parent_chunks (id, doc_hash, content) VALUES (?, ?, ?)",
                  (p_id, current_hash, p_text))
        
        # 2. Add children/tables to the main lists
        documents.extend(chunk_docs)
        metadatas.extend(chunk_metas)
        ids.extend(chunk_ids)

    conn.commit()
    conn.close()

    print(f"\nCreated {len(documents)} total tiny vectors (Child Chunks + Table Prose Rows).")
    
    print("\n3. Loading Local Native Encoder (BAAI/bge-large-en-v1.5)...")
    device_default = "cuda" if torch.cuda.is_available() else "cpu"
    device = os.getenv("RAG_DEVICE", device_default)
    print(f"   -> Encodings will be processed on: {device.upper()}")
    encoder = SentenceTransformer("BAAI/bge-large-en-v1.5", device=device)
    
    print("4. Embedding chunks... (This depends on your hardware)")
    # Added show_progress_bar=True so the CPU encoding doesn't appear frozen!
    embeddings = encoder.encode(documents, normalize_embeddings=True, show_progress_bar=True)
    
    print("5. Inserting into ChromaDB...")
    batch_size = 5000
    for i in range(0, len(documents), batch_size):
        end = min(i + batch_size, len(documents))
        print(f"   -> Saving vectors {i} to {end}...")
        collection.upsert(
            documents=documents[i:end],
            metadatas=metadatas[i:end],
            embeddings=embeddings[i:end].tolist(),
            ids=ids[i:end]
        )
        
    print(f"\n✅ Successfully ingested Multi-Vector Hierarchical RAG into local ChromaDB at {DB_DIR}!")
    print(f"💰 Table Summarization API Cost: ${table_agent.cost:.6f} ({table_agent.tokens_used} tokens) | Total LLM Time: {table_agent.total_response_time:.2f}s")

    # ── SAVE BM25 CORPUS ──────────────────────────────────────────────────────
    # Persist the full text corpus to disk so run_evals_routed.py can build
    # a real BM25 index at startup without re-fetching everything from Chroma.
    BM25_CORPUS_PATH = os.path.join(SCRIPT_DIR, "bm25_corpus.pkl")
    print(f"\n6. Saving BM25 corpus ({len(documents)} docs) to {BM25_CORPUS_PATH}...")
    bm25_corpus = {
        "documents": documents,
        "metadatas": metadatas,
        "ids": ids,
    }
    with open(BM25_CORPUS_PATH, "wb") as f:
        pickle.dump(bm25_corpus, f)
    print(f"✅ BM25 corpus saved. Load it with: pickle.load(open('{BM25_CORPUS_PATH}', 'rb'))")

if __name__ == "__main__":
    asyncio.run(build_database())
