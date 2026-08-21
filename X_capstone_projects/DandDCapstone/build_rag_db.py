import os
import sys
import json
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

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(SCRIPT_DIR, ".."))
from core_agent import CoreAgent
from rag_config import (
    DOC_PATH,
    DB_DIR,
    SQLITE_DB_PATH,
    BM25_CORPUS_PATH,
    INGEST_CACHE_PATH,
    COLLECTION_NAME,
    EMBED_MODEL,
    PARENT_CHUNK_SIZE,
    PARENT_CHUNK_OVERLAP,
    TEXT_CHILD_SIZE,
    TEXT_CHILD_OVERLAP,
    get_rag_device,
)


def calculate_md5(file_path: str) -> str:
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def content_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def header_path(md_doc) -> str:
    order = ["Header 1", "Header 2", "Header 3"]
    parts = [md_doc.metadata[k] for k in order if md_doc.metadata.get(k)]
    return " > ".join(parts)


def build_embed_text(raw_text: str, path: str, parent_summary: str, table_summary: str = "") -> str:
    lines = []
    if path:
        lines.append(f"Section: {path}")
    if table_summary:
        lines.append(f"Table: {table_summary}")
    if parent_summary:
        lines.append(f"Overview: {parent_summary}")
    prefix = "\n".join(lines)
    return f"{prefix}\n\n{raw_text}" if prefix else raw_text


def scalar_meta(d: dict) -> dict:
    out = {}
    for k, v in d.items():
        if v is None or v == "":
            continue
        if isinstance(v, bool):
            out[k] = v
        elif isinstance(v, int) and not isinstance(v, bool):
            out[k] = int(v)
        elif isinstance(v, float):
            out[k] = float(v)
        else:
            out[k] = str(v)
    return out


def load_ingest_cache() -> dict:
    if os.path.exists(INGEST_CACHE_PATH):
        with open(INGEST_CACHE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_ingest_cache(cache: dict) -> None:
    tmp = INGEST_CACHE_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(cache, f)
    os.replace(tmp, INGEST_CACHE_PATH)


def child_char_start(parent: str, child: str, search_from: int = 0) -> int:
    idx = parent.find(child, search_from)
    if idx < 0:
        idx = parent.find(child)
    return idx if idx >= 0 else search_from


class TableSummary(BaseModel):
    thoughts: str = Field(default="", description="Internal reasoning.")
    summary: str = Field(description="A 1-sentence summary.")


class ParentSummary(BaseModel):
    thoughts: str = Field(default="", description="Internal reasoning.")
    title: str = Field(description="A short descriptive title for this section.")
    summary: str = Field(description="A 1-2 sentence overview of what this section explains.")


class TableRowProse(BaseModel):
    thoughts: str = Field(default="", description="Internal reasoning about how to convert these table rows to prose.")
    prose_rows: list[str] = Field(description="One natural-language sentence per data row.")


async def summarize_content(agent: CoreAgent, text: str, mode: str = "table") -> BaseModel:
    if mode == "table":
        prompt = (
            f"Summarize what this data table lists in exactly ONE sentence.\n"
            f"Output ONLY the required JSON fields 'thoughts' and 'summary'. "
            f"Do NOT create an index, nested dictionary, or list.\n\n{text[:2000]}"
        )
        response_model = TableSummary
    else:
        prompt = (
            f"Summarize the core rules/concepts in this D&D rulebook snippet in exactly ONE sentence. "
            f"Output ONLY the required JSON fields 'thoughts', 'title', and 'summary'. "
            f"Do NOT create an index, nested dictionary, or list of items. Snippet:\n{text[:3000]}"
        )
        response_model = ParentSummary

    old_model = agent.response_model
    agent.response_model = response_model
    try:
        return await agent.ask(prompt)
    except Exception as e:
        print(f"⚠️ LLM generation failed for {mode} summary. Providing fallback. Error: {e}")
        if mode == "table":
            return TableSummary(summary="A table containing D&D data.")
        else:
            return ParentSummary(title="D&D Rules Section", summary="A section of the D&D ruleset.")
    finally:
        agent.response_model = old_model


def _md_row_cells(line: str) -> list[str]:
    line = line.strip()
    if line.startswith("|"):
        line = line[1:]
    if line.endswith("|"):
        line = line[:-1]
    return [c.strip() for c in line.split("|")]


def parse_markdown_table_to_prose(table_text: str) -> list[str]:
    """Fast programmatic conversion for standard markdown tables."""
    lines = [l.strip() for l in table_text.strip().split("\n") if l.strip()]
    if len(lines) < 3 or "|" not in lines[0]:
        return []

    headers = _md_row_cells(lines[0])
    if not headers:
        return []

    prose_rows = []
    for line in lines[1:]:
        if "|" not in line:
            continue
        values = _md_row_cells(line)
        if not values or all(set(v) <= {"-", ":", " "} for v in values):
            continue
        if len(values) != len(headers):
            return []

        pairs = [
            f"the {header} is {value}"
            for header, value in zip(headers, values)
            if value and value not in {"-", "—"}
        ]
        if pairs:
            sentence = ", ".join(pairs)
            prose_rows.append(sentence[0].upper() + sentence[1:] + ".")

    return prose_rows


async def convert_table_to_prose_rows(agent: CoreAgent, table_text: str) -> list[str]:
    """Convert table rows to prose. Try fast parsing first, fallback to LLM."""
    prose_rows = parse_markdown_table_to_prose(table_text)
    if prose_rows:
        return prose_rows

    old_model = agent.response_model
    agent.response_model = TableRowProse
    try:
        prompt = (
            f"Convert each DATA ROW in this D&D table into a single natural-language sentence.\n"
            f"Include all important values. Skip headers and separators.\n"
            f"Output ONLY the required JSON fields 'thoughts' and 'prose_rows'. "
            f"Do NOT create a nested dictionary where keys are table items.\n\n"
            f"Table:\n{table_text[:3000]}"
        )
        response = await agent.ask(prompt)
        return response.prose_rows if response.prose_rows else []
    except Exception as e:
        print(f"⚠️ LLM generation failed for table prose conversion. Returning empty list. Error: {e}")
        return []
    finally:
        agent.response_model = old_model


def extract_tables_from_markdown(text: str) -> list[str]:
    """Detect both standard markdown tables and ghost tables."""
    lines = text.split("\n")
    tables = []
    current_table = []
    empty_count = 0

    def is_row(line):
        cleaned = line.strip()
        is_md_table = line.count("|") >= 2
        is_ghost_row = bool(re.search(r"^\d+\s+[+−-]\d+(\s+|$)", cleaned)) or \
                       bool(re.search(r"\d+(?:d\d+)?", cleaned) and re.search(r"\b(GP|SP|CP)\b", cleaned, re.I))
        return is_md_table or is_ghost_row

    for i, line in enumerate(lines):
        if is_row(line):
            if not current_table:
                for j in range(max(0, i - 3), i):
                    prev_line = lines[j].strip()
                    if prev_line and not is_row(lines[j]):
                        current_table.append(lines[j])
            current_table.append(line)
            empty_count = 0
        elif line.strip() == "" and current_table:
            empty_count += 1
            if empty_count > 1:
                if len([l for l in current_table if l.strip()]) > 2:
                    tables.append("\n".join(current_table))
                current_table = []
                empty_count = 0
            else:
                current_table.append(line)
        else:
            if current_table:
                if i + 1 < len(lines) and is_row(lines[i + 1]):
                    current_table.append(line)
                else:
                    if len([l for l in current_table if l.strip()]) > 2:
                        tables.append("\n".join(current_table))
                    current_table = []
            empty_count = 0

    if current_table and len([l for l in current_table if l.strip()]) > 2:
        tables.append("\n".join(current_table))

    return tables


SEM = asyncio.Semaphore(1)
CACHE_LOCK = asyncio.Lock()


async def cached_parent_summary(agent: CoreAgent, p_text: str, cache: dict) -> str:
    key = "parent:" + content_sha256(p_text)
    hit = cache.get(key)
    if hit and hit.get("title") and hit.get("summary"):
        return f"{hit['title']}: {hit['summary']}"
    try:
        review = await summarize_content(agent, p_text, mode="parent")
        entry = {"title": review.title, "summary": review.summary}
    except Exception:
        entry = {"title": "D&D Rules Section", "summary": "A section of the D&D ruleset."}
    async with CACHE_LOCK:
        cache[key] = entry
        save_ingest_cache(cache)
    return f"{entry['title']}: {entry['summary']}"


async def cached_table_bundle(agent: CoreAgent, table: str, cache: dict) -> tuple[str, list[str]]:
    key = "table:" + content_sha256(table)
    hit = cache.get(key)
    if hit and "summary" in hit and "prose_rows" in hit:
        return hit["summary"], list(hit["prose_rows"] or [])
    try:
        t_review = await summarize_content(agent, table, mode="table")
        t_summary = t_review.summary
    except Exception:
        t_summary = "A table containing data."
    prose_rows = await convert_table_to_prose_rows(agent, table)
    entry = {"summary": t_summary, "prose_rows": prose_rows}
    async with CACHE_LOCK:
        cache[key] = entry
        save_ingest_cache(cache)
    return t_summary, prose_rows


async def process_parent_chunk_async(
    p_idx: int,
    p_text: str,
    path: str,
    parent_part_index: int,
    current_hash: str,
    table_agent: CoreAgent,
    child_splitter,
    cache: dict,
):
    p_id = f"{current_hash}_parent_{p_idx}"

    async with SEM:
        p_summary = await cached_parent_summary(table_agent, p_text, cache)

        tables = extract_tables_from_markdown(p_text)
        chunk_docs = []
        chunk_metas = []
        chunk_ids = []
        chunk_embeds = []

        for t_idx, table in enumerate(tables):
            t_summary, prose_rows = await cached_table_bundle(table_agent, table, cache)
            table_start = child_char_start(p_text, table.split("\n", 1)[0]) if table else 0

            if prose_rows:
                for r_idx, prose in enumerate(prose_rows):
                    if not prose.strip():
                        continue
                    r_id = f"{p_id}_table_{t_idx}_row_{r_idx}"
                    embed_text = build_embed_text(prose, path, p_summary, t_summary)
                    chunk_docs.append(prose)
                    chunk_embeds.append(embed_text)
                    chunk_metas.append(scalar_meta({
                        "source": "DandDRuleset.pdf",
                        "doc_hash": current_hash,
                        "parent_id": p_id,
                        "type": "table_row_prose",
                        "header_path": path,
                        "parent_summary": p_summary,
                        "table_summary": t_summary,
                        "raw_text": prose,
                        "embed_text": embed_text,
                        "child_char_start": table_start,
                        "parent_part_index": parent_part_index,
                    }))
                    chunk_ids.append(r_id)
            else:
                t_id = f"{p_id}_table_{t_idx}"
                embed_text = build_embed_text(t_summary, path, p_summary, t_summary)
                chunk_docs.append(t_summary)
                chunk_embeds.append(embed_text)
                chunk_metas.append(scalar_meta({
                    "source": "DandDRuleset.pdf",
                    "doc_hash": current_hash,
                    "parent_id": p_id,
                    "type": "table",
                    "header_path": path,
                    "parent_summary": p_summary,
                    "table_summary": t_summary,
                    "table_content": table,
                    "raw_text": t_summary,
                    "embed_text": embed_text,
                    "child_char_start": table_start,
                    "parent_part_index": parent_part_index,
                }))
                chunk_ids.append(t_id)

    child_chunks = child_splitter.split_text(p_text)
    search_from = 0
    for c_idx, child in enumerate(child_chunks):
        if not child.strip():
            continue
        start = child_char_start(p_text, child, search_from)
        search_from = start + max(1, len(child) - TEXT_CHILD_OVERLAP)
        c_id = f"{p_id}_child_{c_idx}"
        embed_text = build_embed_text(child, path, p_summary)
        chunk_docs.append(child)
        chunk_embeds.append(embed_text)
        chunk_metas.append(scalar_meta({
            "source": "DandDRuleset.pdf",
            "doc_hash": current_hash,
            "parent_id": p_id,
            "type": "text",
            "header_path": path,
            "parent_summary": p_summary,
            "raw_text": child,
            "embed_text": embed_text,
            "child_char_start": start,
            "parent_part_index": parent_part_index,
        }))
        chunk_ids.append(c_id)

    return p_id, p_text, chunk_docs, chunk_metas, chunk_ids, chunk_embeds


async def build_database():
    if not os.path.exists(DOC_PATH):
        print(f"❌ Error: Could not find document at {DOC_PATH}")
        return

    print("0. Initializing ChromaDB...")
    client = chromadb.PersistentClient(path=DB_DIR)
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
    )

    current_hash = calculate_md5(DOC_PATH)

    try:
        if collection.get(where={"doc_hash": current_hash}, limit=1)["ids"]:
            print(f"✅ Document already ingested into {COLLECTION_NAME} (hash: {current_hash[:8]}...). Skipping.")
            return
    except Exception:
        pass

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

    oversized_splitter = RecursiveCharacterTextSplitter(
        chunk_size=PARENT_CHUNK_SIZE, chunk_overlap=PARENT_CHUNK_OVERLAP
    )
    parents = []
    for doc in md_docs:
        if not doc.page_content.strip():
            continue
        path = header_path(doc)
        if len(doc.page_content) > PARENT_CHUNK_SIZE:
            parts = oversized_splitter.split_text(doc.page_content)
            for i, part in enumerate(parts):
                parents.append({"text": part, "header_path": path, "parent_part_index": i})
        else:
            parents.append({"text": doc.page_content, "header_path": path, "parent_part_index": 0})

    print(f"   -> Created {len(parents)} parent chunks.")

    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=TEXT_CHILD_SIZE, chunk_overlap=TEXT_CHILD_OVERLAP
    )

    documents, metadatas, ids, embed_texts = [], [], [], []
    cache = load_ingest_cache()
    print(f"   -> Ingest cache entries: {len(cache)}")

    litellm_kwargs = CoreAgent.load_litellm_kwargs_from_config("phi-agent")
    table_agent = CoreAgent(
        agent_id="table_summarizer",
        system_prompt="You are a data assistant. Summarize markdown tables into logical prose concisely.",
        litellm_kwargs=litellm_kwargs,
        one_shot=True,
        response_model=TableSummary
    )

    print("\nProcessing parent chunks (with table conversion)...")
    tasks = [
        process_parent_chunk_async(
            p_idx,
            parent["text"],
            parent["header_path"],
            parent["parent_part_index"],
            current_hash,
            table_agent,
            child_splitter,
            cache,
        )
        for p_idx, parent in enumerate(parents)
    ]

    for f in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Parent Chunks"):
        p_id, p_text, chunk_docs, chunk_metas, chunk_ids, chunk_embeds = await f

        c.execute(
            "INSERT OR REPLACE INTO parent_chunks (id, doc_hash, content) VALUES (?, ?, ?)",
            (p_id, current_hash, p_text),
        )

        documents.extend(chunk_docs)
        metadatas.extend(chunk_metas)
        ids.extend(chunk_ids)
        embed_texts.extend(chunk_embeds)

    conn.commit()
    conn.close()
    save_ingest_cache(cache)

    print(f"\nCreated {len(documents)} embeddable chunks.")

    device = get_rag_device()
    print(f"\n3. Loading encoder ({EMBED_MODEL}) on {device}...")
    encoder = SentenceTransformer(EMBED_MODEL, device=device)

    print("4. Embedding contextual prefixes...")
    embeddings = encoder.encode(embed_texts, normalize_embeddings=True, show_progress_bar=True)

    print(f"5. Storing in ChromaDB ({COLLECTION_NAME})...")
    batch_size = 5000
    for i in range(0, len(documents), batch_size):
        end = min(i + batch_size, len(documents))
        collection.upsert(
            documents=documents[i:end],
            metadatas=metadatas[i:end],
            embeddings=embeddings[i:end].tolist(),
            ids=ids[i:end]
        )

    print(f"\n✅ Ingestion complete! Stored in {DB_DIR} / {COLLECTION_NAME}")

    print(f"\n6. Saving BM25 corpus to {BM25_CORPUS_PATH}...")
    with open(BM25_CORPUS_PATH, "wb") as f:
        pickle.dump({"documents": documents, "metadatas": metadatas, "ids": ids}, f)
    print("✅ BM25 corpus saved.")


if __name__ == "__main__":
    asyncio.run(build_database())
