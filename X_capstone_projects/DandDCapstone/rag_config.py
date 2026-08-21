import os

import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

DOC_PATH = os.path.join(SCRIPT_DIR, "..", "documents", "DandDRuleset.pdf")
DB_DIR = os.path.join(SCRIPT_DIR, "chroma_db")
SQLITE_DB_PATH = os.path.join(SCRIPT_DIR, "parent_chunks.db")
BM25_CORPUS_PATH = os.path.join(SCRIPT_DIR, "bm25_corpus.pkl")
INGEST_CACHE_PATH = os.path.join(SCRIPT_DIR, "ingest_cache.json")

COLLECTION_NAME = "dnd_rules_multi_v3"
EMBED_MODEL = os.getenv("RAG_EMBED_MODEL", "BAAI/bge-large-en-v1.5")
RERANK_MODEL = os.getenv("RAG_RERANK_MODEL", "BAAI/bge-reranker-v2-m3")

PARENT_CHUNK_SIZE = 4000
PARENT_CHUNK_OVERLAP = 300
TEXT_CHILD_SIZE = 800
TEXT_CHILD_OVERLAP = 120

RRF_CANDIDATES = 40
CE_TOP_K = 12
CE_SCORE_FLOOR = 0.0
CE_MIN_CHUNKS = 4
CE_AMBIGUOUS_GAP = 1.0
MAX_CHILDREN_PER_PARENT = 3
TABLE_CE_BOOST = 1.15
TABLE_SKIP_EXPAND_CE = 0.5
MIN_TABLE_ROWS = 2

PARENT_WINDOW_BEFORE = 400
PARENT_WINDOW_AFTER = 800
PARENT_WINDOW_CAP = 1200

MAX_PASSES_GENERAL = 3
MAX_PASSES_SPECIFIC = 2
FOLLOWUP_OVERLAP_ABORT = 0.9

TABLE_HINTS = (
    "cost", "gp", "price", "damage", "hit die", "progression",
    "level ", "martial arts", "armor class", "ac ", "range",
)
TABLE_TYPES = frozenset({"table_row_prose", "table"})


def get_rag_device() -> str:
    return os.getenv("RAG_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")


def get_ce_device() -> str:
    return os.getenv("RAG_CE_DEVICE", "cpu")


def prefers_tables(query: str) -> bool:
    q = query.lower()
    return any(h in q for h in TABLE_HINTS)
