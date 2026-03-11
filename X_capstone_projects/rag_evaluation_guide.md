# Advanced RAG & Evaluation Study Guide

🏆 **Step 1: Building the "Golden Data Set"**
You cannot measure success without a North Star. This is your "Golden Dataset."
- **What it is:** A collection of 100+ prototypical questions, reference (perfect) answers, and specific keywords that must appear in the retrieved context. 📂
- **Sources:** 
  - *End Users:* The absolute best source. Real questions from production.
  - *Synthetic Data:* Using models (like Claude or GPT) to generate questions from your documents.
- **Categorization:** Tagging questions (e.g., Direct Fact, Temporal, Spanning, Holistic) helps you identify exactly where your RAG is weak.

📏 **Step 2: Retrieval Metrics (The Search)**
Retrieval evals measure how good your "Search" is before the LLM even sees the data.
- **MRR (Mean Reciprocal Rank):** The "Gold Standard" for search.
  - If the correct answer is the 1st chunk, score = 1.
  - If it's the 2nd chunk, score = 1/2.
  - If it's the 3rd, score = 1/3.
  - *Target:* As close to 1.0 as possible. 🎯
- **Recall@K:** Did the correct information appear anywhere in the top K chunks? (e.g., Recall@5).
- **Precision@K:** Out of the K chunks we grabbed, how many were actually relevant? High precision reduces "context pollution."
- **nDCG:** Measures if the most relevant chunks are ranked higher than the less relevant ones.

⚖️ **Step 3: Answer Metrics (LLM as a Judge)**
Retrieval is the "Means"; the Answer is the "End." We use a stronger LLM (the Judge) to grade our RAG's output.
*The Three Pillars of Answer Quality:*
- **Accuracy:** Is the answer factually correct?
- **Completeness:** Does it answer all parts of the question? (e.g., if asked for a full name, does it provide "Maxine Thompson" or just "Maxine"?)
- **Relevance:** Does it provide only what was asked, or is it full of "fluff"? 🌾

🧪 **The Experiment Results (The "Whack-a-Mole")**
*Example eval run results:*
- **Baseline (1000 chunk, K=5):** MRR 0.7298
- **Smaller Chunks (500 chunk, K=10):** MRR 0.7604 (Improved. More granular search helped)
- **Larger Chunks (1667 chunk, K=3):** MRR 0.7475
- **Markdown Splitter (Variable, K=3):** MRR 0.7383 (Disappointing; chunks were too big)
- **OpenAI Small 3 (500 chunk, K=10):** MRR 0.7849
- **OpenAI Large 3 (500 chunk, K=10):** MRR 0.7903 (Champion. High dimensionality wins)

❓ **Combined Q&A Study Guide**
**Q1:** Why is RAG often called "Empirical"?
**A:** Because there is no single "correct" configuration. You must experiment with chunk sizes and encoders, measuring the results against a test set to find what works for your specific data.

**Q2:** What is the "Achilles' Heel" of RAG?
**A:** Spanning and Holistic questions. These require looking across many documents or understanding the "big picture," which is difficult when the system only retrieves a few small chunks at a time.

**Q3:** If your MRR is high but your Accuracy is low, what is likely the problem?
**A:** This suggests the Retrieval is working (the right chunks are being found), but the Generation is failing. The LLM might be too weak, the system prompt might be bad, or the chunks might be too small to contain the full context.

**Q4:** What is the benefit of a .jsonl file over a standard .json file for evals?
**A:** JSONL (JSON Lines) is easier to append to and more memory-efficient for large datasets, as you can process the file line-by-line without loading a massive list into memory.

**Q5:** Describe "LLM as a Judge."
**A:** It is the practice of using a high-performing model (like GPT-4o) to evaluate the responses of your RAG system by comparing them to a "Golden" reference answer and scoring them on accuracy, completeness, and relevance.

---

🚀 **Advanced RAG: The "Pro Day" Philosophy**
The transition from basic to advanced RAG isn't about finding a "magic" architecture; it’s about moving from guesswork to scientific experimentation.
- **The Golden Rule:** Set your metrics (MRR, nDCG, Recall), build a "Golden" test set, and run experiments. Only the data can tell you which technique works for your specific PDF/document set.

🛠️ **The 10 Essential Advanced RAG Techniques**
1. **Chunking R&D:** Testing semantic splitters vs. fixed sizes.
2. **Encoder Selection:** Swapping models (e.g., BGE-M3 vs. GTE).
3. **Prompt Engineering:** Adding static context (date, persona) to the RAG prompt.
4. **Doc Pre-processing:** Using an LLM to rewrite tables/text into "query-friendly" text.
5. **Query Rewriting:** LLM cleans up user questions before searching.
6. **Query Expansion:** Generating 3-5 versions of one question.
7. **Re-ranking:** A "Cross-Encoder" re-orders the top 20 results.
8. **Hierarchical RAG:** Summarizing docs at multiple levels (Chapter -> Page -> Chunk).
9. **Graph RAG:** Mapping relationships between chunks (e.g., Manager -> Employee).
10. **Agentic RAG:** Letting an LLM decide which tool (SQL, Vector, Web) to use.

💻 **Technical Implementation: Going "Native" (No LangChain)**
- **Format Conversion:** Don't feed PDFs to LLMs. Use Python libraries to convert PDF ⮕ Markdown.
- **Semantic Chunking:** Instead of splitting by characters, use an LLM (e.g., GPT-4o-mini) to identify where sections logically end.
- **Structured Output:** Use Pydantic classes to force the LLM to return specific JSON schemas.

🧠 **Modern Additions & Technical Clarifications**
- **Parallelizing Chunking:** Yes. If using a local encoder (like BGE-M3), use `concurrent.futures`.
- **Hierarchical Strategy:** Document Path + Summary approach. By including the file path and a high-level summary in the metadata of every chunk, the LLM always knows "where it is".
- **Privacy (Ollama):** Running local embedding models and local LLMs ensures data never leaves your infrastructure.

🛠️ **Advanced RAG: Native Implementation & Strategy Guide**

**1. Handling Multiple Formats**
Markdown serves as the universal "middle-layer" for RAG. Convert everything into Markdown before chunking. It preserves hierarchical structures (headers, lists, tables).

**2. The Custom Chunk Class (Going "Native")**
Use Pydantic to enforce a strict schema:
```python
from pydantic import BaseModel, Field
from typing import Optional

class ChunkMetadata(BaseModel):
    file_path: str                 # 📍 Where did this come from?
    doc_summary: str               # 📝 High-level summary of the whole file
    parent_heading: str            # 📑 Section header
    chunk_index: int               # 🔢 Ordering

class RAGChunk(BaseModel):
    id: str = Field(...)
    content: str                   # 📄 Raw text
    metadata: ChunkMetadata        # 📦 Context payload
    embedding: Optional[list[float]] = None
```

**3. The Implementation Strategy**
- **Step 1:** Configuration & Native Framework (Pydantic, raw chromadb).
- **Step 2:** Ingestion & Parsing Pipeline (Markdown conversion, Hierarchical Context).
- **Step 3:** Building the Golden Dataset & Evals (MRR, Recall@K, nDCG).
- **Step 4:** The Evaluation Loop.
