# D&D RAG Retrieval & Evaluation Pipeline (`run_evals_pre_release.py`)

This file runs the golden test datasets through the full Retriever-Agent swarm. Its job is to take a user question, find the exact right snippets from our document database, synthesize an answer, critique itself if it missed anything, and then judge if it got the final answer right.

## High-Level Execution Flow

The retrieval engine doesn't just pass the user's question directly to the database. Instead, it uses an advanced Multi-Agent search and refinement workflow to maximize accuracy.

### Process Step-by-Step

1. **Initialization**: Connects to the ChromaDB (Vector Search), loads the BM25 Index (Keyword Search), and connects to the SQLite Database (Parent text storage).
2. **Agent Swarm Setup**: Initializes 8 distinct `phi4` LLM personas, each with a specific strict job (Classifier, Decomposer, HyDE, Critic, Merger, Answer Gen, Judge, Reranker).
3. **Detailed Processing Loop** (For *every* question in the dataset):
   - **Question Classification**: 
     - *Classifier*: Evaluates if the question is "General" (broad concepts like "How do I play?") or "Specific" (data like "Cost of a Longsword?"). This sets the maximum number of Refinement Loops allowed (3 for General, 1 for Specific).
   - **Query Expansion**: 
     - *Decomposer*: Breaks the original question into 2-4 narrower sub-queries (e.g. "What does fireball do?" -> "fireball spell damage", "fireball saving throw").
     - *HyDE (Hypothetical Document Embeddings)*: Hallucinates a fake but realistic snippet of what the D&D rules for this answer *might* look like.
   - **Dual Retrieval**: 
     - *Vector Search (up to ~120 chunks)*: Scans ChromaDB for semantic matches. Retrieves the Top 20 chunks for *each* generated query (the original question, 2-4 sub-queries, and the HyDE text).
     - *BM25 Search (20 chunks)*: Scans the text corpus for exact keyword matches, returning the Top 20 results.
   - **Reciprocal Rank Fusion (RRF) (30 chunks)**: A math algorithm that combines the Vector and BM25 results, boosting the score of chunks that appear in both. It outputs a consolidated list of the **Top 30 chunks**.
   - **LLM Reranking (12 survive)**: The *Reranker* agent looks at the merged list of 30, physically reads the 300-character excerpts, and re-orders them from most to least relevant. Only the **Top 12 chunks** survive this culling and proceed to Context Expansion.
   - **Context Window Expansion (Windowing & Token Jamming)**: The Top 12 300-character chunks are too small to provide the full picture. 
     - The system reads each chunk's `parent_id` and queries the SQLite database to fetch the **Full Parent Rule Section** (which is up to 4,000 characters long).
     - **Deduplication**: If 4 of the winning chunks all come from the same broader "Combat Rules" section, the system only loads that massive parent block *once*.
     - **Final Memory Size**: The 12 tiny chunks effectively \"bloom\" into roughly **2,000 to 8,000 tokens** of highly-relevant contextual D&D rules. This chunk of text is exactly what gets \"jammed\" into the Answer prompt.
   - **Initial Generation**: The *Answer Generator* agent reads the massive expanded context and writes a strict, factual answer.
   - **Critique & Refinement Phase (Dynamic Multi-Pass)**: 
     - The *Critic* agent compares the generated answer to the context. Did the answer forget to list the Gold Piece cost? 
     - If yes, the Critic provides a `follow_up_query`.
     - The system kicks off the **Entire Full Retrieval Pipeline** from scratch purely for the follow-up question (running it through the Decomposer, HyDE, Vector/BM25, RRF, Reranker, and expanding the Parent Section).
     - The *Answer Generator* extracts the missing fact, and the *Merger* agent stitches it smoothly into the original answer.
     - The newly fetched context is locked into memory along with the merged answer, and the Critic reviews it *again*. This loop spins **up to 3 times** for General questions, guaranteeing exhaustive answers.
   - **Final Judgement**: The *Answer Judge* compares the generated Final Answer to the 'Golden Expected Answer' provided by the dataset and scores it pass/fail.

---

## Visual Model (Mermaid Diagram)

Below is a visual flowchart representing the complete retrieval, generation, and critique pipeline:

```mermaid
flowchart TD
    Start[User Question] --> Expand Decomposer
    Start --> Expand HyDE
    
    Expand Decomposer[Decomposer Agent:\nGenerate Sub-Queries] --> Vector
    Expand HyDE[HyDE Agent:\nGenerate Fake Answer] --> Vector
    Start --> Vector
    Start --> BM[BM25 Exact Keyword Search]
    Expand Decomposer --> BM
    
    Vector[Vector Semantic Search:\nChromaDB] --> RRF
    BM --> RRF[Reciprocal Rank Fusion\nMerge & Deduplicate Results]
    
    RRF --> Rerank[Reranker Agent:\nRead & Re-order Chunks]
    
    Rerank --> ExpandWindow[Context Window Expansion:\nFetch Full Parent Clauses from SQLite]
    
    ExpandWindow --> AnswerPass{Answer Generator Agent:\nDraft Initial Answer}
    
    AnswerPass --> Critic[Critic Agent:\nAudit Answer vs Context]
    
    Critic --> IsComplete{Is Answer Complete?}
    
    IsComplete -- No, Missed Fact --> FollowUp[Critic provides Follow-up Query]
    FollowUp --> RefineRet[Trigger Full Retrieval Pipeline:\nDecomposer, Dual Search, RRF, etc.]
    RefineRet --> RefineAns[Answer Gen reads missing fact]
    RefineAns --> Merger[Merger Agent:\nStitches new info into original answer]
    Merger --> LoopBack[Loop Back to Critic\nUp to 3x for General Qs]
    LoopBack --> Critic
    
    IsComplete -- Yes or Max Passes Reached --> FinalAnswer[Final Compiled Answer]
    
    FinalAnswer --> Judge[Judge Agent:\nScore generated vs expected answer]
    Judge --> End([Log Output & Score])
```
