# Challenges Along the Way

This document tracks significant technical hurdles, architectural decisions, and "gotchas" discovered while building the AI Agent infrastructure.

---

## 1. The "Thinking" Suppression Issue (Structured Output vs. Reasoning Models)

**The Problem:**
When using reasoning models (like `deepseek-r1`) or trying to get standard models to explain their rationale, the agents were skipping the "thinking" phase entirely and just spitting out the final answer. For `deepseek-r1`, which is natively trained to output its reasoning inside `<think>...</think>` tags, this behavior disappeared completely.

**Why It Happened:**
The issue was caused by strict **Structured Output enforcement**. 
By using Pydantic models with LiteLLM to force the output into a strict JSON format (e.g., `CharacterIdentity` or `EvaluationResult`), the LLM realized that including XML tags like `<think>` would invalidate the JSON parser. To avoid throwing a formatting error, the models suppressed their natural reasoning behavior entirely to ensure the output remained valid JSON.

**How We Overcame It:**
We implemented **Schema-Driven Chain of Thought**.
Instead of begging the LLM to "think step-by-step" in the system prompt, we built a sandbox for reasoning directly into the JSON schema itself. 

We added a `thoughts: str` field as the *very first variable* in our Pydantic models:

```python
class CharacterIdentity(BaseModel):
    thoughts: str = Field(default="", description="Your internal reasoning for picking this class and role.")
    name: str
    actual_class: str
    # ...
```

**Why This Works:**
1. **For Reasoning Models (`deepseek-r1`):** It gives the model a safe, valid JSON string field to dump its internal reasoning without breaking the parser using XML tags.
2. **For Standard Models (`llama3.1`, `phi4`):** Because `thoughts` is the first key in the JSON object, the LLM is physically forced to generate a paragraph of reasoning *before* it generates the `name` or `actual_class`. This acts as an artificial Chain of Thought, mathematically increasing the quality of the final fields by forcing the model to "talk it out" first.
3. **Cleaner Prompts:** It removes the need to clutter the system prompt with formatting and reasoning instructions.

---

## 2. The Streaming Structured Output Problem

**The Problem:**
While it was possible to get the agent's internal reasoning (via the `thoughts` field), we couldn't easily stream those thoughts "live" to the console character-by-character as the LLM generated them. We had to wait for the entire generation to finish before printing the JSON.

**Why It Happened:**
When you ask an LLM to output a structured Pydantic object, it natively enforces strict JSON. If you set `stream=True`, the LLM sends back partial, "broken" fragments of a JSON string over the network piece by piece (e.g., `{"thoug` -> `hts": "I shou` -> `ld pick a Wiz`). Standard JSON parsers (like Python's `json.loads` or Pydantic itself) immediately crash if you try to parse a JSON string that isn't fully closed.

**How to Overcome It (If Necessary):**
While waiting for the full response and parsing it cleanly is the most stable approach for a CoreAgent MVP, streaming structured output *is* possible with more advanced parsing. It requires either rolling a custom "Partial JSON Parser" or adopting a specialized library (like `jiter` or `instructor`) designed specifically to intercept broken JSON chunks and safely yield the value of specific fields incrementally before the final bracket closes.

---

## 3. Small LLM Hallucinations and Dictionary Lookups

**The Problem:**
When the Judge rejected a party and instructed the next agent (a smaller 8B model like `llama3.1`) to pick a Flex Path (e.g., "The Utility: A Bard or Druid"), the smaller model misinterpreted the instructions. Instead of outputting the actual D&D class (e.g., `Bard`), it literally output `"actual_class": "The Utility"`.
Because `"The Utility"` did not exist in the `CLASS_ATTRIBUTES` dictionary in `constants.py`, the script crashed with a `TypeError` when it tried to look up stats for it.

**Why It Happened:**
Small parameter models (like 8B) struggle with complex, multi-layered constraints. When given Judge feedback combined with long system prompt instructions, their attention mechanisms can get confused about exactly which set of words constitutes a valid "Class". To them, "The Utility" seemed like a valid answer based on the context.

**How We Overcame It:**
We implemented a **Graceful Fallback Mechanism**.
1. **Stronger Prompting:** We added bold, explicit directions: `IMPORTANT: You MUST select an actual_class EXACTLY as it appears in the Available Classes list below.`
2. **Dictionary Safety Checks:** Instead of blindly assuming the LLM's output is safe to use as a dictionary key, we now check the result of `CLASS_ATTRIBUTES.get(choice.actual_class)`. If it returns `None`, we print a `HALLUCINATION DETECTED` warning and skip that agent's turn. The loop simply rolls over to the next agent in the roster to try again, completely avoiding application crashes!

---

## 4. Advanced RAG Techniques for Monolithic Documents

While reading an entire PDF into markdown and chunking it by character count provides a solid baseline for RAG (Retrieval-Augmented Generation), sometimes absolute-precision is needed (e.g., to stop the DM Agent from confusing the rules for *Stealth* with the rules for *Invisibility*). Here are three advanced techniques considered for the D&D Ruleset:

### 4a. Multi-Vector Retrieval (Table/Image Summarization)
Tables are notoriously difficult for standard vector databases to retrieve cleanly because they lack natural prose structure.
**How it works:**
1. You split the document into raw **Text Chunks** and **Table Chunks**.
2. You pass every **Table Chunk** to a small, fast LLM (like `llama3.1`).
3. You prompt the LLM: *"Write a 2-sentence summary of what this table contains."* (e.g., "A table listing martial weapons and their gold costs").
4. You **embed the LLM's prose summary** because vector databases native understand prose better than raw grids.
5. In the Vector Database, you link the summary vector to the raw `| Markdown | Table |` as its metadata.
6. When the user queries "Weapon prices", the database matches against the LLM's prose summary, but returns the raw Markdown table to the DM Agent for perfect formatting.

### 4b. Parent-Document Retrieval (Hierarchical Chunking)
In D&D, a rule might take up an entire page, but a standard chunk size is only ~1,500 characters. If the database returns the middle chunk, the DM Agent might miss the vital context found at the beginning of the page.
**How it works:**
1. **The Parent Chunks:** You slice the document into large 3,000-character chunks (essentially whole pages) and assign each a unique ID (e.g., `parent_chunk_001`). You save these chunks into a standard Key-Value Storage Engine (like a JSON file or SQLite database).
2. **The Child Chunks:** You then slice `parent_chunk_001` again into tiny 300-character chunks (just 1 or 2 sentences each). 
3. **The Search Engine:** You embed those tiny child chunks into your Vector Database (ChromaDB) for razor-precision searching. Inside the metadata for every tiny chunk, you inject `{"parent_id": "parent_chunk_001"}`.
4. **Query Time:** When a player asks a rule question, ChromaDB fetches the tiny 300-character chunk that perfectly matches the query. Your script reads the `parent_id` metadata, reaches into the SQLite database, and hands the DM the massive 3,000-character parent chunk so they have all surrounding context!

### 4c. HyDE (Hypothetical Document Embeddings)
HyDE is a highly effective query-time optimization that mitigates vocabulary mismatch without needing to re-index the database.
**How it works:**
1. A player asks the DM: *"Can I cast a spell and attack in the same turn?"*
2. Before your Python code talks to the Vector Database, it pings an LLM and asks: *"You are a D&D rulebook. Write a fake excerpt explaining how casting a spell and attacking works."*
3. The LLM generates a fake rule: *"Casting a spell requires an action. However, some house rules allow attacks as a bonus action."* (The LLM might hallucinate half the answer, but the *structure* and *vocabulary* will look like a real D&D rule).
4. You embed that *"Hypothetical Answer"* into a vector! 
5. You search the database using the Hypothetical Answer's vector rather than the User's question's vector. Because the structure of the fake answer perfectly matches the structure of the real rulebook, the database finds the exact rule with extremely high accuracy.

### 4d. Window Retrieval (Surrounding Chunks)
A common problem with raw text chunking is that a vital piece of information (like a massive Monster Stat Block) might accidentally get sliced exactly in half by the LangChain text splitter. 
**How we mitigate this:**
1. **The Built-In Protection (Chunk Overlap):** When indexing, we set LangChain to chunk at 1,500 characters, but with a `300` character overlap. That means the bottom 300 characters of Chunk A are duplicated as the top 300 characters of Chunk B. No matter where the knife falls, the LLM will always get enough surrounding context (like the Monster's Name) to figure out what it's looking at.
2. **The Advanced Upgrade (Window Retrieval):** If the LLM analyzes a fetched chunk and determines it is incomplete (e.g. it sees the start of an Orc but not its attacks), we can execute Window Retrieval on the fly. Because we injected `{"chunk_index": i}` into the metadata of every chunk during database generation, Python can easily look at the fetched chunk (e.g. `chunk_index: 45`), reach straight back into ChromaDB without a vector search, explicitly request `chunk_index: 44` and `chunk_index: 46`, logically glue all three strings together chronologically, and hand the LLM the massively expanded text block.

---

## 5. Building and Evaluating the D&D RAG Pipeline

### 5a. What We Built

We implemented a **Multi-Vector Hierarchical RAG Pipeline** for the D&D 5.2 Ruleset PDF:

1. **PDF → Markdown Extraction:** Used `pymupdf4llm` to convert the 300-page PDF into structured Markdown, capturing table layout wherever possible. Saved to `documents/DandDRuleset.md` (1.37 MB, ~350,000 estimated tokens).

2. **Hierarchical Chunking (Parent/Child):**
   - **540 Parent Chunks** (~3,000 chars each) stored in SQLite (`parent_chunks.db`).
   - Each parent is further split into small **Child Chunks** (~300 chars) embedded into ChromaDB for precision retrieval.
   - At query time, the matched child chunk's `parent_id` is used to fetch the full parent for rich LLM context.

3. **Agentic Parent Summarization (Context Injection):**
   - Every Parent Chunk is summarized by a local `phi4` LLM agent, producing a `title + summary`.
   - This summary is **injected into every child chunk's metadata** as `parent_summary`, giving the vector search a "semantic GPS" for each tiny chunk.
   - Implemented with **async parallel processing** (`asyncio.Semaphore(10)`) to process 540 chunks concurrently, tuned for a 16GB Mobile RTX 4090.

4. **"Ghost Table" Detection:**
   - The D&D ruleset contains tables not formatted with Markdown pipe characters (`|`), called "Ghost Tables" (e.g., the Ability Score Modifier table on page 6).
   - Added regex-based detection for rows matching patterns like `18  +4` (score+modifier pairs).
   - These are extracted, summarized by the LLM, and stored with their raw content as metadata so exact table formatting is preserved for the DM agent.

5. **Golden Datasets (Multiple Models Processed):**
   - Datasets were generated across multiple models (Gemini, Grok, Phi) to test vocabulary variance.
   - **Important Observation on AI Psychology:** Less powerful local models tended to produce "Human-like Entity Queries" (e.g., *"What is the exact gold cost of True Resurrection?"*), which are easy for semantic math to find. However, powerful reasoning models like Phi-4 and `gemini-3.1-pro` outputted **Semantic/Logical Trap questions** (e.g., *"How many feet can a Small Beast move in one turn according to general rules?"*). These questions test whether the RAG pipeline will mistakenly hallucinate a general rule when speed is actually tied to specific creature stat blocks.

6. **Evaluation Script (`run_evals.py`):**
   - Embeds each golden question using the same `BAAI/bge-large-en-v1.5` encoder used during ingestion.
   - Scores each using **MRR (Mean Reciprocal Rank)** and **Recall@K**.
   - Uses Unicode-normalized, **fuzzy keyword matching** (≥60% of keywords must match) to avoid false negatives from Unicode minus signs (`−` vs `-`) and minor phrasing differences.

---

### 5b. Current Accuracy (as of March 8, 2026)

#### Initial (Incorrect) Eval — Child Chunk Only
| Metric | K=5 | K=10 |
|---|---|---|
| **MRR** | 0.505 | 0.509 |
| **Recall@K** | 61.7% | 64.5% |

#### Corrected Eval — Child + Parent Chunk (Mirrors Reality)
| Metric | K=10 |
|---|---|
| **MRR** | **0.856** |
| **Recall@K** | **92.5%** |

**Total Questions:** 107 | **Successful Retrievals @10:** 99/107

**The Key Insight:** The initial eval only scored the tiny 300-char child chunk against the keywords. But in production, the DM Agent receives the full 3,000-char **parent chunk** from SQLite. Once the eval was fixed to look up and check the parent chunk (via the child's embedded `parent_id`), the true accuracy emerged. The retrieval pipeline was always working — the scoring was just wrong.

---

### 5c. Known Accuracy Failures & Root Causes

| Failure Type | Example | Root Cause |
|---|---|---|
| **Exact Phrase Mismatch** | "Stake to the Heart" weakness | Stat block uses slightly different phrasing than the query |
| **Deep Stat Block Burial** | Vampire/Sphinx specific traits | Monster traits buried mid-stat-block; child chunk may only contain adjacent text |
| **Multi-Condition Rules** | "Cast two spells in same turn" | Rule spans multiple child chunks; no single chunk contains all keyword signals |
| **Unicode Normalization** | Ability score modifier "-5" | PDF rip uses Unicode minus `−` (U+2212); naive string match misses it |
| **Output-Truncated Keywords** | "Power Word Heal conditions" | Golden dataset keyword too specific for any single small child chunk |

The **ceiling is not chunking completeness** — the database is fully ingested (41 MB ChromaDB, 540 parent chunks). The gap is a **semantic/lexical mismatch** between query vocabulary and chunk vocabulary.

---

### 5d. Accuracy Improvement Roadmap

#### 🟢 Quick Wins (Low Effort, +5-10%)
1. **HyDE (Hypothetical Document Embeddings) [HIGH PRIORITY]:** Before querying the DB, ask an LLM to write a fake "D&D rulebook excerpt" answering the question. Embed the fake answer instead of the raw query. This bridges vocabulary gaps between player language and rulebook language.
2. **Increase Child Chunk Overlap:** Current overlap is 300 chars. Bumping to 500 chars ensures more context bleeds across chunk boundaries, helping rules that span paragraphs.
3. **Re-rank with Cross-Encoder:** After retrieving top-K by vector similarity, pass the (query, chunk) pairs through a `cross-encoder/ms-marco-MiniLM` re-ranker. Better precision at Rank 1.

#### 🟡 Medium Effort (+10-15%)
4. **Hybrid Search (BM25 + Vector):** Add a BM25 (keyword-based) index alongside the vector index. Linearly combine the scores (`0.5 * BM25 + 0.5 * Vector`). BM25 excels at exact monster/spell name lookups that semantic search misses.
5. **Master Router (Dynamic Retrieval + HyDE) [HIGH PRIORITY]:** Combine Query Routing and HyDE into a single pass to save computation time! Before querying the vector database, pass the player's question to a fast, cheap local model (like Llama 3.1 8B or Phi-4) with a system prompt to evaluate the query parameters and output a JSON object.
    * **Scope:** The Router determines if the question is `SPECIFIC` (e.g., "What is a goblin's AC?") or `BROAD` (e.g., "What does the DM do?").
    * **HyDE Generation (Hypothetical Document Embeddings):** The Router writes a fake 3-sentence excerpt from the official D&D rulebook answering the question. We embed this *fake answer* instead of the player's slang-filled query to bridge vocabulary gaps.
        * **Case Study: The "Longsword" Problem (March 8, 2026)**
            * **The Problem:** In SPECIFIC lookups for items like "Longsword", HyDE would generate a fake price (e.g., "5 gold"). This "fake news" in the query embedding would actually pull the Vector DB *away* from the actual weapons table (where the real price is 15 gold) and toward general lore or "Magic Item" entries that mentioned similar numbers.
            * **The Solution: Hybrid Embedding (Query + HyDE)**
                * We modified the search to embed both the raw user question AND the HyDE excerpt.
                * **The Query acts as an Anchor:** It ensures the actual keywords ("Longsword", "15 gp") are prioritized.
                * **The HyDE acts as a Compass:** It helps for conceptual questions (like the T-Rex bite) where the specific keywords might be missing from the user's prompt.
            * **Result:** Longsword retrieval improved from "Failure" to a consistent **Rank 3 hit**, even with HyDE hallucinations.
        * **Case Study:** When evaluating *"what kind of damage does a T rex bite do?"*, the player's slang ("T rex") would normally fail a vector search. The Router translated it into an official-sounding HyDE excerpt: *"The mighty Tyrannosaurus Rex... inflicts grievous damage... Its bite attack deals a significant amount of piercing damage..."* This perfectly formulated D&D vocabulary ensures the Vector DB instantly locates the Tyrannosaurus Rex stat block.
    * **Context Scaling (Dynamic K):** If the Router returns `{"scope": "SPECIFIC"}`, the Python script dynamically sets `K=3` (because an exact lookup needs few chunks). If `BROAD`, it sets `K=15` (for maximum semantic coverage).
    * **Metadata Filtering:** The Router also detects the Category (e.g., `spells`, `monsters`, `rules`). The Python script passes that directly into ChromaDB's `where` filter (`where={"type": "monster_stat_block"}`) to instantly eliminate noise from irrelevant chapters.
        * **Case Study:** When evaluating "Which creature has an Initiative bonus of +3 and what is its AC?", Vanilla Vector Math pulled the *Animate Objects* spell (which creates temporary creatures with those stats) into Ranks 1 and 2, burying the actual monster hit. By pre-filtering for Category, the spelling noise vanishes and the real monster jumps to Rank 1.

#### 🔴 High Effort (+15-20%)
6. **Context Windowing (Neighboring Chunk Expansion):** When the LLM Judge determines the retrieved chunk is conceptually close but misses the exact answer (like a table that spilled over into the next chunk), dynamically pull the immediate adjacent parent chunks (e.g., `parent_331` and `parent_333` if the hit was `parent_332`) from SQLite.
    * **Update (March 8, 2026):** We implemented Context Windowing and it completely solved the "spillover" issue! For example, when searching for the 'Invisibility' spell, the vector database fetched a massive block of "Spell Descriptions" (Chunk 1) but missed the exact subsection. The fallback windowing triggered, pulled the adjacent chunks, and instantly found the 'Invisibility' spell description just one chunk away, turning a failed query into a 1.0 MRR success. It also nailed the fragmented Bag of Beans effect table lookup!
    * **Metric Note:** Generating a hit using Context Windowing on a chunk is counted generously as an unaltered MRR hit for that rank (labeled `MRR+` in our logs), to reflect that while computation/context was spent fixing the result, the Vector DB correctly pointed the system to the right location initially.
7. **Web Search Fallback (Serper API):** Even with 540 chunks, some esoteric rules might be missing from the local database. If the RAG pipeline exhausts all `K` chunks and their windows without finding an answer, default to a fallback module.
    * The system extracts the core question parameters isolated by the Master Router and executes a live web search (e.g., via the Serper API).
    * It then prompts the LLM to read the live web results (e.g., from Sage Advice or Reddit) and answer the user, gracefully preventing the dreaded "I don't know" dead-end.

---

## 6. Silent Retrieval Failures (Metadata Type Inconsistency)

**The Problem:**
After implementing the Master Router, we added **Metadata Filtering** to target specific chapters (e.g., `monsters`, `spells`). However, the RAG evaluations suddenly began failing on simple, previously working questions with a "Missing from context block" error.

**Why It Happened:**
ChromaDB's `where` filters are **silent on mismatch**.
If you query with `where={"type": "monster_stat_block"}` and your database actually stored chunks with `{"type": "text"}`, ChromaDB does not throw an error. It simply returns an empty list `[]` of documents. The Python script then passed an empty string to the LLM Judge, which correctly reported that the answer was "not in context." Because there was no error message, it appeared as if the Vector Search had simply "failed" to find the monster, when in reality, it was being forbidden from looking at them by a mismatched key.

**How We Overcame It:**
We audited the `build_rag_db.py` ingestion script and discovered that metadata was only being tagged as `"table"` or `"text"`, never as specific categories. We temporarily removed the destructive `where` filters in the evaluation script to restore retrieval accuracy until a full re-indexing with richer metadata tags is completed.

---

## 7. The Agent Memory "Memory Leak" (Context Bleeding)

**The Problem:**
During sequential evaluations in `run_evals_routed.py`, the LLM Judge began hallucinating wildly. By Question 5, it started citing rules about Aboleths when the question was about casting multiple spells.

**Why It Happened:**
We discovered **Context Bleeding** in the agentic loop.
The `CoreAgent` class was designed with a `working_memory` array that persists between `ask()` calls to allow for multi-turn conversation. However, in an automated evaluation loop, this meant that the Judge's memory was literally filling up with every single question and every massive context block from the previous 4 questions. By Question 5, the model was reading a 50-page "super-prompt" containing every rule it had seen so far, causing its attention mechanism to snap and pull answers from irrelevant rule blocks.

**How We Overcame It:**
We implemented a **Memory Clear** at the start of every iteration in the evaluation loop. By calling `judge.working_memory.clear()` and `router.working_memory.clear()` before every new question, we effectively "lobotomize" the agent for the next task, ensuring it only sees the specific question and context currently being evaluated. This restored perfect reasoning precision.

---

## 8. The "Strict Judge" vs "Messy Tables" (Analysis Paralysis)

**The Problem:**
We found that when the LLM Judge was told to be "Strict", it would fail to find answers buried in raw table rows (e.g., `Longsword 1d8 Slashing ... 15 GP`). Because the context lacked formal column headers, the Judge assumed the data was "just numbers" and not a valid, complete answer.

**How We Overcame It:**
We liberalized the Judge's system prompt to explicitly state: *"Even if the answer is buried in a large block, return answer_found=True as long as the relevant rule, stat, or number IS present somewhere in the text."* This gave the local Phi-4 model "permission" to ignore the lack of formatting and focus purely on the existence of the data. retrieval success for items like the "Longsword" immediately jumped from 0% to nearly 100%.

---

## 9. The "Format Bias" Penalty (Cross-Encoder vs Tables)

**The Problem:**
Even after successfully retrieving the exact table row for the "Longsword" using a Python-side case-insensitive string match, the RAG pipeline inexplicably deleted the correct chunk from the Top 10 results right before giving it to the Judge!

**Why It Happened:**
We added a `cross-encoder/ms-marco-MiniLM-L-6-v2` re-ranker to improve precision. However, this neural network was trained on natural English sentences (like Bing search queries and Wikipedia answers). When it looked at a raw, comma-less table row (`Longsword 1d8 Slashing Versatile (1d10) Sap 3 lb. 15 GP`), the AI literally hallucinated that it was "unreadable gibberish" and assigned it a massive negative mathematical penalty (`-6.23`). Because the score was so abysmal, the table chunk plummeted to Rank 25+ and was instantly truncated from the `top_k=10` list. 

**How We Overcame It:**
We implemented a **Format Bias Override**. The Python script now deeply tracks the index of any chunk that was found via `Track B`'s Exact Keyword match. Before the Cross-Encoder sorts the chunks, the script iterates through those indices and artificially adds a massive `+10.0` points to their scores. This completely counteracts the neural network's bias against non-prose formatting, guaranteeing that explicitly verified keyword hits smoothly float into the Top 10!

**Bonus Bug (The ID Desync):** 
While fixing this, we also discovered that the merging of Track A (HyDE) and Track B (Exact Match) failed to sync the Chroma `ids` array. This caused the script to accidentally use the ID of the `Hobgoblin Captain` to look up the SQLite parent text for the `Longsword` table! By correctly zipping and syncing the arrays (`docs`, `metas`, **and** `ids`) during the sort phase, the context window mapped correctly.

---

## 10. The "Split Answer" Problem (Answers Spread Across Multiple Chunks)

**The Problem:**
Some questions require pulling together information that is physically stored across **two or more separate child chunks** in the database. A perfect example is `Q21: "What are the four Vampire Weaknesses listed in the vampire stat block?"`.

The retrieval pipeline correctly fetched all the relevant chunks — but no single child chunk contained all four weaknesses simultaneously:
- **Chunk 1 / Chunk 5:** contained `Forbiddance` + `Running Water`
- **Chunk 2:** contained `Stake to the Heart` + `Sunlight`

Because the judge was clearing its memory before every single chunk evaluation (`judge.working_memory.clear()` inside `check_chunk()`), it evaluated each chunk in complete isolation. Every call saw only one partial set of weaknesses and correctly judged it incomplete. The pipeline exhausted all 15 ranks and failed — even though ALL the evidence was present in the database.

**Why It Happened:**
The original design chose strict per-chunk isolation (`judge.working_memory.clear()`) to prevent **Context Bleeding** (Challenge #7). While this solved bleeding between *questions*, it also prevented the judge from accumulating evidence *within* the same question — a case of over-correction.

**How We Overcame It:**
We changed when the judge's memory is cleared:
- **Before:** `working_memory.clear()` was called at the start of **every `check_chunk()` call**.
- **After:** `working_memory.clear()` is only called **once, between questions**, at the top of the eval loop.

This means the judge now **accumulates context within a single question**. By the time Chunk 2 is evaluated, the judge's conversation history already contains Chunk 1's content. When it sees the missing two weaknesses in Chunk 2, it recognizes that the combined evidence now satisfies the question and fires `answer_found=True`.

Updated judge prompts were also adjusted to explicitly tell the model it *should* consider all previously seen chunks:

> *"Consider this chunk AND all chunks you have already evaluated for this question. Together, is there now enough information to correctly answer the question?"*

**Bonus: This is also cheaper on paid APIs.**
When the judge's memory accumulates and is sent as a growing conversation history, providers that support **prompt caching** only charge full price for the **new delta tokens** per call — the cached prefix is heavily discounted. Here are the real numbers (as of March 2026):

| Provider | Model | Standard Input | Cached Input | Discount | Cache Setup |
|---|---|---|---|---|---|
| **OpenAI** | GPT-4o | $2.50 / 1M | $1.25 / 1M | **50% off** | Automatic (≥1024 token prefix) |
| **Anthropic** | Claude Sonnet | $3.00 / 1M | ~$0.30 / 1M | **~90% off** | Explicit `cache_control` markers required |
| **Anthropic** | Claude Haiku | $1.00 / 1M | ~$0.10 / 1M | **~90% off** | Explicit `cache_control` markers required |
| **Gemini** | Gemini 2.5 Pro | $1.25 / 1M | ~$0.31 / 1M | **~75% off** | Automatic implicit + explicit Context Cache API |
| **Gemini** | Gemini 2.5 Flash | $0.30 / 1M | ~$0.075 / 1M | **~75% off** | Automatic implicit + explicit Context Cache API |
| **Grok** | Grok 3 | $3.00 / 1M | ❌ No caching | — | Not supported as of early 2026 |

**Practical impact in our eval loop:**
If a question evaluates 10 chunks and each context block is ~1,000 tokens, with accumulation the 10th call sends ~10,000 cached tokens + ~1,000 new tokens. On Claude Sonnet that's `(10,000 × $0.00030) + (1,000 × $0.003) = $0.006` vs `10 × 1,000 × $0.003 = $0.030` without caching — **5× cheaper**. The savings compound as K increases.

Providers without caching (Grok) still charge for the full growing context each call, so this optimization is most valuable on OpenAI/Anthropic/Gemini.

---

## 10. Dynamic K-Value Tuning (BROAD vs. SPECIFIC Retrieval Depth)

**The Problem:**
The initial `run_evals_routed.py` implementation set `K=5` for `SPECIFIC` queries (direct rule/stat lookups) and `K=15` for `BROAD` queries (book-wide synthesis). The `SPECIFIC` K=5 was too conservative — it caused failures for questions where the answer landed beyond rank 5 due to **semantic crowding** (e.g., `"What does the 'Invisibility' spell do?"` was ranked below `Greater Invisibility`, `Potion of Invisibility`, and `See Invisibility`, pushing the actual spell entry to rank 6+).

**The Fix and Final K-Values:**
| Scope | Old K | New K | Rationale |
|---|---|---|---|
| `BROAD` | 15 | **80** | Book-wide synthesis questions (e.g., "What does the DM do?") need deep sweeps. Running locally on phi4, cost is irrelevant. |
| `SPECIFIC` | 5 | **15** | Matches the vanilla `run_evals.py` baseline (K=10) plus a small buffer for high-collision name spaces (Invisibility, Polymorph variants, etc.). |
| Router error fallback | 5 | **15** | Kept consistent with SPECIFIC default. |

**Why the early-exit makes high K safe:**
The judge loop always `break`s the moment a hit is found. A question that resolves at Rank 2 still only pays for 2 judge calls, regardless of K=80. The high K is only costly for true failures — which is exactly when you *want* maximum coverage.

---

## 11. The "Ugly Duckling" Table Row vs. Semantic Density

**The Problem:**
We are currently facing an issue where the Master Router pipeline (`run_evals_routed.py`) struggles to retrieve short, specific table rows compared to the vanilla pipeline (`run_evals.py`). A prime example is the query: `Q1: How much does a Longsword cost and what damage does it deal?`

In the vanilla pipeline, the true answer (the table row: `Longsword 1d8 Slashing Versatile (1d10) Sap 3 lb. 15 GP`) natively ranks at Chunk 2. However, in the routed pipeline, it falls out of the top 10 entirely, replaced by verbose monster stat blocks (like Guard Captain or Half-Dragon) that happen to mention a Longsword.

**Why It's Happening (The Semantic Trap):**
Dense vector embedding models (like BAAI/bge-large-en-v1.5) are trained on massive datasets of natural human language. When we pass a highly specific, grammatically sound "HyDE" excerpt or even a formal normalized query through the encoder, it creates a vector that expects *other* grammatically sound sentences in response.

A raw table row (`Longsword 1d8 Slashing Versatile...`) has terrible grammar. From a semantic density perspective, it's an "ugly duckling." A chatty monster stat block that says "The Guard Captain wields a Longsword" feels much more linguistically natural to the vector engine, giving it a higher similarity score and pushing the actual table row down the rankings. 

**What We've Tried So Far (And Why It Hasn't Fully Fixed It):**
- **Exact-Match Track (BM25 Simulation):** We added a parallel track to grab up to 50 chunks via Chroma's `where_document={"$contains": kw}` filter for specific names ("Longsword").
- **Cross-Encoder Re-Ranking:** We pass those 50 raw chunks to a `ms-marco` cross-encoder to strictly grade their relevance to the question.

Despite these advanced RAG techniques, the "ugly" table row still isn't consistently making it to the judge. This shows that the initial vector space distance for tables is extremely fragile. When we dilute the original user query `query` with extra context like `hyde_excerpt`, we accidentally increase the distance to the "ugly" table row, pushing it down. We need a way to reliably fetch table data without the semantic vector engine penalizing its lack of prose.

**Root Cause Confirmed (via exact_quote debugging):**
By adding an `exact_quote` field to the `JudgeVerdict` Pydantic model, we forced the judge to prove its FOUND verdict with a verbatim quote. The quote it produced was from the **Sun Blade** entry — *"this magic weapon functions as a Longsword with the Finesse property"* — which contains zero cost or damage information. This confirmed two simultaneous bugs:
1. **Retrieval Failure:** The actual weapons table row (`Longsword 1d8 Slashing 15 GP`) ranked below 10 in the routed pipeline, despite being the true answer.
2. **Judge False Positive:** The judge was treating a secondary mention of "Longsword" inside a magic item description as a valid factual answer.

**Remediation Options:**

### Option 1: Hard Override for Exact Keyword Matches (Quickest, ~5 min)

Bypass the Cross-Encoder scoring entirely for exact keyword match chunks. Force them to the top unconditionally:

```python
# In run_evals_routed.py — after cross_encoder.predict():
for idx in exact_match_indices:
    scores[idx] = 99.0  # Force to top, skip CE judgement for tables
```

**Pro:** Zero risk, no re-indexing, instant fix.  
**Con:** Only works if the table row was captured by the exact-match scan.

---

### Option 2: Replace the Cross-Encoder with a Stronger Model

`ms-marco-MiniLM-L-6-v2` was trained on web passages (no tables). It penalises dry tabular text because it looks like broken prose. Recommended replacements:

| Model | Notes |
|---|---|
| `cross-encoder/ms-marco-MiniLM-L-12-v2` | Larger version of the same, marginally better |
| `cross-encoder/nli-deberta-v3-small` | Better for fact/entailment tasks |
| **`BAAI/bge-reranker-large`** | Best overall; diverse training data including structured text — **Recommended** |

```python
cross_encoder = CrossEncoder("BAAI/bge-reranker-large", device="cpu")
```

**Pro:** Improves ranking quality broadly — not just for tables.  
**Con:** Larger model = slower re-ranking. Downloads on first run.

---

### Option 3: Convert Table Rows to Prose at Indexing Time

Fix the bias at the source. During chunking/indexing, detect table rows and convert to natural language before embedding:

**Before:** `Longsword 1d8 Slashing Versatile (1d10) Sap 3 lb. 15 GP`  
**After:** `The Longsword costs 15 gp, deals 1d8 slashing damage, and has the Versatile property (1d10 two-handed). It weighs 3 lbs.`

**Pro:** Permanently fixes semantic bias for all table queries.  
**Con:** Requires full re-index of ChromaDB. LLM call per table row at build time.

---

**Recommended Path:** Start with **Option 1** (instant, zero risk) to confirm the table now reaches the judge, then switch to **Option 2** (`BAAI/bge-reranker-large`) as a longer-term quality improvement.
