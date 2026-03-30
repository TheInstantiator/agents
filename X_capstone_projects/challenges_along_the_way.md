# Challenges Along the Way

This document tracks significant technical hurdles, architectural decisions, and "gotchas" discovered while building the AI Agent infrastructure.

---

## 1. The "Thinking" Suppression Issue (Structured Output vs. Reasoning Models)

**The Problem:**
When using reasoning models (like `deepseek-r1`) or trying to get standard models to explain their rationale, the agents were skipping the "thinking" phase entirely and just spitting out the final answer. 

**Why It Happened:**
The issue was caused by strict **Structured Output enforcement**. By using Pydantic models to force strict JSON, the LLMs realized that including XML tags like `<think>` would invalidate the JSON parser. To avoid errors, they suppressed their natural reasoning.

**How We Overcame It:**
We implemented **Schema-Driven Chain of Thought**. We added a `thoughts: str` field as the *very first variable* in our Pydantic models. Because `thoughts` is the first key, the LLM is physically forced to generate a paragraph of reasoning *before* it generates the final answer fields. This acts as an artificial Chain of Thought, significantly increasing output quality.

---

## 2. The Streaming Structured Output Problem

**The Problem:**
We couldn't easily stream thoughts character-by-character while maintaining strict JSON validity. Standard parsers crash if you try to read an unclosed JSON string.

**The Solution:**
For the MVP, we favor stability—waiting for the full response and parsing it cleanly. For future live-streaming, we would use a "Partial JSON Parser" or a library like `jiter` that can yield field values incrementally before the final bracket closes.

---

## 3. Small LLM Hallucinations and Dictionary Lookups

**The Problem:**
Small 8B models (like `llama3.1`) often misinterpreted "Flex Path" categories as actual class names, leading to crashes when the script tried to look up non-existent keys in `constants.py`.

**How We Overcame It:**
We implemented a **Graceful Fallback Mechanism**. We now use `CLASS_ATTRIBUTES.get(choice.actual_class)` instead of direct access. If it returns `None`, we print a `HALLUCINATION DETECTED` warning, skip the turn, and let the next agent in the roster try.

---

## 4. The "Ghost Table" & Prose Conversion Breakthrough

**The Problem:**
The D&D ruleset contains hundreds of "Ghost Tables"—unstructured grids of data without markdown pipe characters (`|`). For example:
`Longsword 1d8 Slashing Versatile (1d10) Sap 3 lb. 15 GP`

**The "Ugly Duckling" Penalty:**
Because these tables look like "ungrammatical gibberish" to semantic embedding models and Cross-Encoders, they were assigned massive negative penalties (e.g., `-6.23` from `ms-marco`). This caused the most important weapon data to plummet to Rank 25+ during retrieval, effectively making it invisible to the RAG system.

**The Solution: Agentic Prose Conversion:**
We overhauled `build_rag_db.py` to handle these proactively:
1. **Regex Detection:** We built a custom detector for rows containing dice notation (`1d8`) and currency (`GP`, `SP`).
2. **LLM Fallback:** Any detected "Ghost Table" is handed to a local `phi4` agent with the prompt: *"Convert these table rows into perfectly descriptive natural language sentences."*
3. **Prose Indexing:** Instead of indexing the raw grid, we index the converted sentences:  
   *Before:* `Longsword 1d8 Slashing... 15 GP`  
   *After:* `The Longsword costs 15 gp, deals 1d8 slashing damage, and has the Versatile property.`
4. **Result:** Retrieval accuracy for weapon costs jumped from 0% to near 100%, and MRR hit a perfect 1.0 for these specific factual queries.

---

## 5. The Agent Memory Leak (Context Memory Exhaustion)

**The Problem:**
During indexing, the `phi-agent` started fast (~2s per chunk) but slowed down exponentially—reaching 20+ seconds per chunk by the time it hit page 25.

**Why It Happened:**
The `CoreAgent` class was missing a crucial check for the `one_shot` flag. It was accidentally appending every single prompt and LLM response into a permanent `working_memory` list. By chunk 25, the agent was reading a massive "super-prompt" containing the previous 24 pages of the book, exhausting its 16k context window and forcing the hardware to offload to slow CPU RAM.

**How We Overcame It:**
We modified `core_agent.py` to strictly honor the `one_shot` flag. If set to `True`, the script now calls `self.working_memory.clear()` immediately after a generation is returned. This keeps the context window clean, prevents memory ballooning, and restored stable performance globally.

---

## 6. VRAM Optimization and Ollama Concurrency

**The Problem:**
Running large context windows (16k–128k) while allowing `OLLAMA_NUM_PARALLEL=2` was causing the 16GB VRAM to partition. This forced the model to "split" between the GPU and System RAM, slowing generation to a crawl whenever two chunks were processed at once.

**How We Overcame It:**
1. **Serialization:** We set `OLLAMA_NUM_PARALLEL=1` in the `ollama.service` environment and matched it with `asyncio.Semaphore(1)` in the Python indexing script.
2. **Context Balancing:** We reduced the default `num_ctx` and `max_input_tokens` in `config.json` from `128000` to `16384`. This provides more than enough room for a single Parent Chunk + Summary while ensuring the Entire Model weights stay locked in VRAM at peak speed.

---

## 7. The Split Answer Problem (Evidence Accumulation)

**The Problem:**
Some questions (e.g., "What are the four Vampire Weaknesses?") require pulling information spread across multiple chunks. If the Judge evaluates chunks in complete isolation, it will reject every chunk for being "Incomplete."

**How We Overcame It:**
We changed the evaluation logic to **Accumulate Evidence within a Question**.
1. **Memory Persistence:** We stop clearing the Judge's memory between chunks within the same query.
2. **Additive Prompts:** We tell the Judge: *"Consider this chunk AND all chunks you have already evaluated for this question."*
3. **Cost Savings:** On paid APIs (Gemini/Anthropic), this approach leverages **Prompt Caching**, making the 10th chunk evaluation up to 90% cheaper than the first one.

---

---

## 9. The Class Feature "Ghost Table" (Regex Anchoring Issue)

**The Problem:**
While weapon tables were being caught, **Class Feature tables** (Monk, Barbarian, Paladin) were being completely ignored. For example, a Monk row looks like:  
`15 +5 Perfect Focus 1d10 15 +25 ft.`

**Why It Happened:**
The original "Ghost Table" regex was anchored to the end of the line (`$`) and expected a digit as the final character. Because class features have descriptive text (`Perfect Focus`) and units (`ft.`) at the end of the row, the detector failed silently. This caused these tables to be split into unsearchable 300-char fragments.

**How We Overcame It:**
1. **Flexible Regex:** Updated `is_ghost_row` to `^\d+\s+[+−-]\d+(\s+|$)`, focusing only on the `Level + Proficiency` signature at the start of the line.
2. **Header Lookback:** Updated the extractor to look back 3 lines when a row is found to "grab" the column headers (e.g., *Focus Points*, *Martial Arts Die*), ensuring the LLM has context for the prose conversion.

---

## 10. The Markdown JSON Parsing Failure

**The Problem:**
Small but capable models (like `phi-4`) often wrap their structured output in markdown code blocks (e.g., ` ```json ... ``` `) even when explicitly told not to. This caused Pydantic's `model_validate_json` to crash.

**How We Overcame It:**
We implemented a **Robust JSON Extractor** in `core_agent.py`. Before parsing, the agent now:
1. Strips out markdown code blocks using regex.
2. If that fails, it finds the first `{` and last `}` and crops everything else.
This allows the pipeline to remain stable even if the LLM adds polite preamble or markdown formatting.

---

## 11. The "Interesting Fact" Hallucination Trigger

**The Problem:**
The Critic agent was occasionally encouraging the pipeline to add "interesting facts" from its own internal knowledge if the rulebook was sparse. This caused "Relentless Rage" to hallucinate a "Disadvantage on Perception" mechanic that doesn't exist in the ruleset.

**How We Overcame It:**
We moved to **Zero-Tolerance Grounding**. We removed all "interesting facts" suggestions and updated the System Prompts for both the Critic and Answer Generator to explicitly reject any information not found in the provided `Context`. We now enforce an "I don't know" policy over "educated guesses."

---

## 12. Current Accuracy & Architecture Summary

#### **The Implementation (Multi-Vector Hierarchical RAG)**
*   **Parent Chunks:** 4,000 chars each (Sectional context).
*   **Child Chunks:** 300 chars each (High-precision retrieval).
*   **Agentic Prose conversion:** LLM transformation of dense tables into searchable, descriptive sentences.
*   **Robust Parsing:** Extraction logic to handle markdown-wrapped JSON.

#### **Performance Metrics (As of March 24, 2026)**
*   **Weapon Costs:** success (Rank 1)
*   **Monk Focus Points:** success (via Class Feature Table detection)
*   **Barbarian Relentless Rage:** Fixed (Hallucinations removed via strict grounding)
*   **JSON Stability:** 100% (phi-4 now parses correctly regardless of backticks)

The current pipeline represents a significant jump in robustness, handling not just standard tables but also the complex, text-heavy grids common in RPG rulesets.
