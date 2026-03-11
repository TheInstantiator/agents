# Core Agent Class Design

This document outlines the architecture for a modern, robust AI Agent class. It serves as the blueprint for building autonomous agents in the `me_manager` project. The design avoids bloated frameworks (like LangChain) in favor of simple, standardized Python and industry best practices.

## 1. Core Principles
*   **Encapsulation**: An Agent is a Class. It holds its own state (identity, memory, tools).
*   **Scalability (One-Shot vs Stateful)**: The class must support both complex memory-driven tasks (MeManager) and stateless, one-and-done tasks (D&D Recruiter/Judge) via a simple `one_shot` toggle.
*   **Asynchronous I/O**: Network calls (LLM generation) use `asyncio` to prevent blocking the main thread.
*   **Deterministic Output & Guardrails**: The Agent uses swappable Pydantic models to force the LLM to reply in strict JSON formats. Outbound responses can be validated through custom "guardrails" before proceeding.
*   **Hierarchical Memory**: The Agent manages its context window while persisting long-term knowledge via RAG.

---

## 2. Anatomy of the Agent

A complete Agent class consists of five primary modules:

### A. The Persona (Identity & Configuration)
*   `agent_id`: A unique identifier tying the agent to a specific User Goal (e.g., `goal_python_mastery_01`). This ID is used to prefix and reload its specific DB tables and memory files.
*   `system_prompt`: The core instructions and behavioral boundaries defining the agent's problem space.
*   `litellm_kwargs`: The parsed LLM configuration (e.g., from `config.json`). This includes the `model` string, `api_key`, and `api_base`. *Crucially, this is completely swappable.* The problem the agent solves is fixed, but you can hot-swap the "engine" by passing a different config dictionary.

### B. L1 Memory: Working Memory (Short-Term Context)
*   **The Message History**: A temporary list of dictionaries `[{"role": "user", "content": "..."}]` encompassing the current conversation.
*   **Context Manager**: A method that monitors the token length. When it approaches the limit (e.g., 8,000 tokens), it summarizes the history and drops the oldest messages.

### C. L2 Memory: Semantic Knowledge (Long-Term RAG)
*   **Vector Database**: A connection to a local database. The `agent_id` is used to create or connect to a unique collection/namespace within the DB so agents don't hallucinate across different goals.
*   **Recall Mechanism**: Before answering, the Agent converts the user's prompt into an embedding and searches the database for relevant past facts, preferences, or technical documents.
*   **Reflection Mechanism**: A background process that runs periodically to extract key facts from the *Working Memory* and save them natively to the *Semantic Knowledge* base.

### D. Serialization & Persistence (The "Save State")
*   **Save/Load Methods**: The Agent must be capable of exporting its `working_memory` to disk (JSON) based on its `agent_id`. It must be able to spin up, ingest that JSON, connect to its isolated Vector DB collection, and resume work exactly where it left off yesterday, completely independent of the lifespan of the actual Python script.

### D. The Action Space (Tools)
*   **Tool Registry**: A dictionary mapping tool names to actual Python functions (`{"search_web": my_search_function}`).
*   **Function Execution**: The logic that parses an LLM's request to run a tool, executes the Python code safely, and returns the observation back to the LLM.

---

## 3. The Execution Loop (ReAct / Think-Do)

The public interface of the Agent should be extremely simple: `await agent.ask("Hello")`. Behind the scenes, the loop behaves like this:

1.  **Recall**: Search L2 (RAG) for concepts related to "Hello".
2.  **Inject**: Prepend the retrieved RAG facts into a hidden System Message.
3.  **Generate**: Send the System Prompts + L1 Working Memory to LiteLLM.
4.  **Decide**: The LLM responds. Does it want to use a tool, or answer the user?
    *   *If Tool*: Execute the Python function, add the result to L1 Memory, loop back to Step 3.
    *   *If Answer*: Return the final answer to the user.
5.  **Reflect (Optional)**: If the conversation has reached a milestone, trigger the summarization and vector storage routine.

6. **Tokens/Cost Output**: The agent should output the number of tokens used and the cost of the generation.  It can have accumulators for tokens and cost.  It can then be asked for their values.

---

## 4. Skeleton Code Example

```python
import asyncio
import litellm
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Callable, Type, Optional

class AgentResponse(BaseModel):
    thoughts: str = Field(description="Internal reasoning before answering.")
    final_answer: str = Field(description="The response returned to the user.")

class CoreAgent:
    def __init__(
        self, 
        agent_id: str, 
        system_prompt: str, 
        litellm_kwargs: dict, 
        one_shot: bool = False,
        response_model: Optional[Type[BaseModel]] = AgentResponse
    ):
        # The ID ties this instance to saved files and isolated databases
        self.agent_id = agent_id
        
        # The Problem Definition remains constant
        self.system_prompt = system_prompt
        
        # The Engine is fully swappable (e.g., {'model': 'xai/grok-4', 'api_key': '...'})
        self.litellm_kwargs = litellm_kwargs
        
        # Mode toggle: If True, bypasses all memory persistence routines
        self.one_shot = one_shot
        
        # The Expected Output Schema (Swappable!)
        self.response_model = response_model
        
        # L1 Memory (Short-Term)
        self.working_memory: List[Dict[str, str]] = []
        
        # L2 Memory (Long-Term RAG - isolated by agent_id)
        self.vector_db = None 
        
        # Tools
        self.tools: Dict[str, Callable] = {}

    def swap_model(self, new_litellm_kwargs: dict):
        """Hot-swap the LLM engine config without losing state."""
        self.litellm_kwargs = new_litellm_kwargs

    def save_state(self, filepath: str):
        """Export working memory to JSON using agent_id."""
        pass # e.g., json.dump(self.working_memory, open(f"{self.agent_id}_state.json", 'w'))

    def load_state(self, filepath: str):
        """Resume agent from a saved JSON memory file."""
        pass # e.g., self.working_memory = json.load(open(f"{self.agent_id}_state.json", 'r'))

    def register_tool(self, name: str, func: Callable):
        self.tools[name] = func

    async def _recall_long_term_memory(self, query: str) -> str:
        """Search ChromaDB for relevant past context."""
        if not self.vector_db:
            return ""
        # Implementation: Vector search query here
        return "Retrieved memories..."
        
    async def _reflect_and_store(self):
        """Extract facts from working_memory and save to ChromaDB."""
        # Implementation: Ask a fast LLM to summarize self.working_memory
        pass

    async def ask(self, user_input: str) -> str:
        # 1. Recall (Skip if one_shot)
        past_context = ""
        if not self.one_shot:
            past_context = await self._recall_long_term_memory(user_input)
        
        # 2. Build the exact prompt for this turn
        context_prompt = self.system_prompt
        if past_context:
            context_prompt += f"\n\n[RECALLED MEMORIES]: {past_context}"
        
        self.working_memory.append({"role": "user", "content": user_input})
        
        # Construct the full payload
        payload = [{"role": "system", "content": context_prompt}] + self.working_memory

        # 3. Generate
        # Start with the base kwargs from config.json
        generate_kwargs = self.litellm_kwargs.copy()
        
        # Inject the conversation payload and runtime settings
        generate_kwargs["messages"] = payload
        generate_kwargs["temperature"] = generate_kwargs.get("temperature", 0.3)
        
        if self.response_model:
            generate_kwargs["response_format"] = self.response_model
            
        response = await litellm.acompletion(**generate_kwargs)
        raw_output = response.choices[0].message.content
        
        # 4. Guardrails & Formatting
        # This is where you would hook in NeMo Guardrails or custom validation checks (e.g., toxicity, bounds checking).
        if self.response_model:
            parsed_output = self.response_model.model_validate_json(raw_output)
            # You can decide how to store the parsed object in memory. Usually, stringifying it is best.
            content_to_save = raw_output 
            result = parsed_output
        else:
            content_to_save = raw_output
            result = raw_output
        
        # Save Agent's reply to memory
        self.working_memory.append({"role": "assistant", "content": content_to_save})
        
        # 5. Reflect (Skip if one_shot)
        if not self.one_shot and len(self.working_memory) > 10:
            await self._reflect_and_store()

        return result
```
