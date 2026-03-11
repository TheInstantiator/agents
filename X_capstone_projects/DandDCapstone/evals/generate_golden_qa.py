import os
import sys
import json
import asyncio
from tqdm import tqdm
from pydantic import BaseModel, Field
from typing import List

# Setup pathing for CoreAgent (lives in project root)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(SCRIPT_DIR, "..", ".."))
from core_agent import CoreAgent

MD_PATH = os.path.join(SCRIPT_DIR, "..", "..", "documents", "DandDRuleset.md")
OUTPUT_PATH = os.path.join(SCRIPT_DIR, "golden_dataset.json")

class QAPair(BaseModel):
    question: str = Field(description="A specific, factual question about D&D rules.")
    expected_answer: str = Field(description="A concise, correct answer based on the text.")
    keywords: List[str] = Field(description="3-5 must-have keywords for retrieval evaluation.")
    category: str = Field(description="The type of rule (e.g., Combat, Spell, Class, Table).")

class QABatch(BaseModel):
    items: List[QAPair]

async def generate_questions():
    if not os.path.exists(MD_PATH):
        print(f"❌ Error: Markdown file not found at {MD_PATH}. Run pdf_to_md.py first.")
        return
    print(f"1. Loading Rulebook Markdown ({MD_PATH})...")
    with open(MD_PATH, "r", encoding="utf-8") as f:
        content = f.read()
    print(f"   --> Loaded {len(content)} characters ({len(content)//4} est. tokens)")
    
    # Split into blocks of ~20,000 characters (approx 10-15 pages per block)
    block_size = 20000
    blocks = [content[i:i + block_size] for i in range(0, len(content), block_size)]
    
    print(f"2. Divided rules into {len(blocks)} thematic blocks.")
    
    # Initialize a high-tier Agent for question generation
    # We use Gemini 3.1 Pro for high-fidelity rule understanding
    agent_kwargs = CoreAgent.load_litellm_kwargs_from_config("gemini-agent-tier1-think")
    generator = CoreAgent(
        agent_id="golden_gen",
        system_prompt=(
            "You are a professional D&D 5.2 Rules Architect. Your goal is to create a 'Golden Dataset' "
            "for testing a RAG system. For the provided text, generate 8-10 diverse questions. "
            "Include direct facts, complex rule interactions, and specific table-based data questions. "
            "Format your response as a JSON list of QAPair objects."
        ),
        litellm_kwargs=agent_kwargs,
        response_model=QABatch
    )
    
    # --- RESUMPTION LOGIC ---
    golden_dataset = []
    if os.path.exists(OUTPUT_PATH):
        try:
            with open(OUTPUT_PATH, "r") as f:
                golden_dataset = json.load(f)
            print(f"   --> Found existing dataset with {len(golden_dataset)} questions. Resuming...")
        except Exception:
            print("   --> Could not parse existing dataset. Starting fresh.")
    
    # Heuristic: Each block produces ~8 questions. Skip blocks we've already 'mined'
    start_block_idx = (len(golden_dataset) // 8) * 3 # Rough estimate
    # Actually, let's just use a more precise tracking if we wanted, 
    # but for now, we'll just skip the first N blocks based on count.
    blocks_to_process = blocks[start_block_idx:]
    print(f"3. Processing {len(blocks_to_process)} remaining blocks (Parallel Workers: 2)...")
    
    batch_concurrency = 2
    
    try:
        for i in range(0, len(blocks_to_process), batch_concurrency):
            current_batch = blocks_to_process[i:i + batch_concurrency]
            
            pre_tokens = generator.tokens_used
            pre_cost = generator.cost
            
            # Use gather but catch errors per-call if needed (or just per batch)
            tasks = [generator.ask(f"Text Block:\n\n{block}") for block in current_batch]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for result in results:
                if isinstance(result, QABatch):
                    golden_dataset.extend([item.model_dump() for item in result.items])
                elif isinstance(result, Exception):
                    print(f"   --> ⚠️ Batch Task Error: {result}")
            
            batch_tokens = generator.tokens_used - pre_tokens
            batch_cost = generator.cost - pre_cost
            
            print(f"   --> Batch Tokens: {batch_tokens} | Batch Cost: ${batch_cost:.4f}")
            print(f"   --> [Cumulative] Questions: {len(golden_dataset)} | Tokens: {generator.tokens_used} | Cost: ${generator.cost:.4f}")

            # Persist Progress
            with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
                json.dump(golden_dataset, f, indent=2)
                f.flush()
                os.fsync(f.fileno())
            print(f"   --> [Checkpoint] Saved to {os.path.abspath(OUTPUT_PATH)}")
            
            # --- NEW: Cap at 100 questions ---
            if len(golden_dataset) >= 100:
                print(f"\n🎯 Target reached (100+ questions). Stopping generation as requested.")
                break

            # Rate limit cooldown (especially for 'thinking' models)
            if i + batch_concurrency < len(blocks_to_process):
                print("   --> Sleeping 10s to respect rate limits...")
                await asyncio.sleep(10)

    except (KeyboardInterrupt, asyncio.CancelledError, Exception) as e:
        print(f"\n\n🛑 Process Interrupted! {type(e).__name__}: {e}")
        print(f"--> Finalizing save of {len(golden_dataset)} questions...")
        with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
            json.dump(golden_dataset, f, indent=2)
        print(f"✅ State saved to {os.path.abspath(OUTPUT_PATH)}")
        if not isinstance(e, (KeyboardInterrupt, asyncio.CancelledError)):
            raise e

    print(f"💰 Total Summary --> Questions: {len(golden_dataset)} | Tokens: {generator.tokens_used} | Total Cost: ${generator.cost:.4f}")

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv(override=True)
    asyncio.run(generate_questions())
