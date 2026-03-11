import os
import sys
import json
import asyncio
from typing import List
from pydantic import BaseModel, Field

# Setup pathing for CoreAgent
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(SCRIPT_DIR, "..", ".."))
from core_agent import CoreAgent

MD_PATH = os.path.join(SCRIPT_DIR, "..", "..", "documents", "DandDRuleset.md")
OUTPUT_PATH = os.path.join(SCRIPT_DIR, "golden_dataset.json")

class QAPair(BaseModel):
    question: str
    expected_answer: str
    keywords: List[str]
    category: str

class QABatch(BaseModel):
    items: List[QAPair]

async def generate_long_context():
    if not os.path.exists(MD_PATH):
        print(f"❌ Error: Markdown file missing at {MD_PATH}. Run pdf_to_md.py first.")
        return

    print(f"1. Loading Full Rulebook Context ({MD_PATH})...")
    with open(MD_PATH, "r", encoding="utf-8") as f:
        full_text = f.read()
    
    # Initialize Agent with a high-tier model
    agent_kwargs = CoreAgent.load_litellm_kwargs_from_config("phi-agent")
    generator = CoreAgent(
        agent_id="golden_gen_long",
        system_prompt=(
            "You are a D&D Rules Architect and Search System Evaluator. You have the entire rulebook in your context. "
            "Generate 50 high-quality, professional questions that test a RAG system. "
            "IMPORTANT: Exactly 5 of the 50 questions MUST be 'Hard for RAG' (e.g., requires combining information from multiple distant sections, relies on subtle rule exceptions, or asks for multi-step logical deductions). "
            "Include complex rules, table data, and monster stats. Respond ONLY in JSON matching the schema."
        ),
        litellm_kwargs=agent_kwargs,
        response_model=QABatch
    )

    dataset = []
    # If we already have manual questions, keep them
    if os.path.exists(OUTPUT_PATH):
        with open(OUTPUT_PATH, "r") as f:
            dataset = json.load(f)

    # We do 3 rounds of 50 to get to 150+ questions without hitting output truncation
    for round_idx in range(1, 4):
        print(f"\n🚀 Round {round_idx}/3: Requesting 50 unique questions...")
        
        # We tell the model which questions we already have so it doesn't repeat itself
        existing_q_titles = [q['question'] for q in dataset[-20:]] # Last 20 for brevity
        prompt = (
            f"Based on the full rulebook context provided, generate 50 unique questions to test our RAG system. "
            f"Ensure EXACTLY 5 of these questions are 'Hard for RAG' (e.g. cross-referencing multiple tables or sections, complex negations, or highly specific nested rules). "
            f"Avoid these recent topics: {existing_q_titles}. "
            f"Focus on a mix of different chapters in each batch."
        )
        
        try:
            # We "jam" the full_text into the prompt
            response = await generator.ask(f"CONTEXT:\n{full_text}\n\n{prompt}")
            
            if isinstance(response, QABatch):
                new_items = [i.model_dump() for i in response.items]
                dataset.extend(new_items)
                print(f"✅ Success! Added {len(new_items)} questions.")
            
            # Save progress
            with open(OUTPUT_PATH, "w") as f:
                json.dump(dataset, f, indent=2)
            
            print(f"📊 Total Questions: {len(dataset)} | Total Cost: ${generator.cost:.4f}")

            if round_idx < 3:
                print("⏳ Sleeping 65s to reset 1M TPM Quota...")
                await asyncio.sleep(65)

        except Exception as e:
            print(f"❌ Round {round_idx} failed: {e}")
            break

    print(f"\n🎉 Finished! Final dataset at: {os.path.abspath(OUTPUT_PATH)}")

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv(override=True)
    asyncio.run(generate_long_context())
