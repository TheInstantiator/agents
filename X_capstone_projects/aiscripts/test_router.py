import sys
import os
import asyncio
from pydantic import BaseModel, Field

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Bring in the core_agent
sys.path.append(os.path.join(SCRIPT_DIR, ".."))
from core_agent import CoreAgent

class RouterDecision(BaseModel):
    scope: str = Field(description="'BROAD' or 'SPECIFIC'")
    category: str = Field(description="The D&D category, e.g., 'spells', 'monsters', 'rules', 'classes', 'items', 'lore', or 'other'")
    hyde_excerpt: str = Field(description="A fake 3-sentence excerpt from the official D&D rulebook that sounds like it would answer the player's question.")

async def test_router():
    print("Initializing LLM Router (phi-agent)...")
    phi_kwargs = CoreAgent.load_litellm_kwargs_from_config("phi-agent")
    
    # We create a stateless one-shot agent that acts as our Master Router
    router_agent = CoreAgent(
        agent_id="master_router",
        system_prompt=(
            "You are the Master Routing AI for a D&D 5e completely offline rules engine. "
            "A player will ask a question. Your job is to strictly analyze it and return a JSON object.\n"
            "1. scope: If the question requires pulling info from all over the book (e.g. 'What does the DM do?'), output 'BROAD'. "
            "If it's a direct rule, stat, or item lookup (e.g. 'What is a goblin\\'s AC?'), output 'SPECIFIC'.\n"
            "2. category: Assign it to 'spells', 'monsters', 'rules', 'classes', 'items', 'lore', or 'other'.\n"
            "3. hyde_excerpt: Write a 3-sentence totally fake excerpt that sounds exactly like official D&D 5e rulebook language, "
            "designed to answer the question. This will be embedded to perform semantic search, so use official D&D vocabulary and tone."
        ),
        litellm_kwargs=phi_kwargs,
        response_model=RouterDecision,
        one_shot=True
    )

    test_questions = [
        "Bro how do I not die on the floor bleeding out?",
        "What does the DM do in a campaign?",
        "What's the exact AC and HP of an ancient red dragon?",
        "Do paladins get extra attack and when?",
    ]

    for q in test_questions:
        print("\n" + "="*50)
        print(f"🧐 PLAYER ASKS: {q}")
        print("-" * 50)
        
        try:
            decision = await router_agent.ask(q)
            print(f"🟢 SCOPE        : {decision.scope}")
            print(f"🟢 CATEGORY     : {decision.category}")
            print(f"🟢 HYDE EXCERPT : {decision.hyde_excerpt}")
            
            # Simulated Calculation based on your logic:
            if decision.scope.upper() == "BROAD":
                # E.g. 100k Token limit / ~3.0k Tokens per Window = 33 possible chunks max
                recommended_k = 15 
                max_summary_words = 1000
                print(f"\n🧠 PIPELINE LOGIC: Broad query detected. Setting K={recommended_k}. Requesting LLM to summarize findings to ~{max_summary_words} words.")
            else:
                recommended_k = 3
                print(f"\n🧠 PIPELINE LOGIC: Specific query detected. Setting K={recommended_k} for laser-focused lookup.")
                
        except Exception as e:
            print(f"Error querying router: {e}")

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv(override=True)
    asyncio.run(test_router())
