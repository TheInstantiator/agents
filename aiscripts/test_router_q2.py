import sys
import os
import asyncio
from pydantic import BaseModel, Field

# Correct import
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(SCRIPT_DIR, "..", ".."))

from core_agent import CoreAgent

class RouterDecision(BaseModel):
    scope: str = Field(description="'BROAD' or 'SPECIFIC'")
    category: str = Field(description="The D&D category, e.g., 'spells', 'monsters', 'rules', 'classes', 'items', 'lore', or 'other'")
    hyde_excerpt: str = Field(description="A fake 3-sentence excerpt from the official D&D rulebook that sounds like it would answer the player's question.")

async def test_router():
    os.chdir("/home/ouar/projects/agents/X_capstone_projects")
    phi_kwargs = CoreAgent.load_litellm_kwargs_from_config("phi-agent")
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
    
    q1 = "How many feet can a Huge Beast move in one turn according to general rules?"
    
    print(f"\n🧐 PLAYER ASKS: {q1}")
    decision = await router_agent.ask(q1)
    print(f"SCOPE: {decision.scope}")
    print(f"CATEGORY: {decision.category}")
    print(f"HYDE EXCERPT: {decision.hyde_excerpt}")

if __name__ == "__main__":
    from dotenv import load_dotenv
    os.chdir("/home/ouar/projects/agents/X_capstone_projects")
    load_dotenv(override=True)
    asyncio.run(test_router())
