# recruiter.py
# This is the "Brain" of your application. It uses AI agents to decide which characters 
# should join the party and a "Judge Agent" to make sure the party is balanced.

import asyncio
import json
import os
from agents import Agent, Runner, OpenAIChatCompletionsModel, RunConfig
from openai import AsyncOpenAI

# We import our own custom modules here
from factory import CharacterIdentity, create_pc_from_choice, EvaluationResult, save_party, load_party
from constants import CLASS_ATTRIBUTES

# --- API SETUP ---
# This points Python to your LiteLLM proxy running on port 4000.
# The 'api_key' is just a placeholder because LiteLLM handles the real key in config.yaml.
client = AsyncOpenAI(base_url="http://localhost:4000", api_key="anything")

async def recruit_member(model_name: str, current_party_summary: str, judge_feedback: str = None):
    """
    Connects to an AI Model (like Grok) and asks it to create ONE new character.
    It passes the 'current_party_summary' so the AI knows who is already recruited.
    """
    
    available_classes = ", ".join(CLASS_ATTRIBUTES.keys())

    # These guidelines are injected into the AI's 'Brain' (Instructions)
    guidelines = (
        "BALANCED PARTY RULES (Target: 5 Members):\n"
        "The core party must cover these four roles. You may use the base class or its allowed replacements:\n"
        "1. HEALER: Cleric (Allowed Replacements: Bard, Druid)\n"
        "2. FRONTLINE: Fighter (Allowed Replacements: Barbarian, Monk, Paladin, Ranger)\n"
        "3. STEALTH/UTILITY: Rogue (Allowed Replacements: Bard, Ranger)\n"
        "4. ARCANE: Wizard (Allowed Replacements: Bard, Sorcerer, Warlock)\n"
        "5. FLEX PATHS (Pick ONE for the 5th member once others are filled):\n"
        "   - The Redundancy: A second Fighter or Paladin\n"
        "   - The Utility: A Bard or Druid\n"
        "   - The Specialist: A Warlock or Monk\n"
    )

    recruiter_instructions = (
        f"{guidelines}\n"
        "You are a D&D character recruiter. Review the current party and "
        "choose a Name, Class, and Backstory for a new Level 1 character that fills a missing role.\n"
        f"Available Classes: {available_classes}\n"
        f"Current Party: {current_party_summary}"
    )
    
    # If the Judge hated the previous party, we tell the recruiter WHY here.
    if judge_feedback:
        recruiter_instructions += f"\nJUDGE'S FEEDBACK FROM LAST ATTEMPT: {judge_feedback}"

    # Define the Agent
    recruiter = Agent(
        name="Recruiter",
        instructions=recruiter_instructions,
        model=OpenAIChatCompletionsModel(model=model_name, openai_client=client),
        output_type=CharacterIdentity # This ensures the AI gives us a structured object, not just text
    )

    # We use 'run_streamed' so we can watch the AI 'think' in real-time
    stream = Runner.run_streamed(
        recruiter, 
        input="Choose a character to join the party.",
        run_config=RunConfig(tracing_disabled=True)
    )

    print("\n--- RECRUITER IS THINKING ---")
    async for event in stream.stream_events():
        if event.type == "raw_response_event":
            # This captures the 'Reasoning' chunks from models like Grok
            chunk = getattr(event.data, 'delta', None)
            if chunk:
                print(chunk, end="", flush=True)
                
    # Wait for the stream to finish completely
    while not stream.is_complete:
        await asyncio.sleep(0.1)
        
    return stream.final_output

async def evaluate_party(model_name: str, party_list: list):
    """
    This is the 'Critic' step. A second AI agent looks at the final team of 5 
    and decides if it's actually balanced.
    """
    party_str = "\n".join([f"- {p.identity.name}: {p.identity.actual_class}" for p in party_list])
    
    judge = Agent(
        name="JudgeAgent",
        instructions=(
            "You are a D&D Party Judge. Strictly evaluate the party against the Balanced Party Standard. "
            "If the party is perfectly balanced, set is_valid to True. "
            "If it is missing a core role (Healer, Frontline, Stealth, Arcane), set is_valid to False."
        ),
        model=OpenAIChatCompletionsModel(model=model_name, openai_client=client),
        output_type=EvaluationResult
    )
    
    print("\n--- JUDGE IS EVALUATING THE PARTY ---")
    result = await Runner.run(judge, input=f"Final Party Proposal:\n{party_str}", run_config=RunConfig(tracing_disabled=True))
    return result.final_output

async def main():
    """The main entry point of the script."""
    
    # --- PHASE 1: Try to load from file ---
    if os.path.exists('party_state.json'):
        print("\n--- LOADING EXISTING FELLOWSHIP ---")
        party = load_party('party_state.json')
        if party:
            print(f"✅ Successfully loaded {len(party)} heroes from file.")
            for p in party:
                print(f"[{p.identity.actual_class}] {p.identity.name} | HP: {p.hp} | AC: {p.ac}")
            return

    # --- PHASE 2: Recruitment Loop (Critic/Refinement) ---
    max_attempts = 3
    attempt = 1
    judge_feedback = None
    
    while attempt <= max_attempts:
        print(f"\n--- ATTEMPT {attempt} of {max_attempts} ---")
        party = []
        party_summary = "Empty Party"
        
        # Build the party one by one
        for i in range(1, 6):
            print(f"\n--- RECRUITING PLAYER {i} ---")
            choice = await recruit_member("grok-agent", party_summary, judge_feedback)
            stats = CLASS_ATTRIBUTES.get(choice.actual_class)
            
            # Create the actual character object
            new_hero = create_pc_from_choice(choice, stats)
            party.append(new_hero)
            
            print(f"\nSUCCESS: {new_hero.identity.name} the {new_hero.identity.actual_class} has joined!")
            
            # Update the summary so the next character chose knows who is already there
            party_summary = "\n".join([f"- {p.identity.name}: {p.identity.actual_class}" for p in party])
        
        # Ask the Judge if the team is good
        eval_result = await evaluate_party("grok-agent", party)
        
        if eval_result.is_valid:
            print(f"\n✅ JUDGE APPROVED: {eval_result.feedback}")
            save_party(party, 'party_state.json')
            print("✅ Fellowship Saved to party_state.json.")
            break
        else:
            print(f"\n❌ JUDGE REJECTED: {eval_result.feedback}")
            judge_feedback = eval_result.feedback # Save the feedback to pass back to the recruiter
            attempt += 1

    print("\n--- FINAL PARTY ASSEMBLED ---")
    for p in party:
        print(f"[{p.identity.actual_class}] {p.identity.name} | HP: {p.hp} | AC: {p.ac}")

if __name__ == "__main__":
    asyncio.run(main())