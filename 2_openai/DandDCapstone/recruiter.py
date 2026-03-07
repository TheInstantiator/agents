import asyncio
import json
import os
import yaml
import litellm
import sys

from dotenv import load_dotenv

load_dotenv(override=True)

if os.getenv("GROK_API_KEY") and not os.getenv("XAI_API_KEY"):
    os.environ["XAI_API_KEY"] = os.getenv("GROK_API_KEY")

# We import our own custom modules here
from factory import CharacterIdentity, create_pc_from_choice, EvaluationResult, save_party, load_party
from constants import CLASS_ATTRIBUTES

# Disable telemetry
litellm.telemetry = False

# Globals for cost tracking
TOTAL_COST = 0.0
TOTAL_TOKENS = 0

def get_litellm_params(agent_name: str) -> dict:
    """Read config.yaml and return the litellm_params for the given agent."""
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
        for model in config.get("model_list", []):
            if model.get("model_name") == agent_name:
                params = model.get("litellm_params", {}).copy()
                
                # Resolve environment variables manually if they look like "os.environ/VAR"
                for k, v in params.items():
                    if isinstance(v, str) and v.startswith("os.environ/"):
                        env_var = v.split("/")[1]
                        params[k] = os.getenv(env_var, "")
                return params
    raise ValueError(f"Agent {agent_name} not found in config.yaml")

def track_cost_and_tokens(response):
    global TOTAL_COST, TOTAL_TOKENS
    try:
        cost = litellm.completion_cost(completion_response=response)
        tokens = response.usage.total_tokens
        if cost:
            TOTAL_COST += cost
        if tokens:
            TOTAL_TOKENS += tokens
    except Exception:
        pass

async def recruit_member(agent_name: str, current_party_summary: str, judge_feedback: str = None) -> CharacterIdentity:
    """
    Connects to an AI Model (like Grok) and asks it to create ONE new character.
    """
    available_classes = ", ".join(CLASS_ATTRIBUTES.keys())

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
    
    if judge_feedback:
        recruiter_instructions += f"\nJUDGE'S FEEDBACK FROM LAST ATTEMPT: {judge_feedback}"

    litellm_kwargs = get_litellm_params(agent_name)
    litellm_kwargs["messages"] = [
        {"role": "system", "content": recruiter_instructions},
        {"role": "user", "content": "Choose a character to join the party."}
    ]
    litellm_kwargs["response_format"] = CharacterIdentity
    litellm_kwargs["temperature"] = 0.5
    
    # We use streaming for the "thinking" effect if the model supports it, but since we are demanding a structured
    # response format via litellm Pydantic integration, it's easier to use normal completion.
    print("\n--- RECRUITER IS THINKING ---")
    response = await litellm.acompletion(**litellm_kwargs)
    
    track_cost_and_tokens(response)
    
    # Parse the response into our Pydantic model
    raw_response = response.choices[0].message.content
    return CharacterIdentity.model_validate_json(raw_response)

async def evaluate_party(agent_name: str, party_list: list) -> EvaluationResult:
    """
    This is the 'Critic' step. A second AI agent looks at the final team of 5 
    and decides if it's actually balanced.
    """
    party_str = "\n".join([f"- {p.identity.name}: {p.identity.actual_class}" for p in party_list])
    
    litellm_kwargs = get_litellm_params(agent_name)
    litellm_kwargs["messages"] = [
        {"role": "system", "content": (
            "You are a D&D Party Judge. Strictly evaluate the party against the Balanced Party Standard. "
            "If the party is perfectly balanced, set is_valid to True. "
            "If it is missing a core role (Healer, Frontline, Stealth, Arcane), set is_valid to False."
        )},
        {"role": "user", "content": f"Final Party Proposal:\n{party_str}"}
    ]
    litellm_kwargs["response_format"] = EvaluationResult
    litellm_kwargs["temperature"] = 0.1
    
    print("\n--- JUDGE IS EVALUATING THE PARTY ---")
    response = await litellm.acompletion(**litellm_kwargs)
    
    track_cost_and_tokens(response)
    
    raw_response = response.choices[0].message.content
    return EvaluationResult.model_validate_json(raw_response)

async def main():
    if os.path.exists('party_state.json'):
        print("\n--- LOADING EXISTING FELLOWSHIP ---")
        party = load_party('party_state.json')
        if party:
            print(f"✅ Successfully loaded {len(party)} heroes from file.")
            for p in party:
                print(f"[{p.identity.actual_class} / {p.identity.species}] {p.identity.name} | HP: {p.hp} | AC: {p.ac} | Played By: {p.played_by}\n{p.identity.backstory}\n")
            return

    # If no party file, start recruitment
    # If we can't build a party in 3 attempts, give up something is wrong
    max_attempts = 3
    attempt = 1
    judge_feedback = None
    party = []
    
    while attempt <= max_attempts:
        print(f"\n--- ATTEMPT {attempt} of {max_attempts} ---")
        party = []
        party_summary = "Empty Party"
        
        # The list of agents to cycle through for character creation
        agent_roster = [
            "grok-agent-tier1-think",
            "gemini-agent-tier1-think",
            "gemini-agent-vanilla",
            "grok-agent-vanilla"
        ]

        # Recruit 5 members
        for i in range(1, 6):
            print(f"\n--- RECRUITING PLAYER {i} ---")
            
            # Select the next agent in the round-robin
            current_agent = agent_roster[(i - 1) % len(agent_roster)]
            print(f"Assigning this task to: {current_agent}")
            
            choice = await recruit_member(current_agent, party_summary, judge_feedback)
            stats = CLASS_ATTRIBUTES.get(choice.actual_class)
            
            new_hero = create_pc_from_choice(choice, stats, played_by=current_agent)
            party.append(new_hero)
            
            print(f"SUCCESS: {new_hero.identity.name} the {new_hero.identity.actual_class} has joined!")
            party_summary = "\n".join([f"- {p.identity.name}: {p.identity.actual_class}" for p in party])
        
        # Also updated the judge since "grok-agent" was removed from config.yaml
        eval_result = await evaluate_party("grok-agent-tier1-think", party)
        
        if eval_result.is_valid:
            print(f"\n✅ JUDGE APPROVED: {eval_result.feedback}")
            save_party(party, 'party_state.json')
            print("✅ Fellowship Saved to party_state.json.")
            break
        else:
            print(f"\n❌ JUDGE REJECTED: {eval_result.feedback}")
            judge_feedback = eval_result.feedback
            attempt += 1

    print("\n--- FINAL PARTY ASSEMBLED ---")
    for p in party:
        print(f"[{p.identity.actual_class}] {p.identity.name} | HP: {p.hp} | AC: {p.ac}")

    print(f"\n💰 Total LLM API Cost: ${TOTAL_COST:.6f} ({TOTAL_TOKENS} tokens used)")

if __name__ == "__main__":
    asyncio.run(main())