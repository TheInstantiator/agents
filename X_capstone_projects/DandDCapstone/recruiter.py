import asyncio
import json
import os
import litellm
import sys

from dotenv import load_dotenv

load_dotenv(override=True)

if os.getenv("GROK_API_KEY") and not os.getenv("XAI_API_KEY"):
    os.environ["XAI_API_KEY"] = os.getenv("GROK_API_KEY")

# We import our own custom modules here
from factory import CharacterIdentity, create_pc_from_choice, EvaluationResult, SlotReplacement, save_party, load_party
from constants import CLASS_ATTRIBUTES

# Get the absolute path to this script's directory for robust file loading
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Add parent dir to sys.path so we can import core_agent
sys.path.append(os.path.join(SCRIPT_DIR, ".."))
from core_agent import CoreAgent

# Disable telemetry
litellm.telemetry = False

# Globals for cost tracking
TOTAL_COST = 0.0
TOTAL_TOKENS = 0
TOTAL_TIME = 0.0

def get_litellm_params(agent_name: str) -> dict:
    """Read config.json and return the litellm_params for the given agent."""
    # config.json is now shared and stays in the parent directory
    config_path = os.path.join(SCRIPT_DIR, "..", "config.json")
    
    with open(config_path, "r") as f:
        config = json.load(f)
        for model in config.get("model_list", []):
            if model.get("model_name") == agent_name:
                params = model.get("litellm_params", {}).copy()
                
                # Resolve environment variables manually if they look like "os.environ/VAR"
                for k, v in params.items():
                    if isinstance(v, str) and v.startswith("os.environ/"):
                        env_var = v.split("/")[1]
                        params[k] = os.getenv(env_var, "")
                return params
    raise ValueError(f"Agent {agent_name} not found in {config_path}")


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
        "You are an experienced D&D 5e player about to start a new campaign. "
        "You have a deep understanding of party composition, role balance, class mechanics, "
        "and how different races/species and backgrounds synergize with classes to create highly optimized and flavorfully coherent characters.\n\n"
        f"{guidelines}\n"
        "Your task is to review the current party and recruit a new Level 1 character to fill a missing role or satisfy the Flex path.\n"
        "When designing the character, select a species/race and background that have excellent synergy (both mechanically and narratively) with the class you choose.\n"
        "For example, a Dwarf or Half-Orc makes a fantastic hardy Cleric or Frontline fighter, an Elf or Halfling makes a nimble Rogue, and a High Elf or Tiefling synergizes beautifully with Arcane classes.\n\n"
        "IMPORTANT: You MUST select an `actual_class` EXACTLY as it appears in the Available Classes list below.\n"
        f"Available Classes: {available_classes}\n"
        f"Current Party Composition:\n{current_party_summary}\n\n"
        "In your response, explain your tactical reasoning in the 'thoughts' field: why you chose this class for the party, and why the race/species and background synergize well with it."
    )
    
    if judge_feedback:
        recruiter_instructions += f"\nJUDGE'S FEEDBACK FROM LAST ATTEMPT: {judge_feedback}"

    litellm_kwargs = get_litellm_params(agent_name)
    litellm_kwargs["temperature"] = 0.5
    
    agent = CoreAgent(
        agent_id=f"recruiter_{agent_name}",
        system_prompt=recruiter_instructions,
        litellm_kwargs=litellm_kwargs,
        one_shot=True,
        response_model=CharacterIdentity
    )
    
    print("\n--- RECRUITER IS THINKING ---")
    response_obj = await agent.ask("Choose a character to join the party.")
    
    global TOTAL_COST, TOTAL_TOKENS, TOTAL_TIME
    TOTAL_COST += agent.cost
    TOTAL_TOKENS += agent.tokens_used
    TOTAL_TIME += agent.total_response_time
    
    return response_obj

async def evaluate_party(agent_name: str, party_list: list) -> EvaluationResult:
    """
    This is the 'Critic' step. A second AI agent looks at the final team of 5 
    and decides if it's actually balanced.
    """
    party_str = "\n".join([f"- Slot {idx}: {p.identity.name} ({p.identity.actual_class})" for idx, p in enumerate(party_list)])
    
    litellm_kwargs = get_litellm_params(agent_name)
    litellm_kwargs["temperature"] = 0.1
    
    agent = CoreAgent(
        agent_id=f"judge_{agent_name}",
        system_prompt=(
            "You are a D&D Party Judge. Strictly evaluate the party against the Balanced Party Standard.\n"
            "IMPORTANT: Do NOT evaluate based on the order of the list. Look at the party holistically to verify all 4 core roles are covered somewhere in the group.\n"
            "Balanced Party Standard:\n"
            "1. HEALER: Cleric (Allowed Replacements: Bard, Druid)\n"
            "2. FRONTLINE: Fighter (Allowed Replacements: Barbarian, Monk, Paladin, Ranger)\n"
            "3. STEALTH/UTILITY: Rogue (Allowed Replacements: Bard, Ranger)\n"
            "4. ARCANE: Wizard (Allowed Replacements: Bard, Sorcerer, Warlock)\n"
            "5. FLEX PATH: Any class that adds redundancy or utility.\n\n"
            "If the party has all 4 core roles covered (regardless of order), set is_valid to True and leave the replacements list empty.\n"
            "If it is genuinely missing a core role, set is_valid to False. Specify the index of a redundant or overlapping character "
            "to kick, and explain what role/class should fill it in the 'replacements' list."
        ),
        litellm_kwargs=litellm_kwargs,
        one_shot=True,
        response_model=EvaluationResult
    )
    
    print("\n--- JUDGE IS EVALUATING THE PARTY ---")
    response_obj = await agent.ask(f"Final Party Proposal:\n{party_str}")
    
    global TOTAL_COST, TOTAL_TOKENS, TOTAL_TIME
    TOTAL_COST += agent.cost
    TOTAL_TOKENS += agent.tokens_used
    TOTAL_TIME += agent.total_response_time
    
    return response_obj


async def main():
    party_file = os.path.join(SCRIPT_DIR, 'party_state.json')
    if os.path.exists(party_file):
        print("\n--- LOADING EXISTING FELLOWSHIP ---")
        party = load_party(party_file)
        if party:
            print(f"✅ Successfully loaded {len(party)} heroes from file.")
            for p in party:
                print(f"[{p.identity.actual_class} / {p.identity.species}] {p.identity.name} | HP: {p.hp} | AC: {p.ac} | Played By: {p.played_by}\n{p.identity.backstory}\n")
            return

    # If no party file, start recruitment
    max_attempts = 10
    attempt = 1
    party = []
    party_summary = "Empty Party"
    
    # The list of agents to cycle through for character creation
    agent_roster = [
        "gemma-agent",
        "qwen-agent",
        "gpt-oss-agent",
        "phi-r-agent"
    ]
    agent_index = 0

    print("\n--- INITIAL RECRUITMENT ---")
    while len(party) < 5:
        print(f"\n--- RECRUITING PLAYER {len(party) + 1} ---")
        current_agent = agent_roster[agent_index % len(agent_roster)]
        print(f"Assigning this task to: {current_agent}")
        agent_index += 1
        
        choice = await recruit_member(current_agent, party_summary, judge_feedback=None)
        print(f"\n[THOUGHTS]: {choice.thoughts}\n")
        stats = CLASS_ATTRIBUTES.get(choice.actual_class)
        if stats is None:
            print(f"❌ HALLUCINATION DETECTED: Agent chose '{choice.actual_class}' which is not a valid class. Retrying...")
            continue
            
        new_hero = create_pc_from_choice(choice, stats, played_by=current_agent)
        party.append(new_hero)
        
        print(f"SUCCESS: {new_hero.identity.name} the {new_hero.identity.actual_class} has joined!")
        party_summary = "\n".join([f"- {p.identity.name}: {p.identity.actual_class}" for p in party])

    while attempt <= max_attempts:
        # Evaluate the full party of 5
        eval_result = await evaluate_party("gemma-agent", party)
        print(f"\n[JUDGE THOUGHTS]: {eval_result.thoughts}")
        
        if eval_result.is_valid:
            print(f"\n✅ JUDGE APPROVED: {eval_result.feedback}")
            save_party(party, party_file)
            print("✅ Fellowship Saved to party_state.json.")
            break
        else:
            print(f"\n❌ JUDGE REJECTED: {eval_result.feedback}")
            attempt += 1
            
            if attempt > max_attempts:
                print("\n❌ Max attempts reached. Could not form a valid party.")
                break
                
            replacements = eval_result.replacements
            if not replacements:
                # If Judge didn't specify replacements, default fallback to replacing the last member
                print("No targeted slot replacements specified. Defaulting to replacing last member.")
                replacements = [SlotReplacement(
                    slot_index=4,
                    reason="No specific slot replacements provided. Default fallback.",
                    suggested_role="Any missing party role"
                )]
            
            print(f"\n--- ATTEMPT {attempt} ---")
            
            # Apply all requested replacements
            for rep in replacements:
                idx = rep.slot_index
                if idx < 0 or idx >= len(party):
                    print(f"⚠️ Warning: Judge returned out-of-bounds slot index: {idx}. Skipping.")
                    continue
                    
                kicked_member = party[idx]
                print(f"Kicking slot {idx}: {kicked_member.identity.name} the {kicked_member.identity.actual_class}. Reason: {rep.reason}")
                
                # Re-recruit for this specific slot index in-place
                current_agent = agent_roster[agent_index % len(agent_roster)]
                print(f"\n--- RE-RECRUITING FOR SLOT {idx} ---")
                print(f"Assigning this task to: {current_agent}")
                agent_index += 1
                
                # Exclude the slot currently being replaced from the summary so the recruiter can fill the gap
                party_summary = "\n".join([f"- Slot {i}: {p.identity.name} ({p.identity.actual_class})" for i, p in enumerate(party) if i != idx])
                slot_feedback = f"Replace slot {idx} (previously a {kicked_member.identity.actual_class}). The Judge rejected the party: '{eval_result.feedback}'. Goal for this slot: '{rep.suggested_role}'."
                
                choice = await recruit_member(current_agent, party_summary, slot_feedback)
                print(f"\n[THOUGHTS]: {choice.thoughts}\n")
                
                stats = CLASS_ATTRIBUTES.get(choice.actual_class)
                if stats is None:
                    print(f"❌ HALLUCINATION DETECTED: Agent chose '{choice.actual_class}' which is not a valid class. Skipping slot replacement for this attempt.")
                    continue
                    
                new_hero = create_pc_from_choice(choice, stats, played_by=current_agent)
                party[idx] = new_hero
                print(f"SUCCESS: {new_hero.identity.name} the {new_hero.identity.actual_class} has joined in slot {idx}!")
                
            party_summary = "\n".join([f"- Slot {i}: {p.identity.name} ({p.identity.actual_class})" for i, p in enumerate(party)])

    print("\n--- FINAL PARTY ASSEMBLED ---")
    for p in party:
        print(f"[{p.identity.actual_class}] {p.identity.name} | HP: {p.hp} | AC: {p.ac}")

    print(f"\n💰 Total LLM API Cost: ${TOTAL_COST:.6f} ({TOTAL_TOKENS} tokens used) | Total Time: {TOTAL_TIME:.2f}s")

if __name__ == "__main__":
    asyncio.run(main())